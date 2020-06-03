use crate::{
    common::*,
    convert::MyFrom,
    data::{GqnExample, GqnFeature},
    model::GqnModelInput,
    utils,
};
use futures::stream::{StreamExt, TryStream, TryStreamExt};
use itertools::Itertools;
use rand::Rng;
use std::convert::TryFrom;

pub mod deepmind {
    use super::*;

    pub const NUM_ORIG_CAMERA_PARAMS: usize = 5;
    pub const NUM_CAMERA_PARAMS: usize = 7;

    #[derive(Debug, Clone)]
    pub struct DatasetInit<P, D>
    where
        P: AsRef<Path>,
        D: AsRef<[Device]>,
    {
        pub frame_channels: NonZeroUsize,
        pub train_size: NonZeroUsize,
        pub test_size: NonZeroUsize,
        pub frame_size: usize,
        pub sequence_size: NonZeroUsize,
        pub dataset_dir: P,
        pub check_integrity: bool,
        pub devices: D,
        pub batch_size: NonZeroUsize,
    }

    impl<P, D> DatasetInit<P, D>
    where
        P: AsRef<Path>,
        D: AsRef<[Device]>,
    {
        pub async fn build(self) -> Fallible<Dataset> {
            // verify config
            let DatasetInit {
                frame_channels,
                train_size,
                test_size,
                frame_size,
                sequence_size,
                dataset_dir,
                check_integrity,
                devices,
                batch_size,
            } = self;

            let dataset_dir = dataset_dir.as_ref();
            let train_dir = dataset_dir.join("train");
            let test_dir = dataset_dir.join("test");
            let devices = devices.as_ref().to_owned();
            let num_devices = devices.len();
            ensure!(num_devices > 0, "no device specified");

            // list files
            let train_files: Vec<_> = glob::glob(train_dir.join("*.tfrecord").to_str().unwrap())?
                .take(1)
                .map(|p| p.unwrap())
                .collect();

            let test_files: Vec<_> = glob::glob(test_dir.join("*.tfrecord").to_str().unwrap())?
                .take(1)
                .map(|p| p.unwrap())
                .collect();

            // build tfrecord dataset
            let train_dataset = tfrecord::DatasetInit {
                check_integrity,
                ..Default::default()
            }
            .from_paths(&train_files)
            .await?;
            let test_dataset = tfrecord::DatasetInit {
                check_integrity,
                ..Default::default()
            }
            .from_paths(&test_files)
            .await?;

            let dataset = Dataset {
                train_size: train_size.get(),
                test_size: test_size.get(),
                frame_size: frame_size,
                sequence_size: sequence_size.get(),
                frame_channels: frame_channels.get(),
                batch_size: batch_size.get(),
                train_dataset,
                test_dataset,
            };

            Ok(dataset)
        }
    }

    #[derive(Debug, Clone)]
    pub struct Dataset {
        train_size: usize,
        test_size: usize,
        frame_size: usize,
        sequence_size: usize,
        frame_channels: usize,
        batch_size: usize,
        train_dataset: tfrecord::Dataset,
        test_dataset: tfrecord::Dataset,
    }

    impl Dataset {
        pub fn train_stream(
            &self,
            initial_step: usize,
        ) -> Fallible<impl TryStream<Ok = GqnModelInput, Error = Error> + Send> {
            let Dataset {
                sequence_size,
                frame_size,
                batch_size,
                train_dataset: dataset,
                ..
            } = self.clone();

            let num_records = dataset.num_records();

            // sample records randomly
            let stream = futures::stream::try_unfold(
                (dataset, OsRng::default()),
                move |(mut dataset, mut rng)| {
                    async move {
                        let index = rng.gen_range(0, num_records);
                        let example = dataset.get::<Example>(index).await?;
                        Fallible::Ok(Some((example, (dataset, rng))))
                    }
                },
            )
            .try_filter_map(|example_opt| async move { Fallible::Ok(example_opt) });

            // convert example type
            let stream = stream.map_ok(|example| GqnExample::my_from(example));

            // decode image
            let stream = stream.and_then(|in_example| {
                async move {
                    let out_example = async_std::task::spawn_blocking(|| {
                        utils::decode_image_on_example(
                            in_example,
                            hashmap!("frames".into() => None),
                        )
                    })
                    .await?;
                    Fallible::Ok(out_example)
                }
            });

            // preprocess and transform
            let stream = stream.and_then(move |in_example| {
                async move {
                    let out_example = async_std::task::spawn_blocking(move || {
                        preprocessor(in_example, sequence_size, frame_size)
                    })
                    .await?;
                    Fallible::Ok(out_example)
                }
            });

            // convert to tch tensors
            let stream = stream.and_then(move |in_example| {
                async move {
                    let out_example = async_std::task::spawn_blocking(move || {
                        convert_to_tensors(in_example, sequence_size, frame_size)
                    })
                    .await?;
                    Fallible::Ok(out_example)
                }
            });

            // group into batches
            let stream = stream.chunks(batch_size).then(move |batch_of_results| {
                async move {
                    let batch: HashMap<String, Tensor> = batch_of_results
                        .into_iter()
                        .collect::<Fallible<Vec<_>>>()?
                        .into_iter()
                        .flat_map(|example| example.into_iter())
                        .into_group_map()
                        .into_iter()
                        .map(|(name, features)| {
                            ensure!(features.len() == batch_size);
                            let tensors = features
                                .into_iter()
                                .map(|feature| match feature {
                                    GqnFeature::Tensor(tensor) => tensor,
                                    _ => unreachable!(),
                                })
                                .collect::<Vec<_>>();
                            let batch_tensor = Tensor::stack(&tensors, 0);
                            Ok((name, batch_tensor))
                        })
                        .collect::<Fallible<_>>()?;
                    Fallible::Ok(batch)
                }
            });

            // transform to model input type
            let stream = stream.scan(initial_step, move |step, result| {
                let curr_step = *step;
                *step += 1;

                async move {
                    let mut in_example = match result {
                        Ok(example) => example,
                        Err(err) => return Some(Err(err)),
                    };

                    let context_frames = in_example.remove("context_frames").unwrap();
                    let target_frame = in_example.remove("target_frame").unwrap();
                    let context_params = in_example.remove("context_params").unwrap();
                    let query_params = in_example.remove("query_params").unwrap();

                    let input = GqnModelInput {
                        context_frames,
                        target_frame,
                        context_params,
                        query_params,
                        step: curr_step,
                    };
                    Some(Fallible::Ok(input))
                }
            });

            Ok(stream)
        }

        pub fn test_stream(&self) {
            // Define test iterator
            // let test_devices = devices.clone();
            // let test_options = LoaderOptions {
            //     check_integrity: check_integrity,
            //     auto_close: false,
            //     parallel: true,
            //     open_limit: None,
            //     method: LoaderMethod::Mmap,
            // };
            // let test_loader = IndexedLoader::load_ex(test_files, test_options)?;
            // let test_iter = test_loader
            //     .index_iter()
            //     .cycle()
            //     .shuffle(8192)
            //     .load_by_tfrecord_index(test_loader)
            //     .unwrap_result()
            //     .par_map(|bytes| bytes_to_example(&bytes, None))
            //     .unwrap_result()
            //     .par_map(image_decoder)
            //     .unwrap_result()
            //     .par_map(preprocessor)
            //     .batching(move |it| {
            //         let mut buf = Vec::new();

            //         while buf.len() < batch_size {
            //             match it.next() {
            //                 Some(example) => buf.push(example),
            //                 None => break,
            //             }
            //         }

            //         if buf.is_empty() {
            //             None
            //         } else {
            //             Some(make_batch(buf))
            //         }
            //     })
            //     .unwrap_result()
            //     .prefetch(512)
            //     .enumerate()
            //     .par_map(move |(idx, example)| {
            //         let dev_idx = idx % test_devices.len();
            //         let device = test_devices[dev_idx];
            //         let new_example = match example_to_torch_tensor(example, None, device) {
            //             Ok(ret) => ret,
            //             Err(err) => return Err(err),
            //         };
            //         Ok(new_example)
            //     })
            //     .unwrap_result()
            //     .batching(move |it| {
            //         // Here assumes par_map() is ordered
            //         let mut buf = vec![];
            //         while buf.len() < num_devices {
            //             match it.next() {
            //                 Some(example) => buf.push(example),
            //                 None => break,
            //             }
            //         }

            //         if buf.is_empty() {
            //             None
            //         } else {
            //             Some(buf)
            //         }
            //     });
            todo!();
        }
    }

    fn preprocessor(
        mut example: GqnExample,
        sequence_size: usize,
        frame_size: usize,
    ) -> Fallible<GqnExample> {
        // extract features of interest
        let (_camera_key, cameras) = {
            let (key, value) = example.remove_entry("cameras").unwrap();
            match value {
                GqnFeature::FloatList(list) => (key, list),
                _ => unreachable!(),
            }
        };
        let (_frames_key, frames) = {
            let (key, value) = example.remove_entry("frames").unwrap();
            match value {
                GqnFeature::DynamicImageList(list) => (key, list),
                _ => unreachable!(),
            }
        };

        // Process camera data
        let (context_cameras, query_camera) = {
            ensure!(
                sequence_size * NUM_ORIG_CAMERA_PARAMS == cameras.len(),
                "invalid size",
            );

            // transform parameters
            let mut context_cameras = cameras
                .chunks(NUM_ORIG_CAMERA_PARAMS)
                .map(|chunk| {
                    let (x, y, z, yaw, pitch) = match chunk {
                        &[x, y, z, yaw, pitch] => (x, y, z, yaw, pitch),
                        _ => unreachable!(),
                    };

                    let yaw_cos = yaw.cos();
                    let yaw_sin = yaw.sin();
                    let pitch_cos = pitch.cos();
                    let pitch_sin = pitch.sin();
                    vec![x, y, z, yaw_sin, yaw_cos, pitch_sin, pitch_cos]
                })
                .collect::<Vec<_>>();

            // take the last params
            let query_camera = context_cameras.pop().unwrap();

            (context_cameras, query_camera)
        };

        // Process frame data
        let (context_frames, target_frame) = {
            ensure!(sequence_size == frames.len(), "invalid size");

            // convert to rgb images
            let mut context_frames = frames
                .into_iter()
                .map(|frame| {
                    let rgb_frame = frame.into_rgb();
                    ensure!(
                        rgb_frame.height() as usize == frame_size
                            && rgb_frame.width() as usize == frame_size
                    );
                    Ok(rgb_frame)
                })
                .collect::<Fallible<Vec<_>>>()?;

            // take the last image
            let target_frame = context_frames.pop().unwrap();

            (context_frames, target_frame)
        };

        // Save features
        example.insert("context_frames".into(), context_frames.into());
        example.insert("target_frame".into(), target_frame.into());
        example.insert("context_params".into(), context_cameras.into());
        example.insert("query_params".into(), query_camera.into());

        Ok(example)
    }

    fn convert_to_tensors(
        in_example: GqnExample,
        sequence_size: usize,
        frame_size: usize,
    ) -> Fallible<GqnExample> {
        let out_example = in_example
            .into_iter()
            .map(|(key, value)| {
                let tensor = match value {
                    GqnFeature::FloatList(list) => Tensor::from(list.as_slice()),
                    GqnFeature::FloatsList(list) => {
                        let array = Array2::from_shape_vec(
                            (sequence_size - 1, NUM_CAMERA_PARAMS),
                            list.into_iter().flatten().collect::<Vec<_>>(),
                        )?;
                        Tensor::try_from(array)?
                    }
                    GqnFeature::RgbImage(image) => {
                        let array =
                            Array3::from_shape_vec((frame_size, frame_size, 3), image.into_vec())?
                                .permuted_axes([2, 0, 1]) // HWC to CHW
                                .map(|component| *component as f32 / 255.0);
                        Tensor::try_from(array)?
                    }
                    GqnFeature::RgbImageList(list) => {
                        let tensors = list
                            .into_iter()
                            .map(|image| {
                                let array = Array3::from_shape_vec(
                                    (frame_size, frame_size, 3),
                                    image.into_vec(),
                                )?
                                .permuted_axes([2, 0, 1]) // HWC to CHW
                                .map(|component| *component as f32 / 255.0);
                                let tensor = Tensor::try_from(array)?;
                                Ok(tensor)
                            })
                            .collect::<Fallible<Vec<_>>>()?;
                        Tensor::stack(&tensors, 0)
                    }
                    _ => unreachable!(),
                };
                Ok((key, tensor.into()))
            })
            .collect::<Fallible<GqnExample>>()?;

        Ok(out_example)
    }
}

// pub mod file {
//     use super::*;

//     pub struct Dataset {
//         pub param_channels: i64,
//         pub frame_channels: i64,
//         // pub train_iter: Box<dyn Iterator<Item = Vec<Example>> + 'a>,
//     }

//     impl Dataset {
//         pub fn load<P: AsRef<Path>>(
//             input_dir: P,
//             batch_size: usize,
//             sequence_size: usize,
//             frame_size: usize,
//             time_step: f32,
//             devices: Vec<Device>,
//         ) -> Fallible<Self> {
//             ensure!(sequence_size >= 2, "sequence_size should be at least 2");

//             let num_devices = devices.len();
//             let train_devices = devices.clone();
//             let mut image_paths = vec![];
//             let regex_filename = Regex::new(r"^frame-\d+.png$")?;
//             for entry_result in fs::read_dir(input_dir.as_ref())? {
//                 let entry = entry_result?;
//                 let filename = entry.file_name();
//                 if entry.file_type()?.is_file()
//                     && regex_filename.is_match(filename.to_str().unwrap())
//                 {
//                     image_paths.push(entry.path());
//                 }
//             }
//             ensure!(!image_paths.is_empty(), "Directory is empty");
//             image_paths.sort();

//             let timestamp_file = input_dir.as_ref().join("timestamps.txt");
//             let mut timestamps = vec![];
//             let reader = BufReader::new(File::open(timestamp_file)?);
//             for line in reader.lines() {
//                 let ts: f32 = line?.parse()?;
//                 timestamps.push(ts / 1_000_000.0);
//             }
//             ensure!(
//                 image_paths.len() == timestamps.len(),
//                 "Image files and timestamp mismatch"
//             );

//             // let trunc_size = image_paths.len() / sequence_size * sequence_size;

//             let samples: Vec<_> = timestamps
//                 .into_iter()
//                 .zip(image_paths.into_iter())
//                 .batching(move |it| {
//                     let mut buf = vec![];
//                     let mut expect_ts = 0.;

//                     loop {
//                         match it.next() {
//                             Some((ts, path)) => {
//                                 if buf.is_empty() {
//                                     buf.push((ts, path));
//                                     expect_ts = ts + time_step;
//                                 } else if ts >= expect_ts {
//                                     if ts - expect_ts >= time_step / 4. {
//                                         buf.clear();
//                                     } else {
//                                         buf.push((ts, path));
//                                         expect_ts += time_step;
//                                     }
//                                 }

//                                 if buf.len() == sequence_size {
//                                     return Some(buf);
//                                 }
//                             }
//                             None => return None,
//                         }
//                     }
//                 })
//                 .into_iter()
//                 .map(|chunk| {
//                     let samples: Vec<_> = chunk.into_iter().collect();
//                     let (first_ts, _) = samples[0];
//                     let samples: Vec<_> = samples
//                         .into_iter()
//                         .map(|(ts, path)| (ts - first_ts, path))
//                         .collect();
//                     samples
//                 })
//                 .concat()
//                 .into_iter()
//                 .collect();

//             let train_iter = samples
//                 .into_iter()
//                 .cycle()
//                 .par_map(move |(offset, path)| {
//                     let mut file = File::open(&path)?;
//                     let mut buf = vec![];
//                     let size = file.read_to_end(&mut buf)?;
//                     ensure!(size > 0, format!("{:?} is empty file", &path));

//                     let orig_image = image::load_from_memory_with_format(&buf, ImageFormat::PNG)?;
//                     let resized_image = orig_image.resize(
//                         frame_size as u32,
//                         frame_size as u32,
//                         FilterType::CatmullRom,
//                     );
//                     let new_width = resized_image.width() as usize;
//                     let new_height = resized_image.height() as usize;
//                     let pixels = resized_image.to_rgb().into_vec();
//                     let array = Array3::from_shape_vec((new_height, new_width, 3), pixels)?;

//                     let off = offset as f32 / 1000_f32; // ns to ms
//                     Ok((off, array))
//                 })
//                 .unwrap_result()
//                 .batching(move |it| {
//                     let mut offsets = vec![];
//                     let mut bufs = vec![];

//                     while offsets.len() < sequence_size {
//                         match it.next() {
//                             Some((off, buf)) => {
//                                 offsets.push(off);
//                                 bufs.push(buf);
//                             }
//                             None => {
//                                 assert!(offsets.is_empty());
//                                 break;
//                             }
//                         }
//                     }

//                     if offsets.is_empty() {
//                         None
//                     } else {
//                         let mut example: Example = HashMap::new();
//                         example.insert("offsets".to_owned(), Box::new(offsets));
//                         example.insert("frames".to_owned(), Box::new(bufs));
//                         Some(example)
//                     }
//                 })
//                 .par_map(move |mut example| {
//                     let (_, mut offsets_ref) = example.remove_entry("offsets").unwrap();

//                     let offsets = offsets_ref.downcast_mut::<Vec<f32>>().unwrap();

//                     let query_offset = offsets.pop().unwrap();
//                     let query_offset_array =
//                         Array2::from_shape_vec((1, 1), vec![query_offset]).unwrap();
//                     let context_offsets_array =
//                         Array2::from_shape_vec((sequence_size - 1, 1), offsets.to_vec()).unwrap();

//                     let mut frames = example
//                         .remove_entry("frames")
//                         .unwrap()
//                         .1
//                         .downcast_ref::<Vec<Array3<u8>>>()
//                         .unwrap()
//                         .into_iter()
//                         .map(|array| array.mapv(|val| val as f32 / 255.).permuted_axes([2, 0, 1]))
//                         .collect::<Vec<_>>();

//                     let target_frame = frames.pop().unwrap();
//                     let frames_expanded: Vec<_> = frames
//                         .into_iter()
//                         .map(|array| array.insert_axis(Axis(0)))
//                         .collect();
//                     let frame_views: Vec<_> =
//                         frames_expanded.iter().map(|array| array.view()).collect();

//                     let context_frames = ndarray::stack(Axis(0), &frame_views).unwrap();

//                     let mut new_example: Example = HashMap::new();
//                     new_example.insert("context_frames".to_owned(), Box::new(context_frames));
//                     new_example.insert("target_frame".to_owned(), Box::new(target_frame));
//                     new_example
//                         .insert("context_params".to_owned(), Box::new(context_offsets_array));
//                     new_example.insert("query_params".to_owned(), Box::new(query_offset_array));

//                     new_example
//                 })
//                 .batching(move |it| {
//                     let mut buf = Vec::new();

//                     while buf.len() < batch_size {
//                         match it.next() {
//                             Some(example) => {
//                                 buf.push(example);
//                             }
//                             None => break,
//                         }
//                     }

//                     if buf.is_empty() {
//                         None
//                     } else {
//                         Some(make_batch(buf))
//                     }
//                 })
//                 .unwrap_result()
//                 .enumerate()
//                 .par_map(move |(idx, example)| {
//                     let dev_idx = idx % train_devices.len();
//                     let device = train_devices[dev_idx];
//                     example_to_torch_tensor(example, None, device)
//                 })
//                 .unwrap_result()
//                 .prefetch(512)
//                 .batching(move |it| {
//                     // Here assumes par_map() is ordered
//                     let mut buf = vec![];
//                     while buf.len() < num_devices {
//                         match it.next() {
//                             Some(example) => buf.push(example),
//                             None => break,
//                         }
//                     }

//                     if buf.is_empty() {
//                         None
//                     } else {
//                         Some(buf)
//                     }
//                 });

//             let dataset = Self {
//                 frame_channels: 3,
//                 param_channels: 1,
//                 // train_iter: Box::new(train_iter),
//             };

//             Ok(dataset)
//         }
//     }

// }
