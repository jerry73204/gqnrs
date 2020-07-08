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
        pub async fn build(self) -> Result<Dataset> {
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
        ) -> Result<impl Stream<Item = Result<GqnModelInput>> + Send> {
            let Dataset {
                sequence_size,
                frame_size,
                batch_size,
                train_dataset: dataset,
                ..
            } = self.clone();

            let num_records = dataset.num_records();
            if num_records == 0 {
                bail!("dataset is empty");
            }

            // sample records randomly
            let stream = futures::stream::try_unfold(
                (dataset, OsRng::default()),
                move |(mut dataset, mut rng)| async move {
                    let index = rng.gen_range(0, num_records);
                    let example = dataset.get::<Example>(index).await?;
                    Result::Ok(Some((example, (dataset, rng))))
                },
            )
            .try_filter_map(|example_opt| async move { Result::Ok(example_opt) });

            // convert example type
            let stream = stream.map_ok(|example| GqnExample::my_from(example));

            // decode image
            let stream = stream.and_then(|in_example| async move {
                let out_example = async_std::task::spawn_blocking(|| {
                    utils::decode_image_on_example(in_example, hashmap!("frames".into() => None))
                })
                .await?;
                Result::Ok(out_example)
            });

            // preprocess and transform
            let stream = stream.and_then(move |in_example| async move {
                let out_example = async_std::task::spawn_blocking(move || {
                    preprocessor(in_example, sequence_size, frame_size)
                })
                .await?;
                Result::Ok(out_example)
            });

            // convert to tch tensors
            let stream = stream.and_then(move |in_example| async move {
                let out_example = async_std::task::spawn_blocking(move || {
                    convert_to_tensors(in_example, sequence_size, frame_size)
                })
                .await?;
                Result::Ok(out_example)
            });

            // group into batches
            let stream = stream
                .chunks(batch_size)
                .then(move |batch_of_results| async move {
                    let batch: HashMap<String, Tensor> = batch_of_results
                        .into_iter()
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flat_map(|example| example.into_iter())
                        .into_group_map()
                        .into_iter()
                        .map(|(name, features)| {
                            debug_assert_eq!(features.len(), batch_size);
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
                        .collect::<Result<_>>()?;
                    Result::Ok(batch)
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
                    Some(Result::Ok(input))
                }
            });

            Ok(stream)
        }

        pub fn test_stream(&self) -> Result<impl TryStream<Ok = GqnModelInput, Error = Error>> {
            let Dataset {
                sequence_size,
                frame_size,
                batch_size,
                test_dataset: dataset,
                ..
            } = self.clone();

            let stream = dataset.stream::<Example>();

            // convert example type
            let stream = stream
                .map_ok(|example| GqnExample::my_from(example))
                .err_into::<Error>();

            // decode image
            let stream = stream.and_then(|in_example| async move {
                let out_example = async_std::task::spawn_blocking(|| {
                    utils::decode_image_on_example(in_example, hashmap!("frames".into() => None))
                })
                .await?;
                Result::Ok(out_example)
            });

            // preprocess and transform
            let stream = stream.and_then(move |in_example| async move {
                let out_example = async_std::task::spawn_blocking(move || {
                    preprocessor(in_example, sequence_size, frame_size)
                })
                .await?;
                Result::Ok(out_example)
            });

            // convert to tch tensors
            let stream = stream.and_then(move |in_example| async move {
                let out_example = async_std::task::spawn_blocking(move || {
                    convert_to_tensors(in_example, sequence_size, frame_size)
                })
                .await?;
                Result::Ok(out_example)
            });

            // group into batches
            let stream = stream
                .chunks(batch_size)
                .then(move |batch_of_results| async move {
                    let batch: HashMap<String, Tensor> = batch_of_results
                        .into_iter()
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flat_map(|example| example.into_iter())
                        .into_group_map()
                        .into_iter()
                        .map(|(name, features)| {
                            debug_assert_eq!(features.len(), batch_size);
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
                        .collect::<Result<_>>()?;
                    Result::Ok(batch)
                });

            // transform to model input type
            let stream = stream.map_ok(move |mut in_example| {
                let context_frames = in_example.remove("context_frames").unwrap();
                let target_frame = in_example.remove("target_frame").unwrap();
                let context_params = in_example.remove("context_params").unwrap();
                let query_params = in_example.remove("query_params").unwrap();

                let input = GqnModelInput {
                    context_frames,
                    target_frame,
                    context_params,
                    query_params,
                    step: 0,
                };
                input
            });

            Ok(stream)
        }
    }

    fn preprocessor(
        mut example: GqnExample,
        sequence_size: usize,
        frame_size: usize,
    ) -> Result<GqnExample> {
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
                    debug_assert_eq!(rgb_frame.height() as usize, frame_size);
                    debug_assert_eq!(rgb_frame.width() as usize, frame_size);
                    Ok(rgb_frame)
                })
                .collect::<Result<Vec<_>>>()?;

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
    ) -> Result<GqnExample> {
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
                        Tensor::try_from(array).map_err(|err| format_err!("{:?}", err))?
                    }
                    GqnFeature::RgbImage(image) => {
                        let array =
                            Array3::from_shape_vec((frame_size, frame_size, 3), image.into_vec())?
                                .permuted_axes([2, 0, 1]) // HWC to CHW
                                .map(|component| *component as f32 / 255.0)
                                .as_standard_layout()
                                .to_owned();
                        Tensor::try_from(array).map_err(|err| format_err!("{:?}", err))?
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
                                .map(|component| *component as f32 / 255.0)
                                .as_standard_layout()
                                .to_owned();
                                let tensor = Tensor::try_from(array)
                                    .map_err(|err| format_err!("{:?}", err))?;
                                Ok(tensor)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Tensor::stack(&tensors, 0)
                    }
                    _ => unreachable!(),
                };
                Ok((key, tensor.into()))
            })
            .collect::<Result<GqnExample>>()?;

        Ok(out_example)
    }
}
