use failure::Fallible;
use image::{FilterType, GenericImageView, ImageFormat};
use itertools::Itertools;
use ndarray::{Array2, Array3, ArrayBase, Axis};
use par_map::ParMap;
use regex::Regex;
use std::collections::HashMap;
use std::fs::{read_dir, File};
use std::io::{prelude::*, BufReader};
use std::path::Path;
use tch::{Device, Tensor};
use tfrecord_rs::iter::DsIterator;
use tfrecord_rs::loader::{IndexedLoader, Loader, LoaderMethod, LoaderOptions};
use tfrecord_rs::utils::{
    bytes_to_example, decode_image_on_example, example_to_torch_tensor, make_batch,
};
use tfrecord_rs::ExampleType;
use yaml_rust::YamlLoader;

pub struct DeepMindDataSet<'a> {
    pub name: String,
    pub train_size: i64,
    pub test_size: i64,
    pub frame_size: i64,
    pub sequence_size: i64,
    pub frame_channels: i64,
    pub param_channels: i64,
    pub train_iter: Box<Iterator<Item = Vec<ExampleType>> + 'a>,
    pub test_iter: Box<Iterator<Item = Vec<ExampleType>> + 'a>,
}

pub struct FileDataset<'a> {
    pub param_channels: i64,
    pub frame_channels: i64,
    pub train_iter: Box<Iterator<Item = Vec<ExampleType>> + 'a>,
}

impl<'a> DeepMindDataSet<'a> {
    pub fn load_dir<P: AsRef<Path>>(
        name: &str,
        dataset_dir: P,
        check_integrity: bool,
        devices: Vec<Device>,
        batch_size: usize,
    ) -> Fallible<DeepMindDataSet<'a>> {
        // Load dataset config
        let dataset_spec = &YamlLoader::load_from_str(include_str!("dataset.yaml"))?[0];
        let dataset_info = &dataset_spec["dataset"][name];
        let train_size = dataset_info["train_size"].as_i64().unwrap();
        let test_size = dataset_info["test_size"].as_i64().unwrap();
        let frame_size = dataset_info["frame_size"].as_i64().unwrap();
        let sequence_size = dataset_info["sequence_size"].as_i64().unwrap();
        let frame_channels = dataset_info["frame_channels"].as_i64().unwrap();
        let num_camera_params = dataset_info["num_camera_params"].as_i64().unwrap();
        let num_devices = devices.len();

        // Sanity check
        ensure!(train_size.is_positive(), "train_size must be positive");
        ensure!(test_size.is_positive(), "test_size must be positive");
        ensure!(frame_size.is_positive(), "frame_size must be positive");
        ensure!(sequence_size >= 2, "frame_size must be at least 2");
        ensure!(
            frame_channels.is_positive(),
            "frame_channels must be positive"
        );
        ensure!(
            num_camera_params.is_positive(),
            "num_camera_params must be positive"
        );
        ensure!(num_devices > 0, "num_devices must be positive");

        // List files
        let train_dir = dataset_dir.as_ref().join("train");
        let test_dir = dataset_dir.as_ref().join("test");

        let train_files: Vec<_> = glob::glob(train_dir.join("*.tfrecord").to_str().unwrap())?
            .take(1)
            .map(|p| p.unwrap())
            .collect();

        let test_files: Vec<_> = glob::glob(test_dir.join("*.tfrecord").to_str().unwrap())?
            .take(1)
            .map(|p| p.unwrap())
            .collect();

        // Data processors
        let preprocessor = move |mut example: ExampleType| -> ExampleType {
            let (_, cameras_ref) = example.remove_entry("cameras").unwrap();
            let (_, mut frames_ref) = example.remove_entry("frames").unwrap();

            // Process camera data
            let cameras = cameras_ref.downcast_ref::<Vec<f32>>().unwrap();
            let (context_cameras, query_camera) = {
                assert!((sequence_size * num_camera_params) as usize == cameras.len());

                let orig_array: Box<Array2<f32>> = Box::new(
                    ArrayBase::from_shape_vec(
                        (sequence_size as usize, num_camera_params as usize),
                        cameras.to_owned(),
                    )
                    .unwrap(),
                );

                let pos = orig_array.slice(s![.., 0..3]);
                let yaw = orig_array.slice(s![.., 3..4]);
                let pitch = orig_array.slice(s![.., 4..5]);

                let yaw_cos = yaw.mapv(|v| v.cos());
                let yaw_sin = yaw.mapv(|v| v.sin());
                let pitch_cos = pitch.mapv(|v| v.cos());
                let pitch_sin = pitch.mapv(|v| v.sin());

                let new_array = stack![Axis(1), pos, yaw_sin, yaw_cos, pitch_sin, pitch_cos];
                let context_cameras = new_array
                    .slice(s![..(sequence_size as usize - 1), ..])
                    .to_owned();
                let query_camera = new_array
                    .slice(s![sequence_size as usize - 1, ..])
                    .to_owned();
                (context_cameras, query_camera)
            };

            // Process frame data
            let mut frames = frames_ref
                .downcast_mut::<Vec<Array3<u8>>>()
                .unwrap()
                .iter()
                .map(|array| array.mapv(|val| val as f32 / 255.).permuted_axes([2, 0, 1]))
                .collect::<Vec<_>>();

            let (context_frames, target_frame) = {
                let shape = frames[0].shape();
                let channels = shape[0];
                let height = shape[1];
                let width = shape[2];

                assert!(
                    sequence_size as usize == frames.len()
                        && width == frame_size as usize
                        && height == frame_size as usize
                        && channels == 3
                );

                let target_frame = frames.pop().unwrap();
                let frames_expanded = frames
                    .into_iter()
                    .map(|array| array.insert_axis(Axis(0)))
                    .collect::<Vec<_>>();

                let frame_views = frames_expanded
                    .iter()
                    .map(|array| array.view())
                    .collect::<Vec<_>>();

                let context_frames = ndarray::stack(Axis(0), &frame_views).unwrap();
                (context_frames, target_frame)
            };

            // Save example
            example.insert(
                "context_frames".to_owned(),
                Box::new(context_frames.into_dyn()),
            );
            example.insert("target_frame".to_owned(), Box::new(target_frame.into_dyn()));
            example.insert(
                "context_params".to_owned(),
                Box::new(context_cameras.into_dyn()),
            );
            example.insert("query_params".to_owned(), Box::new(query_camera.into_dyn()));

            example
        };

        let image_decoder = |example: ExampleType| {
            decode_image_on_example(example, Some(hashmap!("frames" => None)))
        };

        // Define train iterator
        let train_devices = devices.clone();
        let train_options = LoaderOptions {
            check_integrity: check_integrity,
            auto_close: false,
            parallel: true,
            open_limit: None,
            method: LoaderMethod::Mmap,
        };
        let train_loader = IndexedLoader::load_ex(train_files, train_options)?;
        let train_iter = train_loader
            .index_iter()
            .cycle()
            .shuffle(8192)
            .load_by_tfrecord_index(train_loader)
            .unwrap_result()
            .par_map(|bytes| bytes_to_example(&bytes, None))
            .unwrap_result()
            .par_map(image_decoder)
            .unwrap_result()
            .par_map(preprocessor)
            .batching(move |it| {
                let mut buf = Vec::new();

                while buf.len() < batch_size {
                    match it.next() {
                        Some(example) => buf.push(example),
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(make_batch(buf))
                }
            })
            .unwrap_result()
            .prefetch(512)
            .enumerate()
            .par_map(move |(idx, example)| {
                let dev_idx = idx % train_devices.len();
                let device = train_devices[dev_idx];
                let new_example = match example_to_torch_tensor(example, None, device) {
                    Ok(ret) => ret,
                    Err(err) => return Err(err),
                };
                Ok(new_example)
            })
            .unwrap_result()
            .batching(move |it| {
                // Here assumes par_map() is ordered
                let mut buf = vec![];
                while buf.len() < num_devices {
                    match it.next() {
                        Some(example) => buf.push(example),
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(buf)
                }
            });

        // Define test iterator
        let test_devices = devices.clone();
        let test_options = LoaderOptions {
            check_integrity: check_integrity,
            auto_close: false,
            parallel: true,
            open_limit: None,
            method: LoaderMethod::Mmap,
        };
        let test_loader = IndexedLoader::load_ex(test_files, test_options)?;
        let test_iter = test_loader
            .index_iter()
            .cycle()
            .shuffle(8192)
            .load_by_tfrecord_index(test_loader)
            .unwrap_result()
            .par_map(|bytes| bytes_to_example(&bytes, None))
            .unwrap_result()
            .par_map(image_decoder)
            .unwrap_result()
            .par_map(preprocessor)
            .batching(move |it| {
                let mut buf = Vec::new();

                while buf.len() < batch_size {
                    match it.next() {
                        Some(example) => buf.push(example),
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(make_batch(buf))
                }
            })
            .unwrap_result()
            .prefetch(512)
            .enumerate()
            .par_map(move |(idx, example)| {
                let dev_idx = idx % test_devices.len();
                let device = test_devices[dev_idx];
                let new_example = match example_to_torch_tensor(example, None, device) {
                    Ok(ret) => ret,
                    Err(err) => return Err(err),
                };
                Ok(new_example)
            })
            .unwrap_result()
            .batching(move |it| {
                // Here assumes par_map() is ordered
                let mut buf = vec![];
                while buf.len() < num_devices {
                    match it.next() {
                        Some(example) => buf.push(example),
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(buf)
                }
            });

        let dataset = DeepMindDataSet {
            name: name.to_owned(),
            train_size,
            test_size,
            frame_size,
            sequence_size,
            frame_channels,
            param_channels: 7, // We transfrom from 5 to 7 channels
            train_iter: Box::new(train_iter),
            test_iter: Box::new(test_iter),
        };
        Ok(dataset)
    }
}

impl<'a> FileDataset<'a> {
    pub fn load<P: AsRef<Path>>(
        input_dir: P,
        batch_size: usize,
        sequence_size: usize,
        frame_size: usize,
        time_step: f32,
        devices: Vec<Device>,
    ) -> Fallible<FileDataset<'a>> {
        ensure!(sequence_size >= 2, "sequence_size should be at least 2");

        let num_devices = devices.len();
        let train_devices = devices.clone();
        let mut image_paths = vec![];
        let regex_filename = Regex::new(r"^frame-\d+.png$")?;
        for entry_result in read_dir(input_dir.as_ref())? {
            let entry = entry_result?;
            let filename = entry.file_name();
            if entry.file_type()?.is_file() && regex_filename.is_match(filename.to_str().unwrap()) {
                image_paths.push(entry.path());
            }
        }
        ensure!(!image_paths.is_empty(), "Directory is empty");
        image_paths.sort();

        let timestamp_file = input_dir.as_ref().join("timestamps.txt");
        let mut timestamps = vec![];
        let reader = BufReader::new(File::open(timestamp_file)?);
        for line in reader.lines() {
            let ts: f32 = line?.parse()?;
            timestamps.push(ts / 1_000_000.0);
        }
        ensure!(
            image_paths.len() == timestamps.len(),
            "Image files and timestamp mismatch"
        );

        // let trunc_size = image_paths.len() / sequence_size * sequence_size;

        let samples: Vec<_> = timestamps
            .into_iter()
            .zip(image_paths.into_iter())
            .batching(move |it| {
                let mut buf = vec![];
                let mut expect_ts = 0.;

                loop {
                    match it.next() {
                        Some((ts, path)) => {
                            if buf.is_empty() {
                                buf.push((ts, path));
                                expect_ts = ts + time_step;
                            } else if ts >= expect_ts {
                                if ts - expect_ts >= time_step / 4. {
                                    buf.clear();
                                } else {
                                    buf.push((ts, path));
                                    expect_ts += time_step;
                                }
                            }

                            if buf.len() == sequence_size {
                                return Some(buf);
                            }
                        }
                        None => return None,
                    }
                }
            })
            .into_iter()
            .map(|chunk| {
                let samples: Vec<_> = chunk.into_iter().collect();
                let (first_ts, _) = samples[0];
                let samples: Vec<_> = samples
                    .into_iter()
                    .map(|(ts, path)| (ts - first_ts, path))
                    .collect();
                samples
            })
            .concat()
            .into_iter()
            .collect();

        let train_iter = samples
            .into_iter()
            .cycle()
            .par_map(move |(offset, path)| {
                let mut file = File::open(&path)?;
                let mut buf = vec![];
                let size = file.read_to_end(&mut buf)?;
                ensure!(size > 0, format!("{:?} is empty file", &path));

                let orig_image = image::load_from_memory_with_format(&buf, ImageFormat::PNG)?;
                let resized_image =
                    orig_image.resize(frame_size as u32, frame_size as u32, FilterType::CatmullRom);
                let new_width = resized_image.width() as usize;
                let new_height = resized_image.height() as usize;
                let pixels = resized_image.to_rgb().into_vec();
                let array = Array3::from_shape_vec((new_height, new_width, 3), pixels)?;

                let off = offset as f32 / 1000_f32; // ns to ms
                Ok((off, array))
            })
            .unwrap_result()
            .batching(move |it| {
                let mut offsets = vec![];
                let mut bufs = vec![];

                while offsets.len() < sequence_size {
                    match it.next() {
                        Some((off, buf)) => {
                            offsets.push(off);
                            bufs.push(buf);
                        }
                        None => {
                            assert!(offsets.is_empty());
                            break;
                        }
                    }
                }

                if offsets.is_empty() {
                    None
                } else {
                    let mut example: ExampleType = HashMap::new();
                    example.insert("offsets".to_owned(), Box::new(offsets));
                    example.insert("frames".to_owned(), Box::new(bufs));
                    Some(example)
                }
            })
            .par_map(move |mut example| {
                let (_, mut offsets_ref) = example.remove_entry("offsets").unwrap();

                let offsets = offsets_ref.downcast_mut::<Vec<f32>>().unwrap();

                let query_offset = offsets.pop().unwrap();
                let query_offset_array =
                    Array2::from_shape_vec((1, 1), vec![query_offset]).unwrap();
                let context_offsets_array =
                    Array2::from_shape_vec((sequence_size - 1, 1), offsets.to_vec()).unwrap();

                let mut frames = example
                    .remove_entry("frames")
                    .unwrap()
                    .1
                    .downcast_ref::<Vec<Array3<u8>>>()
                    .unwrap()
                    .into_iter()
                    .map(|array| array.mapv(|val| val as f32 / 255.).permuted_axes([2, 0, 1]))
                    .collect::<Vec<_>>();

                let target_frame = frames.pop().unwrap();
                let frames_expanded: Vec<_> = frames
                    .into_iter()
                    .map(|array| array.insert_axis(Axis(0)))
                    .collect();
                let frame_views: Vec<_> =
                    frames_expanded.iter().map(|array| array.view()).collect();

                let context_frames = ndarray::stack(Axis(0), &frame_views).unwrap();

                let mut new_example: ExampleType = HashMap::new();
                new_example.insert("context_frames".to_owned(), Box::new(context_frames));
                new_example.insert("target_frame".to_owned(), Box::new(target_frame));
                new_example.insert("context_params".to_owned(), Box::new(context_offsets_array));
                new_example.insert("query_params".to_owned(), Box::new(query_offset_array));

                new_example
            })
            .batching(move |it| {
                let mut buf = Vec::new();

                while buf.len() < batch_size {
                    match it.next() {
                        Some(example) => {
                            buf.push(example);
                        }
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(make_batch(buf))
                }
            })
            .unwrap_result()
            .enumerate()
            .par_map(move |(idx, example)| {
                let dev_idx = idx % train_devices.len();
                let device = train_devices[dev_idx];
                example_to_torch_tensor(example, None, device)
            })
            .unwrap_result()
            .prefetch(512)
            .batching(move |it| {
                // Here assumes par_map() is ordered
                let mut buf = vec![];
                while buf.len() < num_devices {
                    match it.next() {
                        Some(example) => buf.push(example),
                        None => break,
                    }
                }

                if buf.is_empty() {
                    None
                } else {
                    Some(buf)
                }
            });

        let dataset = FileDataset {
            frame_channels: 3,
            param_channels: 1,
            train_iter: Box::new(train_iter),
        };

        Ok(dataset)
    }
}
