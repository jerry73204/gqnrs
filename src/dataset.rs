use std::io;
use std::any::Any;
use std::error::Error;
use std::path;
use std::time::Instant;
use std::collections::HashMap;
// use glob::glob;
use yaml_rust::YamlLoader;
use tch::{Tensor, Device};
use ndarray::{ArrayBase, Array2, Array3, Axis};
use par_map::ParMap;
use itertools::Itertools;
use tfrecord_rs::{ExampleType, ErrorType};
use tfrecord_rs::iter::DsIterator;
use tfrecord_rs::loader::{LoaderOptions, LoaderMethod, Loader, IndexedLoader};
use tfrecord_rs::utils::{bytes_to_example, decode_image_on_example, example_to_torch_tensor, make_batch};

pub struct DeepMindDataSet<'a> {
    pub name: &'a str,
    pub train_size: u64,
    pub test_size: u64,
    pub frame_size: u64,
    pub sequence_size: u64,
    pub num_channels: u64,
    pub train_iter: Box<Iterator<Item=HashMap<String, Box<dyn Any + Send>>> + 'a>,
    pub test_iter: Box<Iterator<Item=HashMap<String, Box<dyn Any + Send>>> + 'a>,
}

impl<'a> DeepMindDataSet<'a> {
    pub fn load_dir(
        name: &'a str,
        dataset_dir: &path::Path,
        check_integrity: bool,
        device: Device,
        batch_size: usize,
    ) -> Result<DeepMindDataSet<'a>, Box<Error + Sync + Send>> {
        let dataset_spec = &YamlLoader::load_from_str(include_str!("dataset.yaml"))?[0];
        let num_channels: u64 = dataset_spec["num_channels"].as_i64().unwrap() as u64;
        let num_camera_params: u64 = dataset_spec["num_camera_params"].as_i64().unwrap() as u64;
        let dataset_info = &dataset_spec["dataset"][name];
        let train_size: u64 = dataset_info["train_size"].as_i64().unwrap() as u64;
        let test_size: u64 = dataset_info["test_size"].as_i64().unwrap() as u64;
        let frame_size: u64 = dataset_info["frame_size"].as_i64().unwrap() as u64;
        let sequence_size: u64 = dataset_info["sequence_size"].as_i64().unwrap() as u64;

        let train_dir = dataset_dir.join("train");
        let test_dir = dataset_dir.join("test");

        let train_files: Vec<_> = glob::glob(train_dir.join("*.tfrecord").to_str().unwrap())?
            .take(1)
            .map(|p| p.unwrap())
            .collect();


        let test_files: Vec<_> = glob::glob(test_dir.join("*.tfrecord").to_str().unwrap())?
            .take(1)
            .map(|p| p.unwrap())
            .collect();

        let preprocessor = move |mut example: ExampleType| -> ExampleType {
            let (_, cameras_ref) = example.remove_entry("cameras").unwrap();
            let (_, mut frames_ref) = example.remove_entry("frames").unwrap();

            // Process camera data
            let cameras = cameras_ref.downcast_ref::<Vec<f32>>().unwrap();
            let (context_cameras, query_camera) = {
                assert!((sequence_size * num_camera_params) as usize == cameras.len());

                let orig_array: Box<Array2<f32>> = Box::new(ArrayBase::from_shape_vec(
                    (sequence_size as usize, num_camera_params as usize),
                    cameras.to_owned(),
                ).unwrap());

                let pos = orig_array.slice(s![.., 0..3]);
                let yaw = orig_array.slice(s![.., 3..4]);
                let pitch = orig_array.slice(s![.., 4..5]);

                let yaw_cos = yaw.mapv(|v| v.cos());
                let yaw_sin = yaw.mapv(|v| v.sin());
                let pitch_cos = pitch.mapv(|v| v.cos());
                let pitch_sin = pitch.mapv(|v| v.sin());

                let new_array = stack![Axis(1), pos, yaw_sin, yaw_cos, pitch_sin, pitch_cos];
                let context_cameras = new_array.slice(s![..(sequence_size as usize - 1), ..]).to_owned();
                let query_camera = new_array.slice(s![sequence_size as usize - 1, ..]).to_owned();
                (context_cameras, query_camera)
            };

            // Process frame data
            let mut frames = frames_ref
                .downcast_mut::<Vec<Array3<u8>>>()
                .unwrap()
                .iter()
                .map(|array| {
                    array.mapv(|val| val as f32)
                        .permuted_axes([2, 0, 1])
                })
                .collect::<Vec<_>>();

            let (context_frames, target_frame) = {
                let shape = frames[0].shape();
                let channels = shape[0];
                let height = shape[1];
                let width = shape[2];

                assert!(sequence_size as usize == frames.len() &&
                        width == frame_size as usize &&
                        height == frame_size as usize &&
                        channels == 3);

                let target_frame = frames.pop()
                    .unwrap();
                let frames_expanded = frames.into_iter()
                    .map(|array| array.insert_axis(Axis(0)))
                    .collect::<Vec<_>>();

                let frame_views = frames_expanded.iter()
                    .map(|array| array.view())
                    .collect::<Vec<_>>();

                let context_frames = ndarray::stack(Axis(0), &frame_views).unwrap();
                (context_frames, target_frame)
            };

            // Save example
            example.insert(
                "context_frames".to_owned(),
                Box::new(context_frames.into_dyn())
            );
            example.insert(
                "target_frame".to_owned(),
                Box::new(target_frame.into_dyn())
            );
            example.insert(
                "context_cameras".to_owned(),
                Box::new(context_cameras.into_dyn())
            );
            example.insert(
                "query_camera".to_owned(),
                Box::new(query_camera.into_dyn())
            );

            example
        };

        let image_decoder = |example: ExampleType| {
            decode_image_on_example(
                example,
                Some(hashmap!("frames" => None))
            )
        };

        let train_options = LoaderOptions {
            check_integrity: check_integrity,
            auto_close: false,
            parallel: true,
            open_limit: None,
            method: LoaderMethod::Mmap,
        };
        let train_loader = IndexedLoader::load_ex(train_files, train_options)?;
        let train_iter = train_loader.index_iter()
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
                }
                else {
                    Some(make_batch(buf))
                }
            })
            .unwrap_result()
            .prefetch(512)
            .par_map(move |example| example_to_torch_tensor(example, None, device))
            .unwrap_result();

        let test_options = LoaderOptions {
            check_integrity: check_integrity,
            auto_close: false,
            parallel: true,
            open_limit: None,
            method: LoaderMethod::Mmap,
        };
        let test_loader = IndexedLoader::load_ex(test_files, test_options)?;
        let test_iter = test_loader.index_iter()
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
                }
                else {
                    Some(make_batch(buf))
                }
            })
            .unwrap_result()
            .prefetch(512)
            .par_map(move |example| example_to_torch_tensor(example, None, device))
            .unwrap_result();

        let dataset = DeepMindDataSet {
            name: name,
            train_size,
            test_size,
            frame_size,
            sequence_size,
            num_channels,
            train_iter: Box::new(train_iter),
            test_iter: Box::new(test_iter),
        };
        Ok(dataset)
    }
}
