use std::io::{self, Read, Seek};
use std::any::Any;
use std::error::Error;
use std::path;
use std::time::Instant;
use std::collections::HashMap;
use serde::Deserialize;
use glob::glob;
use yaml_rust::YamlLoader;
use byteorder::{ReadBytesExt, LittleEndian};
use crc::crc32;
use tch::{Tensor, Device};
use rayon::prelude::*;
use image::ImageFormat;
use ndarray::{ArrayBase, Array2, Array3, Axis};
use tfrecord_rs::iter::{DsIterator};
use tfrecord_rs::loader::{LoaderOptions, LoaderMethod, Loader, IndexedLoader};

pub struct DeepMindDataSet<'a> {
    pub name: &'a str,
    pub train_size: u64,
    pub test_size: u64,
    pub frame_size: u64,
    pub sequence_size: u64,
    pub num_channels: u64,
    pub train_iter: Box<Iterator<Item=HashMap<String, Box<dyn Any>>>>,
    // pub train_iter: Box<Iterator<Item=HashMap<String, Box<dyn Any + Sync + Send>>>>,
    pub test_iter: Box<Iterator<Item=HashMap<String, Box<dyn Any>>>>,
    device: &'a Device,
}


impl<'a> DeepMindDataSet<'a> {
    pub fn load_dir(
        name: &'a str,
        dataset_dir: &path::Path,
        check_integrity: bool,
        device: &'a Device,
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

        let preprocessor = move |mut example: HashMap<String, Box<dyn Any + Sync + Send>>| -> HashMap<String, Box<dyn Any + Sync + Send>> {
            let (_, cameras_ref) = example.remove_entry("cameras").unwrap();
            let (_, mut frames_ref) = example.remove_entry("frames").unwrap();

            let cameras = cameras_ref.downcast_ref::<Vec<f32>>().unwrap();
            let frames = frames_ref.downcast_mut::<Vec<Array3<u8>>>().unwrap();

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

            let (context_frames, target_frame) = {
                let shape = frames[0].shape();
                let width = shape[0];
                let height = shape[1];
                let channels = shape[2];

                assert!(sequence_size as usize == frames.len());
                assert!(width == frame_size as usize &&
                        height == frame_size as usize &&
                        channels == 3);

                let target_frame = frames.pop().unwrap();

                let contextn_frame_views = frames.into_iter()
                    .map(|array| array.view())
                    .collect::<Vec<_>>();

                let context_frames = ndarray::stack(Axis(0), &contextn_frame_views).unwrap();
                (context_frames, target_frame)
            };

            example.insert("context_frames".to_owned(), Box::new(context_frames));
            example.insert("query_frame".to_owned(), Box::new(target_frame));
            example.insert("context_cameras".to_owned(), Box::new(context_cameras));
            example.insert("query_camera".to_owned(), Box::new(query_camera));

            example
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
            .to_tf_example(None)
            .unwrap_result()
            .scan(
                (Instant::now(), 0),
                |(instant, cnt), example| {
                    *cnt += 1;
                    let millis = instant.elapsed().as_millis();
                    if millis >= 1000 {
                        println!("#1 {}", cnt);
                        *cnt = 0;
                        *instant = Instant::now();
                    }
                    Some(example)
            })
            .prefetch(8192)
            .scan(
                (Instant::now(), 0),
                |(instant, cnt), example| {
                    *cnt += 1;
                    let millis = instant.elapsed().as_millis();
                    if millis >= 1000 {
                        println!("#2 {}", cnt);
                        *cnt = 0;
                        *instant = Instant::now();
                    }
                    Some(example)
            })
            .par_decode_image(Some(hashmap!("frames".to_owned() => None)), 8192)
            .unwrap_result()
            .scan(
                (Instant::now(), 0),
                |(instant, cnt), example| {
                    *cnt += 1;
                    let millis = instant.elapsed().as_millis();
                    if millis >= 1000 {
                        println!("#3 {}", cnt);
                        *cnt = 0;
                        *instant = Instant::now();
                    }
                    Some(example)
            })
            .map(preprocessor)
            .to_torch_tensor(None)
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
            .to_tf_example(None)
            .unwrap_result()
            .prefetch(8192)
            .par_decode_image(Some(hashmap!("frames".to_owned() => None)), 8192)
            .unwrap_result()
            .map(preprocessor)
            .to_torch_tensor(None)
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
            device,
        };
        Ok(dataset)
    }
}
