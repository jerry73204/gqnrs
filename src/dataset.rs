use std::io;
use std::io::{Read, Seek};
use std::error;
use std::path;
use std::fs;
use std::collections;
use serde::Deserialize;
use glob::glob;
use yaml_rust::YamlLoader;
use byteorder::{ReadBytesExt, LittleEndian};
use crc::crc32;
use image::jpeg::JPEGDecoder;
use image::ImageDecoder;
use tch::Tensor;
use rayon::prelude::*;
use tfrecord_rs::loader::{MmapRecordLoader, RandomAccessRecordLoader};
use crate::tf_proto::example::Example;

pub struct DeepMindDataSet<'a>
{
    name: &'a str,
    train_size: u64,
    test_size: u64,
    frame_size: u64,
    sequence_size: u64,
}


impl<'a> DeepMindDataSet<'a>
{
    pub fn load_dir(name: &'a str, dataset_dir: &path::Path, check_integrity: bool) -> Result<DeepMindDataSet<'a>, Box<error::Error>>
    {
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

        let build_loaders = |dir: path::PathBuf, expect_num_files: u64| -> Result<_, Box<error::Error>> {
            let mut paths = Vec::<path::PathBuf>::new();
            for entry in glob(dir.join("*.tfrecord").to_str().unwrap())?
            {
                paths.push(entry?);
            }

            assert!(paths.len() as u64 == expect_num_files);

            let loaders: Vec<_> = paths.into_par_iter().map(|path: path::PathBuf| -> (path::PathBuf, RandomAccessRecordLoader<fs::File>) {
                println!("Loading {}", path.as_path().display());
                let loader = RandomAccessRecordLoader::from((path.as_path(), false));
                (path, loader)
            }).collect();

            Ok(loaders)
        };

        let train_loaders = build_loaders(train_dir, train_size)?;
        let test_loaders = build_loaders(test_dir, test_size)?;

        let dataset = DeepMindDataSet {
            name: name,
            train_size,
            test_size,
            frame_size,
            sequence_size,
        };
        Ok(dataset)
    }
}
