extern crate serde_json;
extern crate serde;
extern crate image;
extern crate telamon_utils;
extern crate glob;
extern crate protobuf;

use std::error::Error;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use serde::Deserialize;
use image::GenericImageView;
use glob::glob;
use yaml_rust::YamlLoader;
use super::tf_proto::example::Example;

#[derive(Deserialize, Debug)]
pub struct GqnDataSetInfo
{
    basepath: String,
    train_size: u64,
    test_size: u64,
    frame_size: u64,
    sequence_size: u64,
}

pub fn load_gqn_tfrecord(name: &str, dataset_dir: &Path) -> Result<(), Box<Error>>
{
    let dataset_spec = &YamlLoader::load_from_str(include_str!("dataset.yaml"))?[0];
    let dataset_info = &dataset_spec["dataset"][name];
    let num_camera_params = dataset_spec["num_camera_params"].as_i64().unwrap();
    let train_size = dataset_info["train_size"].as_i64().unwrap();
    let test_size = dataset_info["test_size"].as_i64().unwrap();
    let frame_size = dataset_info["frame_size"].as_i64().unwrap();
    let sequence_size = dataset_info["sequence_size"].as_i64().unwrap();

    let train_dir = dataset_dir.join("train");
    let test_dir = dataset_dir.join("test");

    for entry in glob(train_dir.join("*.tfrecord").to_str().unwrap())?
    {
        let file = File::open(entry?)?;
        let buf_reader = BufReader::new(file);
        let tf_reader = telamon_utils::tfrecord::Reader::from_reader(buf_reader);

        for rec_result in tf_reader.records()
        {
            let buffer = [0; 8192];
            let rec = rec_result?;
            let example: Example = protobuf::parse_from_bytes(&rec)?;
            let features = example.get_features().get_feature();
            let frames = &features["frames"]
                .get_bytes_list()
                .get_value();
            let cameras = &features["cameras"]
                .get_float_list()
                .get_value();

            assert!(sequence_size == frames.len() as i64);
            assert!(sequence_size * num_camera_params == cameras.len() as i64);
        }
        break;
    }

    Ok(())
}
