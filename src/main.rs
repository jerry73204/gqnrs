extern crate byteorder;
extern crate serde;
extern crate image;
extern crate protobuf;
extern crate crc;
extern crate clap;
extern crate tch;
extern crate yaml_rust;
extern crate glob;
extern crate rayon;
#[macro_use] extern crate log;
extern crate tfrecord_rs;

mod encoder;
mod decoder;
mod utils;
mod dataset;
mod rnn;
mod tf_proto;

use std::path::Path;
use tch::{nn, Device, Tensor, Kind};
use yaml_rust::YamlLoader;

fn main() {
    // Parse arguments
    let arg_config = "
name: gqnrs
version: \"1.0\"
author: Jerry Lin <jerry73204@gmail.com>
about: GQN implementation in Rust
args:
    - DATASET_NAME:
        short: n
        long: dataset-name
        required: true
        takes_value: true
    - INPUT_DIR:
        short: i
        long: input-dir
        required: true
        takes_value: true
    - OUTPUT_DIR:
        short: o
        long: output-dir
        required: true
        takes_value: true
";
    let arg_yaml = YamlLoader::load_from_str(arg_config).unwrap();
    let arg_matches = clap::App::from_yaml(&arg_yaml[0]).get_matches();

    let dataset_name = arg_matches.value_of("DATASET_NAME").unwrap();
    let input_dir = Path::new(arg_matches.value_of("INPUT_DIR").unwrap());
    let output_dir = Path::new(arg_matches.value_of("OUTPUT_DIR").unwrap());

    let gqn_dataset = dataset::DeepMindDataSet::load_dir(dataset_name, &input_dir, true).unwrap();

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let dummy_frames = Tensor::zeros(&[1, 3, 64, 64], (Kind::Float, device));
    let dummy_poses = Tensor::zeros(&[1, 7], (Kind::Float, device));

    let tower_encoder = encoder::TowerEncoder::new(&vs.root(), 10);
    let dummy_repr = tower_encoder.forward(&dummy_frames, &dummy_poses);

    let lstm = rnn::GqnLSTM::new(&vs.root(), true, true, 3, 8, 5, 1.0);
    let dummy_inputs = Tensor::zeros(&[32, 10, 3, 48, 48], (Kind::Float, device));
    // let (dummy_outputs, dummy_state) = lstm.seq(&dummy_inputs);
}
