extern crate clap;
extern crate tch;
extern crate yaml_rust;

mod representation;
mod utils;
mod dataset;

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

    let dataset_dir = Path::new(arg_matches.value_of("INPUT_DIR").unwrap());
    let output_dir = Path::new(arg_matches.value_of("OUTPUT_DIR").unwrap());

    let gqn_dataset = dataset::load_gqn_dataset(&dataset_dir);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let dummy_frames = Tensor::zeros(&[1, 3, 64, 64], (Kind::Float, device));
    let dummy_poses = Tensor::zeros(&[1, 7], (Kind::Float, device));
    let tower_encoder = representation::tower_encoder(&vs.root(), &dummy_frames, &dummy_poses);

}
