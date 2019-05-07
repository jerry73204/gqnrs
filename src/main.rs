extern crate byteorder;
extern crate serde;
extern crate image;
extern crate protobuf;
extern crate crc;
#[macro_use] extern crate clap;
extern crate tch;
extern crate yaml_rust;
extern crate glob;
extern crate rayon;
#[macro_use] extern crate maplit;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
#[macro_use] extern crate ndarray;
extern crate tfrecord_rs;

mod dist;
mod model;
mod encoder;
mod decoder;
mod utils;
mod dataset;
mod rnn;
mod objective;

use std::time::Instant;
use std::error::Error;
use std::path::Path;
use tch::{nn, Device, Tensor, Kind};
use yaml_rust::YamlLoader;
use crate::encoder::TowerEncoder;
use crate::model::GqnModel;

fn main() -> Result<(), Box<Error + Sync + Send>> {
    pretty_env_logger::init();

    // Parse arguments
    let arg_yaml = load_yaml!("args.yml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = arg_matches.value_of("DATASET_NAME").unwrap();
    let input_dir = Path::new(arg_matches.value_of("INPUT_DIR").unwrap());
    let output_dir = Path::new(arg_matches.value_of("OUTPUT_DIR").unwrap());

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    // Load dataset
    info!("Loading dataset");

    let gqn_dataset = dataset::DeepMindDataSet::load_dir(
        dataset_name,
        &input_dir,
        false,
        &device,
    )?;

    // Init model
    info!("Initialize model");

    let vs_root = vs.root();
    let model = GqnModel::<TowerEncoder>::new(&vs_root);

    let mut cnt = 0;
    let mut instant = Instant::now();

    for example in gqn_dataset.train_iter {
        cnt += 1;
        let millis = instant.elapsed().as_millis();

        if millis >= 1000 {
            println!("#total {}", cnt);
            cnt = 0;
            instant = Instant::now();
        }

        // let frames = match &example["frames"] {
        //     Feature::TorchTensorList(list) => list,
        //     _ => panic!("Unexpected feature type"),
        // };

        // let cameras = match &example["cameras"] {
        //     Feature::TorchTensor(tensor) => tensor,
        //     _ => panic!("Unexpected feature type"),
        // };

        // info!("{:?} {:?}", frames[0].size(), cameras.size());
        // assert!(frames.len() == utils::SEQ_LENGTH as usize);
    }

    Ok(())
}
