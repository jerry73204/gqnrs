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
extern crate par_map;
#[macro_use] extern crate itertools;

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
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use crate::encoder::TowerEncoder;
use crate::model::{GqnModel, GqnModelOutput};

fn main() -> Result<(), Box<Error + Sync + Send>> {
    pretty_env_logger::init();

    // Parse arguments
    let arg_yaml = load_yaml!("args.yml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = arg_matches.value_of("DATASET_NAME").unwrap();
    let input_dir = Path::new(arg_matches.value_of("INPUT_DIR").unwrap());
    let output_dir = Path::new(arg_matches.value_of("OUTPUT_DIR").unwrap());
    let batch_size: usize = match arg_matches.value_of("BATCH_SIZE") {
        Some(arg) => arg.parse()?,
        None => 3,
    };

    let device = Device::Cuda(0);
    let vs = nn::VarStore::new(device);

    // Load dataset
    info!("Loading dataset");

    let gqn_dataset = dataset::DeepMindDataSet::load_dir(
        dataset_name,
        &input_dir,
        false,
        device,
        batch_size,
    )?;

    // Init model
    info!("Initialize model");

    let vs_root = vs.root();
    let model = GqnModel::<TowerEncoder>::new(&vs_root);
    let opt = nn::Adam::default().build(&vs, 1e-3)?;

    let mut cnt = 0;
    let mut instant = Instant::now();

    for (step, example) in gqn_dataset.train_iter.enumerate() {
        cnt += 1;
        let millis = instant.elapsed().as_millis();

        if millis >= 1000 {
            info!("rate: {}/s", cnt);
            cnt = 0;
            instant = Instant::now();
        }

        let context_frames = example["context_frames"].downcast_ref::<Tensor>().unwrap();
        let target_frame = example["target_frame"].downcast_ref::<Tensor>().unwrap();
        let context_cameras = example["context_cameras"].downcast_ref::<Tensor>().unwrap();
        let query_camera = example["query_camera"].downcast_ref::<Tensor>().unwrap();

        let GqnModelOutput {
            elbo_loss,
            target_mse,
            means_target,
            stds_target,
            target_sample,
            canvases,
            means_inf,
            stds_inf,
            means_gen,
            stds_gen,
        } = model.forward_t(
            context_frames,
            context_cameras,
            query_camera,
            target_frame,
            step as i64,
            true,
        );

        info!("step: {}\telbo_loss: {}", step, elbo_loss.double_value(&[]));

        opt.backward_step(&elbo_loss);
    }

    Ok(())
}
