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
extern crate crossbeam;
extern crate ctrlc;
#[macro_use] extern crate lazy_static;

mod dist;
mod model;
mod encoder;
mod decoder;
mod utils;
mod dataset;
mod rnn;
mod objective;

use std::fmt::Debug;
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::error::Error;
use std::path::Path;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use par_map::ParMap;
use crossbeam::channel::bounded;
use tfrecord_rs::ExampleType;
use crate::encoder::TowerEncoder;
use crate::model::{GqnModel, GqnModelOutput};

lazy_static! {
    static ref SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);
}

fn main() -> Result<(), Box<Error + Sync + Send>> {
    pretty_env_logger::init();

    // Set signal handler
    // let mut flag_shutdown: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler(|| {
        warn!("Interrupted by user");
        SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    })?;

    // Parse arguments
    let arg_yaml = load_yaml!("args.yml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = arg_matches.value_of("DATASET_NAME").unwrap();
    let input_dir = Path::new(arg_matches.value_of("INPUT_DIR").unwrap());
    let model_file = match arg_matches.value_of("MODEL_FILE") {
        Some(file) => Some(Path::new(file)),
        None => None,
    };
    let save_steps: usize = match arg_matches.value_of("SAVE_STEPS") {
        Some(steps) => {
            let save_steps = steps.parse()?;
            assert!(save_steps > 0);
            save_steps
        },
        None => 128,
    };
    let batch_size: usize = match arg_matches.value_of("BATCH_SIZE") {
        Some(arg) => {
            let batch_size = arg.parse()?;
            assert!(batch_size > 0);
            batch_size
        }
        None => 3,
    };
    let num_gpus: usize = match arg_matches.value_of("NUM_GPUS") {
        Some(arg) => {
            let num_gpus = arg.parse()?;
            assert!(num_gpus > 0);
            num_gpus
        }
        None => 1,
    };

    // Init varaible stores
    let mut devices = vec![];
    let mut var_stores = vec![];

    for n in 0..num_gpus {
        let device = Device::Cuda(n);
        let vs = nn::VarStore::new(device);

        devices.push(device);
        var_stores.push(vs);
    }

    // Load dataset
    info!("Loading dataset");

    let gqn_dataset = dataset::DeepMindDataSet::load_dir(
        dataset_name,
        &input_dir,
        false,
        devices.clone(),
        batch_size,
    )?;

    // Init model
    info!("Initialize model");

    // Load model params
    if let Some(path) = model_file {
        if path.is_file() {
            info!("Loading model parameters");

            for vs in &mut var_stores {
                vs.load(path)?;
            }
        }
    }

    // Init optimizer
    let opt = nn::Adam::default().build(&var_stores[0], 1e-3)?;

    crossbeam::scope(|scope| -> Result<(), Box<dyn Error>> {
        // Spawn train workers
        let mut req_senders = vec![];
        let (resp_sender, resp_receiver) = bounded(num_gpus);

        for n in 0..num_gpus {
            let (req_sender, req_receiver) = bounded(1);
            let resp_sender_worker = resp_sender.clone();
            let vs = &var_stores[n];

            scope.builder()
                .name(format!("train_worker-{}", n))
                .spawn(move |_| {
                    // Init model
                    let root = vs.root();
                    let model = GqnModel::<TowerEncoder>::new(&root);

                    loop {
                        let (step, example) = match req_receiver.recv() {
                            Ok(Some(req)) => req,
                            Ok(None) => {
                                info!("Worker {} finished", n);
                                return;
                            },
                            Err(err) => {
                                error!("Worker {} failed: {:?}", n, err);
                                return;
                            }
                        };

                        let ret = run_model(&model, example, step);
                        match resp_sender_worker.send(ret) {
                            Err(err) => {
                                error!("Worker {} failed: {:?}", n, err);
                                return;
                            }
                            _ => {},
                        }
                    }
                })?;

            req_senders.push(req_sender);
        }

        // Produce train examples
        let mut train_iter = gqn_dataset.train_iter.enumerate();
        let global_instant = Instant::now();

        while !SHUTDOWN_FLAG.load(Ordering::SeqCst) {
            let (step, examples) = match train_iter.next() {
                Some(ret) => ret,
                None => break,
            };

            let step_instant = Instant::now();

            // Send to workers
            let n_examples = examples.len();
            for (sender, example) in req_senders.iter().zip(examples.into_iter()) {
                sender.send(Some((step as i64, example)))?;
            }

            // Receive from workers
            let mut resps = vec![];
            while resps.len() < n_examples {
                let output = resp_receiver.recv()?;
                resps.push(output);
            }

            // TODO run optimizer and copy trainable variables accross gpus
            let first_device = devices[0].clone();

            let total_batch_size: i64 = resps.iter()
                .map(|resp| resp.means_target.size()[0])
                .sum();

            let elbo_loss: Tensor = resps.iter()
                .map(|resp| {
                    let batch_size = resp.means_target.size()[0];
                    resp.elbo_loss.to_device(first_device) * batch_size
                })
                .sum::<Tensor>() / total_batch_size;

            let target_mse: Tensor = resps.iter()
                .map(|resp| {
                    let batch_size = resp.means_target.size()[0];
                    resp.target_mse.to_device(first_device) * batch_size
                })
                .sum::<Tensor>() / total_batch_size;

            // let means_target: Tensor = Tensor::cat(
            //     &resps.iter()
            //         .map(|resp| resp.means_target.to_device(first_device))
            //         .collect::<Vec<_>>(),
            //     0,
            // );

            // Backward step
            opt.backward_step(&elbo_loss);
            // TODO copy varaibles over GPUs

            info!(
                "step: {}\tglobal_elapsed: {}s\tstep_elapsed: {}ms\telbo_loss: {}\ttarget_mse: {}",
                step,
                global_instant.elapsed().as_secs(),
                step_instant.elapsed().as_millis(),
                elbo_loss.double_value(&[]),
                target_mse.double_value(&[]),
            );

            // Save model params
            if let Some(path) = model_file {
                if step % save_steps == 0 {
                    let vs = &var_stores[0];
                    vs.save(path)?;
                }
            }
        }

        // Gracefully terminate workers
        for (n, sender) in req_senders.into_iter().enumerate() {
            info!("Terminating train_worker-{}", n);
            sender.send(None)?;
        }

        Ok(())
    }).unwrap().unwrap();

    Ok(())
}

fn run_model(model: &GqnModel<TowerEncoder>, example: ExampleType, step: i64) -> GqnModelOutput {
    let context_frames = example["context_frames"].downcast_ref::<Tensor>().unwrap();
    let target_frame = example["target_frame"].downcast_ref::<Tensor>().unwrap();
    let context_cameras = example["context_cameras"].downcast_ref::<Tensor>().unwrap();
    let query_camera = example["query_camera"].downcast_ref::<Tensor>().unwrap();

    model.forward_t(
        context_frames,
        context_cameras,
        query_camera,
        target_frame,
        step,
        true,
    )
}
