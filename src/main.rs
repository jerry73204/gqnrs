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
extern crate crossbeam;
extern crate ctrlc;
#[macro_use] extern crate lazy_static;
extern crate regex;
#[macro_use] extern crate failure;

mod dist;
mod model;
mod encoder;
mod decoder;
mod params;
mod dataset;
mod rnn;
mod objective;

use std::fs::create_dir;
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::path::{Path, PathBuf};
use std::thread;
use tch::{nn, nn::Init, nn::OptimizerConfig, Device, Tensor, Kind};
use crossbeam::channel::bounded;
use tfrecord_rs::ExampleType;
use regex::Regex;
use image::{Rgb, ImageBuffer};
use failure::Fallible;
use crate::encoder::TowerEncoder;
use crate::model::{GqnModel, GqnModelOutput};

lazy_static! {
    static ref SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);
    static ref SIGUSR_FLAG: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

struct Args {
    dataset_name: String,
    input_dir: PathBuf,
    model_file: Option<PathBuf>,
    log_dir: Option<PathBuf>,
    save_steps: i64,
    log_steps: i64,
    batch_size: usize,
    initial_step: Option<i64>,
    devices: Vec<Device>,
    save_images: bool,
}

enum WorkerAction where
{
    Forward((i64, ExampleType)),
    Backward((i64, Tensor)),
    CopyParams,
    LoadParams(PathBuf),
    SaveParams(PathBuf, i64),
    Terminate,
}

enum WorkerResponse {
    ForwardOutput(GqnModelOutput),
    Step(i64),
}

fn main() -> Fallible<()> {
    pretty_env_logger::init();

    // Set signal handler
    ctrlc::set_handler(|| {
        warn!("Interrupted by user");
        SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    })?;

    // Parse arguments
    let args = parse_args()?;

    // Init log dir
    if let Some(path) = &args.log_dir {
        if path.is_file() {
            panic!("The specified log dir path {:?} is a file", path);
        }
        else if !path.exists() {
            create_dir(path)?;
        }
    }
    // Load dataset
    info!("Loading dataset");

    let gqn_dataset = dataset::DeepMindDataSet::load_dir(
        &args.dataset_name,
        &args.input_dir,
        false,
        args.devices.clone(),
        args.batch_size,
    )?;
    let input_frame_channels = gqn_dataset.frame_channels;

    // Spawn train workers
    let mut req_senders = vec![];
    let (resp_sender, resp_receiver) = bounded(args.devices.len());
    let (param_senders, param_receivers) = {
        let mut senders = vec![];
        let mut receivers = vec![];
        for dev in &args.devices[1..] {
            let (sender, receiver) = bounded(1);
            senders.push((*dev, sender));
            receivers.push((*dev, receiver));
        }
        (senders, receivers)
    };
    let update_barrier = Arc::new(Barrier::new(args.devices.len()));

    for (worker_id, worker_dev) in args.devices.iter().enumerate() {
        let (req_sender, req_receiver) = bounded(1);
        let resp_sender_worker = resp_sender.clone();
        let param_receiver = match worker_id {
            0 => None,
            n => {
                let (to_dev, receiver) = &param_receivers[n - 1];
                assert!(worker_dev == to_dev);
                Some(receiver.clone())
            }
        };
        let param_senders_worker = match worker_id {
            0 => {
                let senders: Vec<_> = param_senders.iter()
                    .map(|(to_dev, sender)| ((*to_dev).clone(), sender.clone()))
                    .collect();
                Some(senders)
            }
            _ => None,
        };
        let update_barrier_worker = update_barrier.clone();
        let worker_dev_thread = *worker_dev;

        thread::Builder::new()
            .name(format!("train_worker-{}", worker_id))
            .spawn(move || {
                // Load model params
                let mut vs = nn::VarStore::new(worker_dev_thread);

                // Init model
                debug!("Initialize model on worker {}", worker_id);
                let model = {
                    let root = vs.root();
                    let model = GqnModel::<TowerEncoder>::new(&root, input_frame_channels);
                    let _ = root.zeros("step", &[]);
                    model
                };

                // Init optimizer
                let mut optimizer_opt = match worker_id {
                    0 => {
                        let opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
                        Some(opt)
                    }
                    _ => None,
                };

                loop {
                    match req_receiver.recv() {
                        Ok(WorkerAction::Forward((step, example))) => {
                            debug!("Forward pass on worker {}", worker_id);
                            let output = run_model(&model, example, step);
                            let resp = WorkerResponse::ForwardOutput(output);
                            match resp_sender_worker.send(resp) {
                                Err(err) => {
                                    error!("Worker {} failed: {:?}", worker_id, err);
                                    return;
                                }
                                _ => {},
                            }
                        }
                        Ok(WorkerAction::Backward((step, elbo_loss))) => {
                            debug!("Backward pass on worker {}", worker_id);

                            let begin = params::ADAM_LR_BEGIN;
                            let end = params::ADAM_LR_END;
                            let max_step = params::ANNEAL_LR_MAX;
                            let lr = begin + (begin - end) * (1. - (step as f64 / max_step as f64).min(1.));

                            let opt = optimizer_opt.as_mut().unwrap();
                            opt.set_lr(lr);
                            opt.backward_step(&elbo_loss);
                        }
                        Ok(WorkerAction::LoadParams(path)) => {
                            debug!("Load model parameters to worker {}", worker_id);
                            vs.load(path).unwrap();

                            // Update step variable if necessary
                            if worker_id == 0 {
                                let step = {
                                    let root = vs.root();
                                    let step_tensor = root.get("step")
                                        .unwrap();
                                    // Uncomment this for backward compatibility
                                    // let step_tensor = root.entry("step")
                                    //     .or_zeros(&[]);
                                    step_tensor.int64_value(&[])
                                };
                                let resp = WorkerResponse::Step(step);
                                match resp_sender_worker.send(resp) {
                                    Err(err) => {
                                        error!("Worker {} failed: {:?}", worker_id, err);
                                        return;
                                    }
                                    _ => {},
                                }
                            }
                        }
                        Ok(WorkerAction::SaveParams(path, step)) => {
                            debug!("Save model params from worker {}", worker_id);
                            {
                                let root = vs.root();
                                let mut step_tensor = root.get("step").unwrap();
                                tch::no_grad(|| step_tensor.init(Init::Const(step as f64)));
                            }
                            vs.save(path).unwrap();
                        }
                        Ok(WorkerAction::CopyParams) => {
                            match worker_id {
                                0 => {
                                    debug!("Send param copies from worker {}", worker_id);
                                    let vs_rc = Arc::new(vs);
                                    for (to_dev, sender) in param_senders_worker.as_ref().unwrap().iter() {
                                        sender.send(vs_rc.clone()).unwrap();
                                    }
                                    update_barrier_worker.wait();
                                    vs = Arc::try_unwrap(vs_rc).unwrap();
                                }
                                _ => {
                                    debug!("Update params on worker {}", worker_id);
                                    {
                                        let receiver = param_receiver.as_ref().unwrap();
                                        let from_vs = receiver.recv().unwrap();
                                        vs.copy(&from_vs).unwrap();
                                    } // This scope makes sures deallocation of from_vs
                                    update_barrier_worker.wait();
                                }
                            }
                        },
                        Ok(WorkerAction::Terminate) => {
                            debug!("Worker {} finished", worker_id);
                            return;
                        },
                        Err(err) => {
                            error!("Worker {} failed: {:?}", worker_id, err);
                            return;
                        }
                    };
                }
            })?;

        req_senders.push(req_sender);
    }

    // Produce train examples
    let mut train_iter = gqn_dataset.train_iter;
    let global_instant = Instant::now();
    let mut step = 0;

    // Load model params
    if let Some(path) = &args.model_file {
        if path.is_file() {
            info!("Load model file {:?}", path);
            for sender in req_senders.iter() {
                sender.send(WorkerAction::LoadParams(path.to_path_buf())).unwrap();
            }

            // Update step count
            step = match resp_receiver.recv()? {
                WorkerResponse::Step(resp_step) => {
                    match args.initial_step {
                        Some(init_step) => init_step,
                        None => resp_step,
                    }
                }
                _ => panic!("Wrong response type"),
            }
        }
    }

    // Main loop
    while !SHUTDOWN_FLAG.load(Ordering::SeqCst) {
        let examples = match train_iter.next() {
            Some(ret) => ret,
            None => break,
        };

        let step_instant = Instant::now();

        // Send examples to workers
        let n_examples = examples.len();
        for (sender, example) in req_senders.iter().zip(examples.into_iter()) {
            sender.send(WorkerAction::Forward((step as i64, example))).unwrap();
        }

        // Receive outputs from workers
        let mut outputs = vec![];
        while outputs.len() < n_examples {
            let output = match resp_receiver.recv()? {
                WorkerResponse::ForwardOutput(output) => output,
                _ => panic!("Wrong response type"),
            };
            outputs.push(output);
        }

        // combine model outputs
        let mut elbo_loss_grad = outputs[0].elbo_loss.shallow_clone();
        let combined = combine_gqn_outputs(outputs, args.devices[0]);

        // Backward step
        tch::no_grad(|| elbo_loss_grad.copy_(&combined.elbo_loss));
        req_senders[0].send(WorkerAction::Backward((step as i64, elbo_loss_grad))).unwrap();

        // let workers update params
        for sender in req_senders.iter() {
            sender.send(WorkerAction::CopyParams).unwrap();
        }

        // Save model params
        if let Some(path) = &args.model_file {
            if step % args.save_steps == 0 {
                req_senders[0].send(WorkerAction::SaveParams(path.to_path_buf(), step)).unwrap();
            }
        }

        // Log data
        if let Some(path) = &args.log_dir {
            if step % args.log_steps == 0 {
                // Save images
                if args.save_images {
                    save_images(step, &combined, path);
                }

                // Save model outputs
                save_model_outputs(step, &combined, path)?;
            }
        }

        // Write output
        info!(
            "step: {}\tglobal_elapsed: {}s\tstep_elapsed: {}ms\telbo_loss: {:.3}\ttarget_mse: {:.6}",
            step,
            global_instant.elapsed().as_secs(),
            step_instant.elapsed().as_millis(),
            combined.elbo_loss.double_value(&[]),
            combined.target_mse.double_value(&[]),
        );

        // Update step
        step += 1;
    }

    // Gracefully terminate workers
    if let Some(path) = &args.model_file {
        req_senders[0].send(WorkerAction::SaveParams(path.to_path_buf(), step)).unwrap();
    }

    for (n, sender) in req_senders.into_iter().enumerate() {
        debug!("Terminating train_worker-{}", n);
        sender.send(WorkerAction::Terminate).unwrap();
    }

    Ok(())
}

fn combine_gqn_outputs(outputs: Vec<GqnModelOutput>, target_device: Device) -> GqnModelOutput {
    let combine_cat = |tensors: &[&Tensor]| {
        Tensor::cat(
            &tensors.iter()
                .map(|tensor| tensor.to_device(target_device))
                .collect::<Vec<_>>(),
            0,
        )
    };

    let combine_mean = |tensors: &[&Tensor]| {
        combine_cat(tensors).mean2(&[], false)
    };

    // let tensor_to_vec = |tensor: &Tensor| {
    //     let buf_size = tensor.numel();
    //     let mut buf = vec![0_f32; buf_size as usize];
    //     tensor.copy_data(&mut buf, buf_size);
    //     buf
    // };

    let elbo_loss = combine_mean(
        &outputs.iter()
            .map(|resp| &resp.elbo_loss)
            .collect::<Vec<_>>()
    );

    let target_mse = combine_mean(
        &outputs.iter()
            .map(|resp| &resp.target_mse)
            .collect::<Vec<_>>()
    );

    let canvases = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.canvases)
            .collect::<Vec<_>>()
    );

    let target_sample = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.target_sample)
            .collect::<Vec<_>>()
    );

    let means_target = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.means_target)
            .collect::<Vec<_>>()
    );

    let stds_target = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.stds_target)
            .collect::<Vec<_>>()
    );

    let means_inf = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.means_inf)
            .collect::<Vec<_>>()
    );

    let stds_inf = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.stds_inf)
            .collect::<Vec<_>>()
    );

    let means_gen = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.means_gen)
            .collect::<Vec<_>>()
    );

    let stds_gen = combine_cat(
        &outputs.iter()
            .map(|resp| &resp.stds_gen)
            .collect::<Vec<_>>()
    );

    GqnModelOutput {
        elbo_loss,
        target_mse,
        target_sample,
        means_target,
        stds_target,
        canvases,
        means_inf,
        stds_inf,
        means_gen,
        stds_gen,
    }
}

fn parse_args() -> Fallible<Args> {
    let arg_yaml = load_yaml!("args.yml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = arg_matches.value_of("DATASET_NAME").unwrap().to_owned();
    let input_dir = PathBuf::from(arg_matches.value_of("INPUT_DIR").unwrap());
    let model_file = match arg_matches.value_of("MODEL_FILE") {
        Some(file) => Some(PathBuf::from(file)),
        None => None,
    };
    let log_dir = match arg_matches.value_of("LOG_DIR") {
        Some(path) => Some(PathBuf::from(path)),
        None => None,
    };
    let save_steps: i64 = {
        let arg = arg_matches.value_of("SAVE_STEPS")
            .unwrap_or("100");

        let save_steps = arg.parse()?;
        ensure!(save_steps > 0, "SAVE_STEPS should be positive");
        save_steps
    };
    let log_steps: i64 = {
        let arg = arg_matches.value_of("LOG_STEPS")
            .unwrap_or("100");

        let log_steps = arg.parse()?;
        ensure!(log_steps > 0, "LOG_STEPS should be positive");
        log_steps
    };
    let batch_size: usize = {
        let arg = arg_matches.value_of("BATCH_SIZE")
            .unwrap_or("4");
        let batch_size = arg.parse()?;
        ensure!(batch_size > 0, "BATCH_SIZE should be positive");
        batch_size
    };
    let initial_step: Option<i64> = match arg_matches.value_of("INIT_STEP") {
        Some(arg) => {
            let initial_step = arg.parse()?;
            ensure!(initial_step >= 0, "INIT_STEP should be positive");
            Some(initial_step)
        }
        None => None,
    };
    let devices: Vec<Device> = match arg_matches.value_of("DEVICES") {
        Some(arg) => {
            let mut devices = vec![];
            for token in arg.split(",") {
                let cap = Regex::new(r"cuda\((\d+)\)$")?
                    .captures(&token)
                    .unwrap();
                let dev_index = cap[1].parse()?;
                devices.push(Device::Cuda(dev_index));
            }
            devices
        }
        None => vec![Device::Cuda(0)],
    };
    let save_images: bool = arg_matches.is_present("SAVE_IMAGE");

    Ok(Args {
        dataset_name,
        input_dir,
        model_file,
        log_dir,
        save_steps,
        log_steps,
        batch_size,
        initial_step,
        devices,
        save_images,
    })
}

fn save_images<P: AsRef<Path>>(step: i64, combined: &GqnModelOutput, log_dir: P) {
    let size = combined.means_target.size();
    let batch_size = size[0];
    let height = size[2];
    let width = size[3];

    let min_val: Tensor = 255_f32.into();
    let min_val =  min_val.to_device(combined.means_target.device());

    for batch_idx in 0..batch_size {
        let result_image = (combined.means_target.select(0, batch_idx) * 255.)
            .min1(&min_val)
            .permute(&[1, 2, 0])
            .to_kind(Kind::Uint8);

        let buf_size = result_image.numel()as usize;
        let mut buf = vec![0_u8; buf_size];
        result_image.copy_data(&mut buf, buf_size as i64);

        let filename = format!("{:0>10}-{:0>2}.jpg", step, batch_idx);
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, buf).unwrap()
            .save(log_dir.as_ref().join(filename)).unwrap();
    }

}

fn save_model_outputs<P: AsRef<Path>>(step: i64, combined: &GqnModelOutput, log_dir: P) -> Fallible<()> {
    let sys_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis();

    let filename = format!("{:0>10}-{:0>13}.zip", step, sys_time);
    let file_path = log_dir.as_ref().join(filename);

    let data = vec![
        ("elbo_loss", combined.elbo_loss.shallow_clone()),
        ("target_mse", combined.target_mse.shallow_clone()),
        ("target_sample", combined.target_sample.shallow_clone()),
        ("means_target", combined.means_target.shallow_clone()),
        ("stds_target", combined.stds_target.shallow_clone()),
        ("means_inf", combined.means_inf.shallow_clone()),
        ("stds_inf", combined.stds_inf.shallow_clone()),
        ("means_gen", combined.means_gen.shallow_clone()),
        ("stds_gen", combined.stds_gen.shallow_clone()),
    ];

    Tensor::save_multi(
        &data,
        file_path,
    )?;

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
