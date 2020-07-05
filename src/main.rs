use gqnrs::{
    common::*,
    config::{Config, DatasetConfig, DeepMindDatasetConfig},
    dataset,
    message::{WorkerAction, WorkerResponse},
    model::{GqnModel, GqnModelInput, GqnModelOutput, TowerEncoder},
};

lazy_static::lazy_static! {
    static ref SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);
    static ref SIGUSR_FLAG: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

/// The Rust implementation of Generative Query Network.
#[derive(FromArgs)]
struct Args {
    /// the config file.
    #[argh(option, default = "PathBuf::from(\"config.json5\")")]
    config: PathBuf,
}

#[async_std::main]
async fn main() -> Fallible<()> {
    pretty_env_logger::init();

    // Set signal handler
    ctrlc::set_handler(|| {
        warn!("Interrupted by user");
        SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    })?;

    // Parse arguments
    let args: Args = argh::from_env();

    // load config
    let config = Config::open(&args.config)?;

    // init log dir
    if config.logging.enabled {
        async_std::fs::create_dir_all(&config.logging.log_dir).await?;
    }

    // Load dataset
    info!("Loading dataset");

    let (dataset, frame_channels, param_channels) = match config.dataset {
        DatasetConfig::DeepMind(dataset_config) => {
            let DeepMindDatasetConfig {
                frame_channels,
                dataset_dir,
                train_size,
                test_size,
                frame_size,
                sequence_size,
                check_integrity,
                ..
            } = dataset_config;

            let dataset = dataset::deepmind::DatasetInit {
                frame_channels,
                dataset_dir,
                train_size,
                test_size,
                frame_size,
                sequence_size,
                check_integrity,
                batch_size: config.training.batch_size,
                devices: &config.training.devices,
            }
            .build()
            .await?;

            let param_channels = dataset::deepmind::NUM_CAMERA_PARAMS;

            (dataset, frame_channels.get(), param_channels)
        }
    };

    // spawn training workers
    struct UploadMessage {}
    struct DownloadMessage {}
    struct DataMessage {
        pub step: usize,
        pub input: GqnModelInput,
    }

    let num_workers = config.training.devices.len();
    let (upload_tx, upload_rx) = mpsc::channel::<UploadMessage>(num_workers);
    let (download_tx, _download_rx) = broadcast::channel::<DownloadMessage>(num_workers);
    let mut upload_rx = Some(upload_rx);
    let (mut data_tx_set, mut data_rx_set) = (0..num_workers).fold(
        (vec![], HashMap::new()),
        |(mut tx_set, mut rx_set), worker_index| {
            let (tx, rx) = mpsc::channel::<DataMessage>(1);
            tx_set.push(tx);
            rx_set.insert(worker_index, rx);
            (tx_set, rx_set)
        },
    );

    let feed_future = async move {
        let mut train_stream = Box::pin(dataset.train_stream(0)?); // TODO: initial step
        let mut step: usize = 0;

        loop {
            let (train_stream_, inputs) = futures::stream::iter(0..num_workers)
                .fold(Ok((train_stream, vec![])), |result, _| {
                    async move {
                        let (mut stream, mut inputs) = result?;
                        inputs.push(stream.next().await.unwrap()?);
                        Fallible::Ok((stream, inputs))
                    }
                })
                .await?;
            train_stream = train_stream_;

            for (data_tx, input) in data_tx_set.iter_mut().zip(inputs.into_iter()) {
                if let Err(_) = data_tx.send(DataMessage { step, input }).await {
                    panic!("please report bug");
                }
            }

            let (step_, overflow) = step.overflowing_add(1);
            if overflow {
                warn!("step value overflow");
            }
            step = step_;
        }

        Fallible::Ok(())
    };

    let train_futures = config
        .training
        .devices
        .iter()
        .map(ToOwned::to_owned)
        .enumerate()
        .map(|(worker_index, device)| {
            let is_master = worker_index == 0;
            let mut data_rx = data_rx_set.remove(&worker_index).unwrap();
            let (mut upload_tx_opt, mut upload_rx_opt, mut download_tx_opt, mut download_rx_opt) =
                if is_master {
                    (
                        None,
                        Some(upload_rx.take().unwrap()),
                        Some(download_tx.clone()),
                        None,
                    )
                } else {
                    (
                        Some(upload_tx.clone()),
                        None,
                        None,
                        Some(download_tx.subscribe()),
                    )
                };

            async move {
                let vs = VarStore::new(device);
                let model = {
                    let root = vs.root();
                    let model = GqnModel::<TowerEncoder>::new(
                        &root,
                        frame_channels as i64,
                        param_channels as i64,
                    );
                    let _ = root.zeros("step", &[]);
                    model
                };

                while let Some(data_msg) = data_rx.recv().await {
                    let DataMessage { step, input } = data_msg;

                    let output = model.forward_t(input, true);

                    // TODO

                    if is_master {
                        let upload_rx = upload_rx_opt.as_mut().unwrap();
                        let download_tx = download_tx_opt.as_mut().unwrap();
                    } else {
                        let upload_tx = upload_tx_opt.as_mut().unwrap();
                        let download_rx = download_rx_opt.as_mut().unwrap();
                    }
                }

                Fallible::Ok(())
            }
        })
        .map(async_std::task::spawn);

    futures::future::try_join(feed_future, futures::future::try_join_all(train_futures)).await?;

    // let train_futures = config
    //     .devices
    //     .iter()
    //     .map(ToOwned::to_owned)
    //     .enumerate()
    //     .map(|(worker_index, device)| {
    //         async move {
    //             train_worker(worker_index, device).await?;
    //             Fallible::Ok(())
    //         }
    //     })
    //     .map(async_std::task::spawn);

    // Spawn train workers
    // let mut req_senders = vec![];
    // let (resp_sender, resp_receiver) = crossbeam::channel::bounded(config.devices.len());
    // let (param_senders, param_receivers) = {
    //     let mut senders = vec![];
    //     let mut receivers = vec![];
    //     for dev in &config.devices[1..] {
    //         let (sender, receiver) = crossbeam::channel::bounded(1);
    //         senders.push((*dev, sender));
    //         receivers.push((*dev, receiver));
    //     }
    //     (senders, receivers)
    // };
    // let update_barrier = Arc::new(Barrier::new(config.devices.len()));

    // for (worker_id, worker_dev) in config.devices.iter().enumerate() {
    //     let (req_sender, req_receiver) = crossbeam::channel::bounded(1);
    //     let resp_sender_worker = resp_sender.clone();
    //     let param_receiver = match worker_id {
    //         0 => None,
    //         n => {
    //             let (to_dev, receiver) = &param_receivers[n - 1];
    //             assert!(worker_dev == to_dev);
    //             Some(receiver.clone())
    //         }
    //     };
    //     let param_senders_worker = match worker_id {
    //         0 => {
    //             let senders: Vec<_> = param_senders
    //                 .iter()
    //                 .map(|(to_dev, sender)| ((*to_dev).clone(), sender.clone()))
    //                 .collect();
    //             Some(senders)
    //         }
    //         _ => None,
    //     };
    //     let update_barrier_worker = update_barrier.clone();
    //     let worker_dev_thread = *worker_dev;

    //     thread::Builder::new()
    //         .name(format!("train_worker-{}", worker_id))
    //         .spawn(move || {
    //             // Load model params
    //             let mut vs = VarStore::new(worker_dev_thread);

    //             // Init model
    //             debug!("Initialize model on worker {}", worker_id);
    //             let model = {
    //                 let root = vs.root();
    //                 let model =
    //                     GqnModel::<TowerEncoder>::new(&root, frame_channels, param_channels);
    //                 let _ = root.zeros("step", &[]);
    //                 model
    //             };

    //             // Init optimizer
    //             let mut optimizer_opt = match worker_id {
    //                 0 => {
    //                     let opt = Adam::default().build(&vs, 1e-3).unwrap();
    //                     Some(opt)
    //                 }
    //                 _ => None,
    //             };

    //             loop {
    //                 match req_receiver.recv() {
    //                     Ok(WorkerAction::Forward((step, example))) => {
    //                         debug!("Forward pass on worker {}", worker_id);
    //                         let output = run_model(&model, example, step);
    //                         let resp = WorkerResponse::ForwardOutput(output);
    //                         match resp_sender_worker.send(resp) {
    //                             Err(err) => {
    //                                 error!("Worker {} failed: {:?}", worker_id, err);
    //                                 return;
    //                             }
    //                             _ => {}
    //                         }
    //                     }
    //                     Ok(WorkerAction::Backward((step, elbo_loss))) => {
    //                         debug!("Backward pass on worker {}", worker_id);

    //                         let begin = params::ADAM_LR_BEGIN;
    //                         let end = params::ADAM_LR_END;
    //                         let max_step = params::ANNEAL_LR_MAX;
    //                         let lr = begin
    //                             + (begin - end) * (1. - (step as f64 / max_step as f64).min(1.));

    //                         let opt = optimizer_opt.as_mut().unwrap();
    //                         opt.set_lr(lr);
    //                         opt.backward_step(&elbo_loss);
    //                     }
    //                     Ok(WorkerAction::LoadParams(path)) => {
    //                         debug!("Load model parameters to worker {}", worker_id);
    //                         vs.load(path).unwrap();

    //                         // Update step variable if necessary
    //                         if worker_id == 0 {
    //                             let step = {
    //                                 let root = vs.root();
    //                                 let step_tensor = root.get("step").unwrap();
    //                                 // Uncomment this for backward compatibility
    //                                 // let step_tensor = root.entry("step")
    //                                 //     .or_zeros(&[]);
    //                                 step_tensor.int64_value(&[])
    //                             };
    //                             let resp = WorkerResponse::Step(step);
    //                             match resp_sender_worker.send(resp) {
    //                                 Err(err) => {
    //                                     error!("Worker {} failed: {:?}", worker_id, err);
    //                                     return;
    //                                 }
    //                                 _ => {}
    //                             }
    //                         }
    //                     }
    //                     Ok(WorkerAction::SaveParams(path, step)) => {
    //                         debug!("Save model params from worker {}", worker_id);
    //                         {
    //                             let root = vs.root();
    //                             let mut step_tensor = root.get("step").unwrap();
    //                             tch::no_grad(|| step_tensor.init(Init::Const(step as f64)));
    //                         }
    //                         vs.save(path).unwrap();
    //                     }
    //                     Ok(WorkerAction::CopyParams) => {
    //                         match worker_id {
    //                             0 => {
    //                                 debug!("Send param copies from worker {}", worker_id);
    //                                 let vs_rc = Arc::new(vs);
    //                                 for (_to_dev, sender) in
    //                                     param_senders_worker.as_ref().unwrap().iter()
    //                                 {
    //                                     sender.send(vs_rc.clone()).unwrap();
    //                                 }
    //                                 update_barrier_worker.wait();
    //                                 vs = Arc::try_unwrap(vs_rc).unwrap();
    //                             }
    //                             _ => {
    //                                 debug!("Update params on worker {}", worker_id);
    //                                 {
    //                                     let receiver = param_receiver.as_ref().unwrap();
    //                                     let from_vs = receiver.recv().unwrap();
    //                                     vs.copy(&from_vs).unwrap();
    //                                 } // This scope makes sures deallocation of from_vs
    //                                 update_barrier_worker.wait();
    //                             }
    //                         }
    //                     }
    //                     Ok(WorkerAction::Terminate) => {
    //                         debug!("Worker {} finished", worker_id);
    //                         return;
    //                     }
    //                     Err(err) => {
    //                         error!("Worker {} failed: {:?}", worker_id, err);
    //                         return;
    //                     }
    //                 };
    //             }
    //         })?;

    //     req_senders.push(req_sender);
    // }

    // // Produce train examples
    // let global_instant = Instant::now();
    // let mut step = 0;

    // // Load model params
    // if let Some(path) = &config.model_file {
    //     if path.is_file() {
    //         info!("Load model file {:?}", path);
    //         for sender in req_senders.iter() {
    //             sender
    //                 .send(WorkerAction::LoadParams(path.to_path_buf()))
    //                 .unwrap();
    //         }

    //         // Update step count
    //         step = match resp_receiver.recv()? {
    //             WorkerResponse::Step(resp_step) => match config.initial_step {
    //                 Some(init_step) => init_step,
    //                 None => resp_step,
    //             },
    //             _ => panic!("Wrong response type"),
    //         }
    //     }
    // }

    // // Main loop
    // while !SHUTDOWN_FLAG.load(Ordering::SeqCst) {
    //     let examples = match train_iter.next() {
    //         Some(ret) => ret,
    //         None => break,
    //     };

    //     let step_instant = Instant::now();

    //     // Send examples to workers
    //     let n_examples = examples.len();
    //     for (sender, example) in req_senders.iter().zip(examples.into_iter()) {
    //         sender
    //             .send(WorkerAction::Forward((step as i64, example)))
    //             .unwrap();
    //     }

    //     // Receive outputs from workers
    //     let mut outputs = vec![];
    //     while outputs.len() < n_examples {
    //         let output = match resp_receiver.recv()? {
    //             WorkerResponse::ForwardOutput(output) => output,
    //             _ => panic!("Wrong response type"),
    //         };
    //         outputs.push(output);
    //     }

    //     // combine model outputs
    //     let mut elbo_loss_grad = outputs[0].elbo_loss.shallow_clone();
    //     let combined = combine_gqn_outputs(outputs, config.devices[0]);

    //     // Backward step
    //     tch::no_grad(|| elbo_loss_grad.copy_(&combined.elbo_loss));
    //     req_senders[0]
    //         .send(WorkerAction::Backward((step as i64, elbo_loss_grad)))
    //         .unwrap();

    //     // let workers update params
    //     for sender in req_senders.iter() {
    //         sender.send(WorkerAction::CopyParams).unwrap();
    //     }

    //     // Save model params
    //     if let Some(path) = &config.model_file {
    //         if step % config.save_steps == 0 {
    //             req_senders[0]
    //                 .send(WorkerAction::SaveParams(path.to_path_buf(), step))
    //                 .unwrap();
    //         }
    //     }

    //     // Log data
    //     if let Some(path) = &config.log_dir {
    //         if step % config.log_steps == 0 {
    //             // Save images
    //             if config.save_images {
    //                 save_images(step, &combined, path);
    //             }

    //             // Save model outputs
    //             save_model_outputs(step, &combined, path)?;
    //         }
    //     }

    //     // Write output
    //     info!(
    //         "step: {}\tglobal_elapsed: {}s\tstep_elapsed: {}ms\telbo_loss: {:.3}\ttarget_mse: {:.6}",
    //         step,
    //         global_instant.elapsed().as_secs(),
    //         step_instant.elapsed().as_millis(),
    //         combined.elbo_loss.double_value(&[]),
    //         combined.target_mse.double_value(&[]),
    //     );

    //     // Update step
    //     step += 1;
    // }

    // // Gracefully terminate workers
    // if let Some(path) = &config.model_file {
    //     req_senders[0]
    //         .send(WorkerAction::SaveParams(path.to_path_buf(), step))
    //         .unwrap();
    // }

    // for (n, sender) in req_senders.into_iter().enumerate() {
    //     debug!("Terminating train_worker-{}", n);
    //     sender.send(WorkerAction::Terminate).unwrap();
    // }

    Ok(())
}

fn combine_gqn_outputs(outputs: Vec<GqnModelOutput>, target_device: Device) -> GqnModelOutput {
    let combine_cat = |tensors: &[&Tensor]| {
        Tensor::cat(
            &tensors
                .iter()
                .map(|tensor| tensor.to_device(target_device))
                .collect::<Vec<_>>(),
            0,
        )
    };

    let combine_mean =
        |tensors: &[&Tensor]| combine_cat(tensors).mean1(&[], false, tensors[0].kind());

    // let tensor_to_vec = |tensor: &Tensor| {
    //     let buf_size = tensor.numel();
    //     let mut buf = vec![0_f32; buf_size as usize];
    //     tensor.copy_data(&mut buf, buf_size);
    //     buf
    // };

    let elbo_loss = combine_mean(
        &outputs
            .iter()
            .map(|resp| &resp.elbo_loss)
            .collect::<Vec<_>>(),
    );

    let target_mse = combine_mean(
        &outputs
            .iter()
            .map(|resp| &resp.target_mse)
            .collect::<Vec<_>>(),
    );

    let canvases = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.canvases)
            .collect::<Vec<_>>(),
    );

    let target_sample = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.target_sample)
            .collect::<Vec<_>>(),
    );

    let means_target = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.means_target)
            .collect::<Vec<_>>(),
    );

    let stds_target = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.stds_target)
            .collect::<Vec<_>>(),
    );

    let means_inf = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.means_inf)
            .collect::<Vec<_>>(),
    );

    let stds_inf = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.stds_inf)
            .collect::<Vec<_>>(),
    );

    let means_gen = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.means_gen)
            .collect::<Vec<_>>(),
    );

    let stds_gen = combine_cat(
        &outputs
            .iter()
            .map(|resp| &resp.stds_gen)
            .collect::<Vec<_>>(),
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

// fn save_images<P: AsRef<Path>>(step: i64, combined: &GqnModelOutput, log_dir: P) {
//     let size = combined.means_target.size();
//     let batch_size = size[0];
//     let height = size[2];
//     let width = size[3];

//     let min_val: Tensor = 255_f32.into();
//     let min_val = min_val.to_device(combined.means_target.device());

//     for batch_idx in 0..batch_size {
//         let result_image = (combined.means_target.select(0, batch_idx) * 255.)
//             .min1(&min_val)
//             .permute(&[1, 2, 0])
//             .to_kind(Kind::Uint8);

//         let buf_size = result_image.numel() as usize;
//         let mut buf = vec![0_u8; buf_size];
//         result_image.copy_data(&mut buf, buf_size as i64);

//         let filename = format!("{:0>10}-{:0>2}.jpg", step, batch_idx);
//         ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, buf)
//             .unwrap()
//             .save(log_dir.as_ref().join(filename))
//             .unwrap();
//     }
// }

// fn save_model_outputs<P: AsRef<Path>>(
//     step: i64,
//     combined: &GqnModelOutput,
//     log_dir: P,
// ) -> Fallible<()> {
//     let sys_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

//     let filename = format!("{:0>10}-{:0>13}.zip", step, sys_time);
//     let file_path = log_dir.as_ref().join(filename);

//     let data = vec![
//         ("elbo_loss", combined.elbo_loss.shallow_clone()),
//         ("target_mse", combined.target_mse.shallow_clone()),
//         ("target_sample", combined.target_sample.shallow_clone()),
//         ("means_target", combined.means_target.shallow_clone()),
//         ("stds_target", combined.stds_target.shallow_clone()),
//         ("means_inf", combined.means_inf.shallow_clone()),
//         ("stds_inf", combined.stds_inf.shallow_clone()),
//         ("means_gen", combined.means_gen.shallow_clone()),
//         ("stds_gen", combined.stds_gen.shallow_clone()),
//     ];

//     Tensor::save_multi(&data, file_path)?;

//     Ok(())
// }

// async fn train_worker(worker_index: usize, device: Device, frame_channels: usize, param_channels: usize) -> Fallible<()> {
//     // Load model params
//     let mut vs = VarStore::new(device);

//     // Init model
//     debug!("Initialize model on worker {}", worker_index);
//     let model = {
//         let root = vs.root();
//         let model = GqnModel::<TowerEncoder>::new(&root, frame_channels as i64, param_channels as i64);
//         let _ = root.zeros("step", &[]);
//         model
//     };

//     // Init optimizer
//     let mut optimizer_opt = match worker_index {
//         0 => {
//             let opt = Adam::default().build(&vs, 1e-3).unwrap();
//             Some(opt)
//         }
//         _ => None,
//     };

//     loop {
//         match req_receiver.recv() {
//             Ok(WorkerAction::Forward((step, example))) => {
//                 debug!("Forward pass on worker {}", worker_index);
//                 let output = run_model(&model, example, step);
//                 let resp = WorkerResponse::ForwardOutput(output);
//                 match resp_sender_worker.send(resp) {
//                     Err(err) => {
//                         error!("Worker {} failed: {:?}", worker_index, err);
//                         return;
//                     }
//                     _ => {}
//                 }
//             }
//             Ok(WorkerAction::Backward((step, elbo_loss))) => {
//                 debug!("Backward pass on worker {}", worker_index);

//                 let begin = params::ADAM_LR_BEGIN;
//                 let end = params::ADAM_LR_END;
//                 let max_step = params::ANNEAL_LR_MAX;
//                 let lr = begin + (begin - end) * (1. - (step as f64 / max_step as f64).min(1.));

//                 let opt = optimizer_opt.as_mut().unwrap();
//                 opt.set_lr(lr);
//                 opt.backward_step(&elbo_loss);
//             }
//             Ok(WorkerAction::LoadParams(path)) => {
//                 debug!("Load model parameters to worker {}", worker_index);
//                 vs.load(path).unwrap();

//                 // Update step variable if necessary
//                 if worker_index == 0 {
//                     let step = {
//                         let root = vs.root();
//                         let step_tensor = root.get("step").unwrap();
//                         // Uncomment this for backward compatibility
//                         // let step_tensor = root.entry("step")
//                         //     .or_zeros(&[]);
//                         step_tensor.int64_value(&[])
//                     };
//                     let resp = WorkerResponse::Step(step);
//                     match resp_sender_worker.send(resp) {
//                         Err(err) => {
//                             error!("Worker {} failed: {:?}", worker_index, err);
//                             return;
//                         }
//                         _ => {}
//                     }
//                 }
//             }
//             Ok(WorkerAction::SaveParams(path, step)) => {
//                 debug!("Save model params from worker {}", worker_index);
//                 {
//                     let root = vs.root();
//                     let mut step_tensor = root.get("step").unwrap();
//                     tch::no_grad(|| step_tensor.init(Init::Const(step as f64)));
//                 }
//                 vs.save(path).unwrap();
//             }
//             Ok(WorkerAction::CopyParams) => {
//                 match worker_index {
//                     0 => {
//                         debug!("Send param copies from worker {}", worker_index);
//                         let vs_rc = Arc::new(vs);
//                         for (_to_dev, sender) in param_senders_worker.as_ref().unwrap().iter() {
//                             sender.send(vs_rc.clone()).unwrap();
//                         }
//                         update_barrier_worker.wait();
//                         vs = Arc::try_unwrap(vs_rc).unwrap();
//                     }
//                     _ => {
//                         debug!("Update params on worker {}", worker_index);
//                         {
//                             let receiver = param_receiver.as_ref().unwrap();
//                             let from_vs = receiver.recv().unwrap();
//                             vs.copy(&from_vs).unwrap();
//                         } // This scope makes sures deallocation of from_vs
//                         update_barrier_worker.wait();
//                     }
//                 }
//             }
//             Ok(WorkerAction::Terminate) => {
//                 debug!("Worker {} finished", worker_index);
//                 return;
//             }
//             Err(err) => {
//                 error!("Worker {} failed: {:?}", worker_index, err);
//                 return;
//             }
//         };
//     }
// }
