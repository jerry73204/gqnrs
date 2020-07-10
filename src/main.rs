use gqnrs::{
    common::*,
    config::{Config, DatasetConfig, DeepMindDatasetConfig},
    dataset,
    model::{GqnModelInit, GqnModelInput, GqnModelOutput},
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
async fn main() -> Result<()> {
    pretty_env_logger::init();

    // Set signal handler
    // ctrlc::set_handler(|| {
    //     warn!("Interrupted by user");
    //     SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    // })?;

    // Parse arguments
    let args: Args = argh::from_env();

    // load config
    let config = Config::open(&args.config)?;

    // init log dir
    if config.logging.enabled {
        async_std::fs::create_dir_all(&config.logging.log_dir).await?;
    }

    // Load dataset
    info!("loading dataset");

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

    info!("dataset is ready");

    // types
    #[derive(Debug)]
    struct MasterContext {
        pub optimizer: nn::Optimizer<nn::Adam>,
        pub upload_rx: mpsc::Receiver<UploadMessage>,
        pub download_tx: broadcast::Sender<DownloadMessage>,
    }

    #[derive(Debug)]
    struct UploadMessage {
        worker_index: usize,
        output: GqnModelOutput,
        grads: Vec<Tensor>,
    }

    #[derive(Debug)]
    struct DownloadMessage {
        weights: Vec<Tensor>,
    }

    impl Clone for DownloadMessage {
        fn clone(&self) -> Self {
            let Self { weights } = self;
            Self {
                weights: weights.shallow_clone(),
            }
        }
    }

    #[derive(Debug)]
    struct DataMessage {
        pub step: usize,
        pub input: GqnModelInput,
    }

    // channels among workers
    let num_workers = config.training.devices.len();
    let (upload_tx, upload_rx) = mpsc::channel::<UploadMessage>(num_workers);
    let (download_tx, _download_rx) = broadcast::channel::<DownloadMessage>(num_workers);
    let mut upload_rx_opt = Some(upload_rx);
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
        info!("starded feeding worker");
        let mut train_stream = Box::pin(dataset.train_stream(0)?) // TODO: initial step
            .chunks(num_workers)
            .map(|chunk_of_results| -> Result<_> {
                let chunk = chunk_of_results.into_iter().collect::<Result<Vec<_>>>()?;
                Ok(chunk)
            })
            .overflowing_enumerate();

        while let Some((step, result)) = train_stream.next().await {
            let inputs = result?;
            info!("step: {}", step);

            for (data_tx, input) in data_tx_set.iter_mut().zip(inputs.into_iter()) {
                if let Err(_) = data_tx.send(DataMessage { step, input }).await {
                    panic!("please report bug");
                }
            }
        }

        Result::<_, Error>::Ok(())
    };

    let train_futures = config
        .training
        .devices
        .iter()
        .map(ToOwned::to_owned)
        .enumerate()
        .map(|(worker_index, device)| {
            info!("starded training worker {}", worker_index);
            let is_master = worker_index == 0;
            let mut data_rx = data_rx_set.remove(&worker_index).unwrap();
            let mut upload_tx = upload_tx.clone();
            let mut download_rx = download_tx.subscribe();

            let vs = VarStore::new(device);
            let model = {
                let root = vs.root();
                let model =
                    GqnModelInit::new(frame_channels as i64, param_channels as i64).build(&root);
                let _ = root.zeros("step", &[]);
                model
            };
            let mut trainable_variables = vs.trainable_variables();
            let mut master_context_opt = if is_master {
                Some(MasterContext {
                    optimizer: nn::Adam::default().build(&vs, 1e-3).unwrap(),
                    upload_rx: upload_rx_opt.take().unwrap(),
                    download_tx: download_tx.clone(),
                })
            } else {
                None
            };

            async move {
                while let Some(data_msg) = data_rx.recv().await {
                    let DataMessage { step, input } = data_msg;
                    let input = input.to_device(device);
                    let output = model(&input, true);

                    // compute gradient
                    let mean_elbo_loss = output.elbo_loss.mean(Kind::Float);
                    info!(
                        "worker: {},\tloss: {}",
                        worker_index,
                        f32::from(&mean_elbo_loss)
                    );
                    mean_elbo_loss.backward();
                    let grads = trainable_variables
                        .iter()
                        .map(|tensor| tensor.grad())
                        .collect::<Vec<_>>();

                    // upload to master
                    {
                        let upload_msg = UploadMessage {
                            worker_index,
                            output,
                            grads,
                        };
                        if let Err(_) = upload_tx.send(upload_msg).await {
                            panic!("please report bug");
                        }
                    }

                    // process outcomes in master
                    if is_master {
                        let MasterContext {
                            optimizer,
                            upload_rx,
                            download_tx,
                        } = master_context_opt.as_mut().unwrap();

                        let upload_msgs = {
                            let mut upload_msgs = vec![];
                            for _ in 0..num_workers {
                                upload_msgs.push(upload_rx.recv().await.unwrap());
                            }
                            upload_msgs
                        };

                        tch::no_grad(|| {
                            // compute mean gradients
                            let mean_grads = {
                                let num_msgs = upload_msgs.len();
                                let num_grads = upload_msgs[0].grads.len();

                                (0..num_grads)
                                    .map(|grad_index| {
                                        let is_defined =
                                            &upload_msgs[0].grads[grad_index].defined();
                                        if !is_defined {
                                            return None;
                                        }

                                        let mean_grad = (0..num_msgs)
                                            .map(|msg_index| {
                                                upload_msgs[msg_index].grads[grad_index]
                                                    .shallow_clone()
                                            })
                                            .fold1(|lhs, rhs| {
                                                lhs.to_device(device) + rhs.to_device(device)
                                            })
                                            .unwrap()
                                            / num_msgs as f64;

                                        Some(mean_grad)
                                    })
                                    .collect::<Vec<_>>()
                            };

                            // assign mean gradients to gradient tensors
                            trainable_variables
                                .iter()
                                .zip_eq(mean_grads.iter())
                                .filter_map(|(var, grad_opt)| {
                                    grad_opt.as_ref().map(|grad| (var, grad))
                                })
                                .for_each(|(var, grad)| {
                                    var.grad().copy_(&grad);
                                });
                        });
                        // optimization step
                        optimizer.step();

                        // broadcast updated weights
                        let download_msg = DownloadMessage {
                            weights: trainable_variables.shallow_clone(),
                        };
                        if let Err(_) = download_tx.send(download_msg) {
                            panic!("please report bug");
                        }
                    }

                    // download from master
                    {
                        let DownloadMessage { weights } = download_rx.recv().await.unwrap();

                        // update weights
                        tch::no_grad(|| {
                            trainable_variables
                                .iter_mut()
                                .zip_eq(weights.iter())
                                .for_each(|(var, weight)| {
                                    var.copy_(weight);
                                });
                        });
                    }
                }

                Result::<_, Error>::Ok(())
            }
        })
        .map(async_std::task::spawn);

    futures::future::try_join(feed_future, futures::future::try_join_all(train_futures)).await?;

    Ok(())
}
