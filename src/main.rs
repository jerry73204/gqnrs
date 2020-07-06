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
async fn main() -> Fallible<()> {
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

    // message types
    #[derive(Debug)]
    struct UploadMessage {
        worker_index: usize,
        output: GqnModelOutput,
        grads: Vec<Tensor>,
    }

    #[derive(Debug, Clone)]
    struct DownloadMessage {}

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

    // traning data stream
    let mut train_stream = Box::pin(dataset.train_stream(0)?); // TODO: initial step

    // TODO: use concurrent workers
    {
        let device = config.training.devices[0];
        let vs = VarStore::new(device);
        let model = {
            let root = vs.root();
            let model =
                GqnModelInit::new(frame_channels as i64, param_channels as i64).build(&root);
            let _ = root.zeros("step", &[]);
            model
        };
        let variables = vs.variables().into_iter().collect::<Vec<_>>();
        let trainable_variables = variables
            .iter()
            .map(|(_name, var)| var.shallow_clone())
            .collect::<Vec<_>>();
        // let trainable_variables = vs.trainable_variables();
        let mut step = 0;

        while let Some(result) = train_stream.next().await {
            let input = result?.to_device(device);
            let output = model(&input, true);
            let mean_elbo_loss = output.elbo_loss.mean(Kind::Float);

            let grads = Tensor::run_backward(&[mean_elbo_loss], &trainable_variables, true, true);

            izip!(variables.iter(), grads.iter()).enumerate().for_each(
                |(index, ((name, _), grad))| {
                    println!("{}\t{}", index, name);
                    let _ = grad.kind();
                    // println!("{:?}", grad);
                },
            );

            step += 1;
        }
    }

    // let feed_future = async move {
    //     info!("starded feeding worker");
    //     let mut train_stream = Box::pin(dataset.train_stream(0)?); // TODO: initial step
    //     let mut step: usize = 0;

    //     loop {
    //         info!("step: {}", step);

    //         let mut inputs = vec![];
    //         for _ in 0..num_workers {
    //             inputs.push(train_stream.next().await.unwrap()?);
    //         }

    //         for (data_tx, input) in data_tx_set.iter_mut().zip(inputs.into_iter()) {
    //             if let Err(_) = data_tx.send(DataMessage { step, input }).await {
    //                 panic!("please report bug");
    //             }
    //         }

    //         let (step_, overflow) = step.overflowing_add(1);
    //         if overflow {
    //             warn!("step value overflow");
    //         }
    //         step = step_;
    //     }

    //     Fallible::Ok(())
    // };

    // let train_futures = config
    //     .training
    //     .devices
    //     .iter()
    //     .map(ToOwned::to_owned)
    //     .enumerate()
    //     .map(|(worker_index, device)| {
    //         info!("starded training worker {}", worker_index);
    //         let is_master = worker_index == 0;
    //         let mut data_rx = data_rx_set.remove(&worker_index).unwrap();
    //         let mut upload_tx = upload_tx.clone();
    //         let mut upload_rx_opt = if is_master {
    //             Some(upload_rx_opt.take().unwrap())
    //         } else {
    //             None
    //         };
    //         let mut download_tx_opt = if is_master {
    //             Some(download_tx.clone())
    //         } else {
    //             None
    //         };
    //         let mut download_rx = download_tx.subscribe();

    //         async move {
    //             let vs = VarStore::new(device);
    //             let model = {
    //                 let root = vs.root();
    //                 let model = GqnModelInit::new(frame_channels as i64, param_channels as i64)
    //                     .build(&root);
    //                 let _ = root.zeros("step", &[]);
    //                 model
    //             };
    //             let trainable_variables = vs.trainable_variables();

    //             while let Some(data_msg) = data_rx.recv().await {
    //                 let DataMessage { step, input } = data_msg;
    //                 let input = input.to_device(device);
    //                 let output = model(&input, true);

    //                 let mean_elbo_loss = output.elbo_loss.mean(Kind::Float);

    //                 // compute gradient

    //                 // DEBUG
    //                 vs.variables()
    //                     .into_iter()
    //                     .enumerate()
    //                     .for_each(|(index, (name, tensor))| {
    //                         println!("{}\t{}\t{:?}", index, name, tensor);
    //                     });

    //                 let _grads = Tensor::run_backward(
    //                     &[mean_elbo_loss],
    //                     &trainable_variables,
    //                     // &vs.variables()
    //                     //     .into_iter()
    //                     //     .map(|(_, tensor)| tensor)
    //                     //     .collect::<Vec<_>>(),
    //                     true,
    //                     true,
    //                 );

    //                 let grads = trainable_variables
    //                     .iter()
    //                     .map(|tensor| tensor.grad().detach().copy())
    //                     .collect::<Vec<_>>();

    //                 // upload to master
    //                 {
    //                     let upload_msg = UploadMessage {
    //                         worker_index,
    //                         output,
    //                         grads,
    //                     };
    //                     if let Err(_) = upload_tx.send(upload_msg).await {
    //                         panic!("please report bug");
    //                     }
    //                 }

    //                 // process outcomes in master
    //                 if is_master {
    //                     let upload_rx = upload_rx_opt.as_mut().unwrap();
    //                     let download_tx = download_tx_opt.as_mut().unwrap();

    //                     let instant = Instant::now();
    //                     let upload_msgs = {
    //                         let mut upload_msgs = vec![];
    //                         for _ in 0..num_workers {
    //                             upload_msgs.push(upload_rx.recv().await.unwrap());
    //                         }
    //                         upload_msgs
    //                     };

    //                     // DEBUG
    //                     {
    //                         let num_msgs = upload_msgs.len();
    //                         let num_grads = upload_msgs[0].grads.len();

    //                         for grad_index in 0..num_grads {
    //                             for msg_index in 0..num_msgs {
    //                                 let grad = &upload_msgs[msg_index].grads[grad_index];
    //                                 dbg!(msg_index, grad_index, grad);
    //                             }
    //                         }
    //                     }

    //                     // upload_msgs
    //                     //     .iter()
    //                     //     .map(|msg| {
    //                     //         msg.grads
    //                     //             .iter()
    //                     //             .enumerate()
    //                     //             .map(|(index, tensor)| {
    //                     //                 let tensor = tensor.clone();
    //                     //                 dbg!(index, tensor.size());
    //                     //                 let tensor = tensor.to_device(device);
    //                     //                 tensor
    //                     //             })
    //                     //             .collect::<Vec<_>>()
    //                     //     })
    //                     //     .collect::<Vec<_>>();

    //                     // upload_msgs.iter().map(|msg| msg.output.to_device(device));

    //                     // TODO

    //                     let download_msg = DownloadMessage {};
    //                     if let Err(_) = download_tx.send(download_msg) {
    //                         panic!("please report bug");
    //                     }
    //                 }

    //                 // download from master
    //                 {
    //                     let download_msg = download_rx.recv().await.unwrap();
    //                 }
    //             }

    //             Fallible::Ok(())
    //         }
    //     })
    //     .map(async_std::task::spawn);

    // futures::future::try_join(feed_future, futures::future::try_join_all(train_futures)).await?;

    Ok(())
}
