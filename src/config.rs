use crate::common::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_file: Option<PathBuf>,
    pub log_dir: Option<PathBuf>,
    pub save_steps: i64,
    pub log_steps: i64,
    pub batch_size: NonZeroUsize,
    pub initial_step: Option<i64>,
    #[serde(
        serialize_with = "serialize_devices",
        deserialize_with = "deserialize_devices",
        default = "default_devices"
    )]
    pub devices: Vec<Device>,
    pub dataset: DatasetConfig,
    pub save_images: bool,
}

impl Config {
    pub fn open<P>(path: P) -> Fallible<Self>
    where
        P: AsRef<Path>,
    {
        let text = fs::read_to_string(path)?;
        let config = json5::from_str(&text)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetConfig {
    #[serde(rename = "deepmind")]
    DeepMind(DeepMindDatasetConfig),
    #[serde(rename = "file")]
    File(FileDatasetConfig),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepMindDatasetConfig {
    pub frame_channels: NonZeroUsize,
    pub dataset_dir: PathBuf,
    pub train_size: NonZeroUsize,
    pub test_size: NonZeroUsize,
    pub frame_size: usize,
    pub sequence_size: NonZeroUsize,
    pub check_integrity: bool,
    raw_config: deepmind_config::RawDeepMindDataset,
}

impl Serialize for DeepMindDatasetConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.raw_config.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DeepMindDatasetConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use deepmind_config::{DeepMindConfig, DeepMindConfigList, RawDeepMindDataset};
        let raw_config = RawDeepMindDataset::deserialize(deserializer)?;
        let RawDeepMindDataset {
            base_dir,
            config_file,
            dataset_name,
            ..
        } = &raw_config;
        let RawDeepMindDataset {
            check_integrity, ..
        } = raw_config;

        let config_text = fs::read_to_string(config_file)
            .map_err(|err| D::Error::custom(format!("{:?}", err)))?;
        let DeepMindConfigList {
            dataset: dataset_list,
        } = serde_yaml::from_str(&config_text)
            .map_err(|err| D::Error::custom(format!("{:?}", err)))?;

        let DeepMindConfig {
            frame_channels,
            basepath,
            train_size,
            test_size,
            frame_size,
            sequence_size,
        } = dataset_list
            .get(dataset_name)
            .ok_or_else(|| {
                D::Error::custom(format!(
                    r#"the dataset name "{}" does not exist"#,
                    dataset_name
                ))
            })?
            .clone();

        Ok(Self {
            frame_channels,
            dataset_dir: base_dir.join(basepath),
            train_size,
            test_size,
            frame_size,
            sequence_size,
            raw_config,
            check_integrity,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileDatasetConfig {
    pub input_dir: PathBuf,
    pub time_step: f64,
    pub sequence_size: NonZeroUsize,
    pub frame_size: NonZeroUsize,
}

mod deepmind_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct DeepMindConfigList {
        pub dataset: HashMap<String, DeepMindConfig>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct DeepMindConfig {
        pub frame_channels: NonZeroUsize,
        pub basepath: String,
        pub train_size: NonZeroUsize,
        pub test_size: NonZeroUsize,
        #[serde(
            serialize_with = "serialize_frame_size",
            deserialize_with = "deserialize_frame_size"
        )]
        pub frame_size: usize,
        pub sequence_size: NonZeroUsize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RawDeepMindDataset {
        pub base_dir: PathBuf,
        pub config_file: PathBuf,
        pub dataset_name: String,
        pub check_integrity: bool,
    }

}

fn default_devices() -> Vec<Device> {
    vec![Device::cuda_if_available()]
}

fn serialize_devices<S>(devices: &Vec<Device>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let device_names = devices
        .into_iter()
        .map(|device| {
            let text = match device {
                Device::Cpu => "cpu".into(),
                Device::Cuda(n) => format!("cuda({})", n),
            };
            text
        })
        .collect::<Vec<_>>();
    device_names.serialize(serializer)
}

fn deserialize_devices<'de, D>(deserializer: D) -> Result<Vec<Device>, D::Error>
where
    D: Deserializer<'de>,
{
    let device_names = Vec::<String>::deserialize(deserializer)?;
    let devices = device_names
        .into_iter()
        .map(|name| {
            let device = match name.as_str() {
                "cpu" => Device::Cpu,
                _ => {
                    let prefix = "cuda(";
                    let suffix = ")";
                    if name.starts_with(prefix) && name.ends_with(suffix) {
                        let number: usize = name[(prefix.len())..(name.len() - suffix.len())]
                            .parse()
                            .map_err(|_err| {
                                D::Error::custom(format!("invalid device name {}", name))
                            })?;
                        Device::Cuda(number)
                    } else {
                        return Err(D::Error::custom(""));
                    }
                }
            };
            Ok(device)
        })
        .collect::<Result<Vec<_>, D::Error>>()?;

    Ok(devices)
}

fn serialize_frame_size<S>(frame_size: &usize, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if *frame_size < 2 {
        return Err(S::Error::custom("frame_size must be at least 2"));
    }
    frame_size.serialize(serializer)
}

fn deserialize_frame_size<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let size = usize::deserialize(deserializer)?;
    if size < 2 {
        return Err(D::Error::custom("frame_size must be at least 2"));
    }
    Ok(size)
}
