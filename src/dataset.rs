extern crate serde_json;
extern crate serde;
extern crate image;

use std::error::Error;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use serde::Deserialize;
use image::GenericImageView;

#[derive(Deserialize, Debug)]
pub struct GqnDataSetInfo
{
    basepath: String,
    train_size: u64,
    test_size: u64,
    frame_size: u64,
    sequence_size: u64,
}

pub struct GqnDataSet
{
    info: GqnDataSetInfo,
}


pub fn load_gqn_dataset(dataset_dir: &Path) -> Result<GqnDataSet, Box<Error>>
{
    let train_dir = dataset_dir.join("train");
    let test_dir = dataset_dir.join("test");
    let info_file = dataset_dir.join("info.json");

    // Load dataset info
    let info_file = File::open(info_file)?;
    let info_reader = BufReader::new(info_file);
    let info: GqnDataSetInfo = serde_json::from_reader(info_reader)?;

    let train_frame_dir = train_dir.join("frame");
    for sample_ind in 0..info.train_size
    {
        for seq_ind in 0..info.sequence_size
        {
            let fname = format!("frame_{}_{}.jpg", sample_ind, seq_ind);
            let image_path = train_frame_dir.join(fname);
            let image = image::open(image_path)?;
            let rgb_image = image.as_rgb8().unwrap();
        }
    }

    Ok(GqnDataSet {info})
}
