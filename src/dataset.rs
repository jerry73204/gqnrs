use std::io;
use std::io::{Read, Seek};
use std::error;
use std::path;
use std::fs;
use std::collections;
use serde::Deserialize;
use glob::glob;
use yaml_rust::YamlLoader;
use byteorder::{ReadBytesExt, LittleEndian};
use crc::crc32;
use image::jpeg::JPEGDecoder;
use image::ImageDecoder;
use tch::Tensor;
use rayon::prelude::*;
use crate::tf_proto::example::Example;

#[derive(Deserialize, Debug)]
pub struct GqnDataSetInfo
{
    basepath: String,
    train_size: u64,
    test_size: u64,
    frame_size: u64,
    sequence_size: u64,
}

pub fn load_gqn_tfrecord(name: &str, dataset_dir: &path::Path) -> Result<(), Box<error::Error>>
{
    let dataset_spec = &YamlLoader::load_from_str(include_str!("dataset.yaml"))?[0];
    let dataset_info = &dataset_spec["dataset"][name];
    let num_camera_params = dataset_spec["num_camera_params"].as_i64().unwrap();
    let train_size = dataset_info["train_size"].as_i64().unwrap();
    let test_size = dataset_info["test_size"].as_i64().unwrap();
    let frame_size = dataset_info["frame_size"].as_i64().unwrap();
    let sequence_size = dataset_info["sequence_size"].as_i64().unwrap();

    let train_dir = dataset_dir.join("train");
    let test_dir = dataset_dir.join("test");

    let mut train_tfrecord_paths = Vec::<path::PathBuf>::new();
    for entry in glob(train_dir.join("*.tfrecord").to_str().unwrap())?
    {
        train_tfrecord_paths.push(entry?);
    }

    let mut test_tfrecord_paths = Vec::<path::PathBuf>::new();
    for entry in glob(test_dir.join("*.tfrecord").to_str().unwrap())?
    {
        test_tfrecord_paths.push(entry?);
    }

    let train_tfrecord_indexes: collections::HashMap<_, _> = train_tfrecord_paths.par_iter().map(|path| {
        println!("Loading {}", path.display());
        let record_index = build_tfrecord_index(&path, false).unwrap();
        (path, record_index)
    }).collect();

    let test_tfrecord_indexes: collections::HashMap<_, _> = test_tfrecord_paths.par_iter().map(|path| {
        println!("Loading {}", path.display());
        let record_index = build_tfrecord_index(&path, false).unwrap();
        (path, record_index)
    }).collect();

    for (path, index) in train_tfrecord_indexes
    {
        for (offset, len) in index
        {
            println!("{} {} {}", path.display(), offset, len);
        }
    }

    Ok(())
}

pub fn parse_example_record(
    buf: &[u8],
    sequence_size: i64,
    num_camera_params: i64,
    frame_size: i64,
    channels: i64,
) -> Result<(Tensor, Tensor), Box<error::Error>>
{
    let make_corrupted_error = || {
        io::Error::new(io::ErrorKind::Other, "corrupted error")
    };

    let example: Example = protobuf::parse_from_bytes(buf)?;
    let features = example.get_features().get_feature();
    let frames = features["frames"]
        .get_bytes_list()
        .get_value();
    let cameras = features["cameras"]
        .get_float_list()
        .get_value();

    if sequence_size != frames.len() as i64 ||
        sequence_size * num_camera_params != cameras.len() as i64
    {
        return Err(Box::new(make_corrupted_error()));
    }

    let mut decoded_frames: Vec<f32> = Vec::new();
    for ind in 0..sequence_size
    {
        let jpeg_bytes = &frames[ind as usize];
        let mut decoder = JPEGDecoder::new(io::Cursor::new(jpeg_bytes));
        let image = decoder.read_image()?;

        let actual_size = match image
        {
            image::DecodingResult::U8(ref data) => data.len(),
            image::DecodingResult::U16(ref data) => data.len(),
        };
        let expect_size: i64 = frame_size * frame_size * channels;
        if expect_size != actual_size as i64
        {
            return Err(Box::new(make_corrupted_error()));
        }

        match image
        {
            image::DecodingResult::U8(ref data) => {
                let pixels = data.into_iter()
                    .map(|v| *v as f32 / 255.);
                decoded_frames.extend(pixels);
            }
            image::DecodingResult::U16(ref data) => {
                let pixels = data.into_iter()
                    .map(|v| *v as f32 / 255.);
                decoded_frames.extend(pixels);
            }
        };
    }

    let mut frames_tensor = Tensor::from(decoded_frames.as_slice());
    let mut cameras_tensor = Tensor::from(cameras);

    frames_tensor = frames_tensor.reshape(&[sequence_size, frame_size, frame_size, channels])
        .permute(&[0, 3, 1, 2]);           // channel last to channel first
    cameras_tensor = cameras_tensor.reshape(&[sequence_size, num_camera_params]);

    Ok((frames_tensor, cameras_tensor))
}

pub fn build_tfrecord_index(path: &path::Path, check_integrity: bool) -> Result<Vec<(u64, u64)>, Box<error::Error>>
{
    let make_corrupted_error = || { io::Error::new(io::ErrorKind::Other, "corrupted error") };
    let make_truncated_error = || { io::Error::new(io::ErrorKind::UnexpectedEof, "corrupted error") };

    let mut record_index: Vec<(u64, u64)> = Vec::new();
    let mut file = fs::File::open(path)?;

    let checksum = |buf: &[u8]| {
        let cksum = crc32::checksum_castagnoli(buf);
        ((cksum >> 15) | (cksum << 17)).wrapping_add(0xa282ead8u32)
    };

    let try_read_len = |file: &mut fs::File| -> Result<Option<u64>, Box<error::Error>> {
        let mut len_buf = [0u8; 8];

        match file.read(&mut len_buf)
        {
            Ok(0) => Ok(None),
            Ok(n) if n == len_buf.len() => {
                let len = (&len_buf[..]).read_u64::<LittleEndian>()?;

                if check_integrity
                {
                    let answer_cksum = file.read_u32::<LittleEndian>()?;
                    if answer_cksum == checksum(&len_buf)
                    {
                        Ok(Some(len))
                    }
                    else
                    {
                        Err(Box::new(make_corrupted_error()))
                    }
                }
                else
                {
                    file.seek(io::SeekFrom::Current(4))?;
                    Ok(Some(len))
                }
            }
            Ok(_) => Err(Box::new(make_truncated_error())),
            Err(e) => Err(Box::new(e)),
        }
    };

    let verify_record_integrity = |file: &mut fs::File, len: u64| -> Result<(), Box<error::Error>> {
        let mut buf = Vec::<u8>::new();
        file.take(len).read_to_end(&mut buf)?;
        let answer_cksum = file.read_u32::<LittleEndian>()?;

        if answer_cksum == checksum(&buf.as_slice())
        {
            Ok(())
        }
        else
        {
            Err(Box::new(make_corrupted_error()))
        }
    };

    loop
    {
        match try_read_len(&mut file)?
        {
            None => break,
            Some(len) => {
                let offset = file.seek(io::SeekFrom::Current(0))?;
                if check_integrity
                {
                    verify_record_integrity(&mut file, len)?;
                }
                else
                {
                    file.seek(io::SeekFrom::Current(len as i64 + 4))?;
                }
                record_index.push((offset, len));
            }
        }
    }

    Ok(record_index)
}
