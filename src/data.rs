use crate::common::*;

pub type GqnExample = HashMap<String, GqnFeature>;

pub enum GqnFeature {
    BytesList(Vec<Vec<u8>>),
    FloatList(Vec<f32>),
    FloatsList(Vec<Vec<f32>>),
    Int64List(Vec<i64>),
    BytesArray3(Array3<u8>),
    BytesArray3List(Vec<Array3<u8>>),
    Tensor(Tensor),
    DynamicImage(DynamicImage),
    DynamicImageList(Vec<DynamicImage>),
    RgbImage(RgbImage),
    RgbImageList(Vec<RgbImage>),
    None,
}

impl From<Feature> for GqnFeature {
    fn from(from: Feature) -> Self {
        match from {
            Feature::BytesList(list) => Self::BytesList(list),
            Feature::FloatList(list) => Self::FloatList(list),
            Feature::Int64List(list) => Self::Int64List(list),
            Feature::None => Self::None,
        }
    }
}

impl From<Vec<Vec<u8>>> for GqnFeature {
    fn from(from: Vec<Vec<u8>>) -> Self {
        Self::BytesList(from)
    }
}

impl From<Vec<f32>> for GqnFeature {
    fn from(from: Vec<f32>) -> Self {
        Self::FloatList(from)
    }
}

impl From<Vec<Vec<f32>>> for GqnFeature {
    fn from(from: Vec<Vec<f32>>) -> Self {
        Self::FloatsList(from)
    }
}

impl From<Vec<i64>> for GqnFeature {
    fn from(from: Vec<i64>) -> Self {
        Self::Int64List(from)
    }
}

impl From<Array3<u8>> for GqnFeature {
    fn from(from: Array3<u8>) -> Self {
        Self::BytesArray3(from)
    }
}

impl From<Vec<Array3<u8>>> for GqnFeature {
    fn from(from: Vec<Array3<u8>>) -> Self {
        Self::BytesArray3List(from)
    }
}

impl From<Tensor> for GqnFeature {
    fn from(from: Tensor) -> Self {
        Self::Tensor(from)
    }
}

impl From<DynamicImage> for GqnFeature {
    fn from(from: DynamicImage) -> Self {
        Self::DynamicImage(from)
    }
}

impl From<Vec<DynamicImage>> for GqnFeature {
    fn from(from: Vec<DynamicImage>) -> Self {
        Self::DynamicImageList(from)
    }
}

impl From<RgbImage> for GqnFeature {
    fn from(from: RgbImage) -> Self {
        Self::RgbImage(from)
    }
}

impl From<Vec<RgbImage>> for GqnFeature {
    fn from(from: Vec<RgbImage>) -> Self {
        Self::RgbImageList(from)
    }
}
