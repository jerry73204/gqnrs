use crate::{
    common::*,
    data::{GqnExample, GqnFeature},
};

pub trait MyFrom<T> {
    fn my_from(from: T) -> Self;
}

impl MyFrom<Example> for GqnExample {
    fn my_from(from: Example) -> Self {
        from.into_iter()
            .map(|(name, feature)| (name, GqnFeature::from(feature)))
            .collect()
    }
}
