use crate::{common::*, model::GqnModelOutput};

#[derive(Debug)]
pub enum WorkerAction {
    Forward((i64, Example)),
    Backward((i64, Tensor)),
    CopyParams,
    LoadParams(PathBuf),
    SaveParams(PathBuf, i64),
    Terminate,
}

#[derive(Debug)]
pub enum WorkerResponse {
    ForwardOutput(GqnModelOutput),
    Step(i64),
}
