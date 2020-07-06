use super::{
    decoder::{GqnDecoder, GqnDecoderOutput},
    encoder, params,
};

use crate::{
    common::*,
    dist::{Normal, Rv},
    objective,
};

#[derive(Debug)]
pub struct GqnModelInput {
    pub context_frames: Tensor,
    pub target_frame: Tensor,
    pub context_params: Tensor,
    pub query_params: Tensor,
    pub step: usize,
}

impl GqnModelInput {
    pub fn to_device(&self, device: Device) -> Self {
        let Self {
            context_frames,
            target_frame,
            context_params,
            query_params,
            step,
        } = self;

        Self {
            context_frames: context_frames.to_device(device),
            target_frame: target_frame.to_device(device),
            context_params: context_params.to_device(device),
            query_params: query_params.to_device(device),
            step: *step,
        }
    }
}

#[derive(Debug)]
pub struct GqnModelOutput {
    pub elbo_loss: Tensor,
    pub target_mse: Tensor,
    pub target_sample: Tensor,
    pub means_target: Tensor,
    pub stds_target: Tensor,
    pub canvases: Tensor,
    pub means_inf: Tensor,
    pub stds_inf: Tensor,
    pub means_gen: Tensor,
    pub stds_gen: Tensor,
}

impl GqnModelOutput {
    pub fn to_device(&self, device: Device) -> Self {
        let Self {
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
        } = self;

        Self {
            elbo_loss: elbo_loss.to_device(device),
            target_mse: target_mse.to_device(device),
            target_sample: target_sample.to_device(device),
            means_target: means_target.to_device(device),
            stds_target: stds_target.to_device(device),
            canvases: canvases.to_device(device),
            means_inf: means_inf.to_device(device),
            stds_inf: stds_inf.to_device(device),
            means_gen: means_gen.to_device(device),
            stds_gen: stds_gen.to_device(device),
        }
    }

    pub fn shallow_clone(&self) -> Self {
        let Self {
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
        } = self;

        Self {
            elbo_loss: elbo_loss.shallow_clone(),
            target_mse: target_mse.shallow_clone(),
            target_sample: target_sample.shallow_clone(),
            means_target: means_target.shallow_clone(),
            stds_target: stds_target.shallow_clone(),
            canvases: canvases.shallow_clone(),
            means_inf: means_inf.shallow_clone(),
            stds_inf: stds_inf.shallow_clone(),
            means_gen: means_gen.shallow_clone(),
            stds_gen: stds_gen.shallow_clone(),
        }
    }

    pub fn cat<T>(outputs: &[T]) -> Self
    where
        T: Borrow<Self>,
    {
        let outputs = outputs.iter().map(|out| out.borrow()).collect::<Vec<_>>();
        let elbo_loss = Tensor::cat(
            &outputs.iter().map(|out| &out.elbo_loss).collect::<Vec<_>>(),
            0,
        );
        let target_mse = Tensor::cat(
            &outputs
                .iter()
                .map(|out| &out.target_mse)
                .collect::<Vec<_>>(),
            0,
        );
        let target_sample = Tensor::cat(
            &outputs
                .iter()
                .map(|out| &out.target_sample)
                .collect::<Vec<_>>(),
            0,
        );
        let means_target = Tensor::cat(
            &outputs
                .iter()
                .map(|out| &out.means_target)
                .collect::<Vec<_>>(),
            0,
        );
        let stds_target = Tensor::cat(
            &outputs
                .iter()
                .map(|out| &out.stds_target)
                .collect::<Vec<_>>(),
            0,
        );
        let canvases = Tensor::cat(
            &outputs.iter().map(|out| &out.canvases).collect::<Vec<_>>(),
            0,
        );
        let means_inf = Tensor::cat(
            &outputs.iter().map(|out| &out.means_inf).collect::<Vec<_>>(),
            0,
        );
        let stds_inf = Tensor::cat(
            &outputs.iter().map(|out| &out.stds_inf).collect::<Vec<_>>(),
            0,
        );
        let means_gen = Tensor::cat(
            &outputs.iter().map(|out| &out.means_gen).collect::<Vec<_>>(),
            0,
        );
        let stds_gen = Tensor::cat(
            &outputs.iter().map(|out| &out.stds_gen).collect::<Vec<_>>(),
            0,
        );

        Self {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GqnEncoderKind {
    Tower,
    Pool,
}

#[derive(Debug, Clone)]
pub struct GqnModelInit {
    pub frame_channels: i64,
    pub param_channels: i64,
    pub encoder_kind: GqnEncoderKind,
    pub seq_len: i64,
    pub enc_channels: i64,
    pub noise_channels: i64,
    pub lstm_output_channels: i64,
    pub lstm_canvas_channels: i64,
    pub lstm_kernel_size: i64,
    pub eta_internal_kernel_size: i64,
    pub eta_external_kernel_size: i64,
}

impl GqnModelInit {
    pub fn new(frame_channels: i64, param_channels: i64) -> Self {
        Self {
            frame_channels,
            param_channels,
            encoder_kind: GqnEncoderKind::Tower,
            seq_len: params::SEQ_LENGTH,
            enc_channels: params::ENC_CHANNELS,
            noise_channels: params::Z_CHANNELS,
            lstm_output_channels: params::LSTM_OUTPUT_CHANNELS,
            lstm_canvas_channels: params::LSTM_CANVAS_CHANNELS,
            lstm_kernel_size: params::LSTM_KERNEL_SIZE,
            eta_internal_kernel_size: params::ETA_INTERNAL_KERNEL_SIZE,
            eta_external_kernel_size: params::ETA_EXTERNAL_KERNEL_SIZE,
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&GqnModelInput, bool) -> GqnModelOutput + Send>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let device = path.device();

        let Self {
            frame_channels,
            param_channels,
            encoder_kind,
            seq_len,
            enc_channels,
            noise_channels,
            lstm_output_channels,
            lstm_canvas_channels,
            lstm_kernel_size,
            eta_internal_kernel_size,
            eta_external_kernel_size,
        } = self;

        let encoder = match encoder_kind {
            GqnEncoderKind::Tower => {
                encoder::tower_encoder(path / "encoder", enc_channels, param_channels)
            }
            GqnEncoderKind::Pool => {
                encoder::pool_encoder(path / "encoder", enc_channels, param_channels)
            }
        };

        let decoder = GqnDecoder::new(
            path / "decoder", // path
            seq_len,          // num layers
            true,             // biases
            enc_channels,
            param_channels,
            noise_channels,
            lstm_output_channels,
            lstm_canvas_channels,
            frame_channels,
            lstm_kernel_size,
            eta_internal_kernel_size,
            eta_external_kernel_size,
        );

        Box::new(
            move |input: &GqnModelInput, train: bool| -> GqnModelOutput {
                let GqnModelInput {
                    context_frames,
                    target_frame,
                    context_params,
                    query_params,
                    step,
                } = input;
                let step = *step;

                // Pack encoder input, melt batch and seq dimensions
                let (batch_size, seq_size, channels, height, width) = {
                    let context_frames_size = context_frames.size();
                    let batch_size = context_frames_size[0];
                    let seq_size = context_frames_size[1];
                    let channels = context_frames_size[2];
                    let height = context_frames_size[3];
                    let width = context_frames_size[4];
                    (batch_size, seq_size, channels, height, width)
                };

                let n_params = {
                    let context_poses_size = context_params.size();
                    let batch_size2 = context_poses_size[0];
                    let seq_size2 = context_poses_size[1];
                    let n_params = context_poses_size[2];
                    assert!(batch_size == batch_size2 && seq_size == seq_size2);
                    n_params
                };

                let packed_context_frames =
                    context_frames.view(&[batch_size * seq_size, channels, height, width][..]);
                let packed_context_poses =
                    context_params.view(&[batch_size * seq_size, n_params][..]);
                let packed_representation =
                    encoder(&packed_context_frames, &packed_context_poses, train);

                let representation = {
                    let size = packed_representation.size();
                    let repr_channels = size[1];
                    let repr_height = size[2];
                    let repr_width = size[3];
                    let stacked_repr = packed_representation
                        .view(&[batch_size, seq_size, repr_channels, repr_height, repr_width][..]);
                    let repr = stacked_repr.sum1(&[1], false, Kind::Float);
                    repr
                };

                // Broadcast encoding
                let broadcast_repr = {
                    match encoder_kind {
                        GqnEncoderKind::Tower => {
                            let repr_size = representation.size();
                            let repr_height = repr_size[2];
                            let repr_width = repr_size[3];

                            let target_size = target_frame.size();
                            let target_height = target_size[2];
                            let target_width = target_size[3];

                            assert!(
                                target_height == repr_height * 4 && target_width == repr_width * 4
                            );
                            representation
                        }
                        GqnEncoderKind::Pool => {
                            let target_size = target_frame.size();
                            let target_height = target_size[2];
                            let target_width = target_size[3];
                            let repr_height = target_height / 4;
                            let repr_width = target_width / 4;

                            representation.repeat(&[1, 1, repr_height, repr_width])
                        }
                    }
                };

                let GqnDecoderOutput {
                    means_target,
                    canvases,
                    // inf_states: Vec<rnn::GqnLSTMState>,
                    // gen_states: Vec<rnn::GqnLSTMState>,
                    means_inf,
                    stds_inf,
                    means_gen,
                    stds_gen,
                } = decoder.forward_t(&broadcast_repr, &query_params, &target_frame, train);

                let stds_target = pixel_std_annealing(&means_target.size(), step, device);
                let target_normal = Normal::new(&means_target, &stds_target);
                let target_sample = target_normal.sample();

                let target_frame_no_grad = target_frame.set_requires_grad(false);
                let target_mse = means_target.mse_loss(&target_frame_no_grad, Reduction::None);

                let elbo_loss = objective::elbo(
                    &means_target,
                    &stds_target,
                    &means_inf,
                    &stds_inf,
                    &means_gen,
                    &stds_gen,
                    &target_frame_no_grad,
                );

                GqnModelOutput {
                    elbo_loss,
                    target_mse,
                    means_target,
                    stds_target,
                    target_sample,
                    canvases,
                    means_inf,
                    stds_inf,
                    means_gen,
                    stds_gen,
                }
            },
        )
    }
}

fn pixel_std_annealing(shape: &[i64], step: usize, device: Device) -> Tensor {
    let begin = params::GENERATOR_SIGMA_BEGIN;
    let end = params::GENERATOR_SIGMA_END;
    let max_step = params::ANNEAL_SIGMA_MAX;
    let std = end + (begin - end) * (1.0 - step as f64 / max_step).max(0.0);

    Tensor::zeros(shape, (Kind::Float, device)).fill_(std)
}
