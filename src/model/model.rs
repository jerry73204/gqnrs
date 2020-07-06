use super::{
    decoder::{GqnDecoder, GqnDecoderOutput},
    encoder, params,
    rnn::{GqnDecoderCellState, GqnNoise},
};

use crate::{
    common::*,
    dist::{Normal, Rv},
    objective,
};

// input type

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

// output type

#[derive(Debug)]
pub struct GqnModelOutput {
    // losses
    pub elbo_loss: Tensor,
    pub target_mse: Tensor,
    // target sample
    pub target_sample: Tensor,
    pub target_mean: Tensor,
    pub target_std: Tensor,
    // states
    pub decoder_states: Vec<GqnDecoderCellState>,
    pub inf_noises: Vec<GqnNoise>,
    pub gen_noises: Vec<GqnNoise>,
}

impl GqnModelOutput {
    pub fn to_device(&self, device: Device) -> Self {
        let Self {
            elbo_loss,
            target_mse,
            target_sample,
            target_mean,
            target_std,
            decoder_states,
            inf_noises,
            gen_noises,
        } = self;

        Self {
            elbo_loss: elbo_loss.to_device(device),
            target_mse: target_mse.to_device(device),
            target_sample: target_sample.to_device(device),
            target_mean: target_mean.to_device(device),
            target_std: target_std.to_device(device),
            decoder_states: decoder_states
                .iter()
                .map(|state| state.to_device(device))
                .collect(),
            inf_noises: inf_noises
                .iter()
                .map(|noise| noise.to_device(device))
                .collect(),
            gen_noises: gen_noises
                .iter()
                .map(|noise| noise.to_device(device))
                .collect(),
        }
    }

    pub fn shallow_clone(&self) -> Self {
        let Self {
            elbo_loss,
            target_mse,
            target_sample,
            target_mean,
            target_std,
            decoder_states,
            inf_noises,
            gen_noises,
        } = self;

        Self {
            elbo_loss: elbo_loss.shallow_clone(),
            target_mse: target_mse.shallow_clone(),
            target_sample: target_sample.shallow_clone(),
            target_mean: target_mean.shallow_clone(),
            target_std: target_std.shallow_clone(),
            decoder_states: decoder_states
                .iter()
                .map(|state| state.shallow_clone())
                .collect(),
            inf_noises: inf_noises
                .iter()
                .map(|noise| noise.shallow_clone())
                .collect(),
            gen_noises: gen_noises
                .iter()
                .map(|noise| noise.shallow_clone())
                .collect(),
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
                encoder::tower_encoder(path / "tower_encoder", enc_channels, param_channels)
            }
            GqnEncoderKind::Pool => {
                encoder::pool_encoder(path / "pool_encoder", enc_channels, param_channels)
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
                let target_frame = target_frame.set_requires_grad(false);
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
                    target_mean,
                    decoder_states,
                    inf_noises,
                    gen_noises,
                } = decoder.forward_t(&broadcast_repr, &query_params, &target_frame, train);

                // sample target
                let target_std = pixel_std_annealing(&target_mean.size(), step, device);
                let target_sample = Normal::new(&target_mean, &target_std).sample();

                // compute target loss
                let target_mse = target_mean.mse_loss(&target_frame, Reduction::Mean);

                // compute elbo loss
                let elbo_loss = objective::elbo(
                    &target_frame,
                    &target_mean,
                    &target_std,
                    &inf_noises,
                    &gen_noises,
                );

                GqnModelOutput {
                    // losses
                    elbo_loss,
                    target_mse,
                    // target sample
                    target_mean,
                    target_std,
                    target_sample,
                    // states
                    decoder_states,
                    inf_noises,
                    gen_noises,
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
