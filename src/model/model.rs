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

#[derive(Debug, TensorLike)]
pub struct GqnModelInput {
    pub context_frames: Tensor,
    pub target_frame: Tensor,
    pub context_params: Tensor,
    pub query_params: Tensor,
    pub step: usize,
}

// output type

#[derive(Debug, TensorLike)]
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

    pub fn build<'p, P>(
        self,
        path: P,
    ) -> Box<dyn FnMut(&GqnModelInput, bool) -> GqnModelOutput + Send>
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

        let mut step_tensor = path.zeros("step", &[]);

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
                    ..
                } = input;
                let step = input.step;

                // save recent step
                tch::no_grad(|| {
                    step_tensor.copy_(&Tensor::from(step as i64));
                });

                // get sizes
                let (batch_size, seq_size, channels, height, width) =
                    match context_frames.size().as_slice() {
                        &[b, s, c, h, w] => (b, s, c, h, w),
                        _ => unreachable!(),
                    };
                let n_params = {
                    let (b, s, n) = context_params.size3().unwrap();
                    debug_assert_eq!(b, batch_size);
                    debug_assert_eq!(s, seq_size);
                    n
                };
                let (target_height, target_width) = {
                    let (b, _c, h, w) = target_frame.size4().unwrap();
                    debug_assert_eq!(b, batch_size);
                    (h, w)
                };

                // run encoder
                let (representation, repr_height, repr_width) = {
                    let packed_context_frames =
                        context_frames.view([batch_size * seq_size, channels, height, width]);
                    let packed_context_poses =
                        context_params.view([batch_size * seq_size, n_params]);
                    let packed_representation =
                        encoder(&packed_context_frames, &packed_context_poses, train);

                    let (_b, c, h, w) = packed_representation.size4().unwrap();
                    let repr = packed_representation
                        .view([batch_size, seq_size, c, h, w])
                        .sum_dim_intlist(&[1], false, Kind::Float);
                    (repr, h, w)
                };
                debug_assert_eq!(target_height, repr_height * 4);
                debug_assert_eq!(target_width, repr_width * 4);

                // Broadcast encoding
                let broadcast_repr = match encoder_kind {
                    GqnEncoderKind::Tower => representation,
                    GqnEncoderKind::Pool => {
                        representation.repeat(&[1, 1, target_height / 4, target_width / 4])
                    }
                };

                {
                    let (b, _c, h, w) = broadcast_repr.size4().unwrap();
                    debug_assert_eq!(b, batch_size);
                    debug_assert_eq!(h, repr_height);
                    debug_assert_eq!(w, repr_width);
                }

                // run decoder
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

    nn::init(nn::Init::Const(std), shape, device)
}
