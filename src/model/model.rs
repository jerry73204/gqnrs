use super::{params, GqnDecoder, GqnDecoderOutput, GqnEncoder, PoolEncoder, TowerEncoder};

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

pub struct GqnModel<E: GqnEncoder> {
    encoder: E,
    decoder: GqnDecoder,
    device: Device,
}

impl<E: 'static> GqnModel<E>
where
    E: GqnEncoder,
{
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(
        path: P,
        frame_channels: i64,
        param_channels: i64,
    ) -> GqnModel<E> {
        let pathb = path.borrow();

        let encoder = E::new(pathb / "encoder", params::ENC_CHANNELS, param_channels);

        let decoder = GqnDecoder::new(
            pathb / "decoder",  // path
            params::SEQ_LENGTH, // num layers
            true,               // biases
            true,               // train
            params::ENC_CHANNELS,
            param_channels,
            params::Z_CHANNELS,
            params::LSTM_OUTPUT_CHANNELS,
            params::LSTM_CANVAS_CHANNELS,
            frame_channels,
            params::LSTM_KERNEL_SIZE,
            params::ETA_INTERNAL_KERNEL_SIZE,
            params::ETA_EXTERNAL_KERNEL_SIZE,
        );

        let device = pathb.device();

        GqnModel {
            encoder,
            decoder,
            device,
        }
    }

    pub fn forward_t(&self, input: GqnModelInput, train: bool) -> GqnModelOutput {
        let GqnModelInput {
            context_frames,
            target_frame,
            context_params,
            query_params,
            step,
        } = input;

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
        let packed_context_poses = context_params.view(&[batch_size * seq_size, n_params][..]);
        let packed_representation =
            self.encoder
                .forward_t(&packed_context_frames, &packed_context_poses, train);

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
            let encoder_type = TypeId::of::<E>();

            if encoder_type == TypeId::of::<TowerEncoder>() {
                let repr_size = representation.size();
                let repr_height = repr_size[2];
                let repr_width = repr_size[3];

                let target_size = target_frame.size();
                let target_height = target_size[2];
                let target_width = target_size[3];

                assert!(target_height == repr_height * 4 && target_width == repr_width * 4);
                representation
            } else if encoder_type == TypeId::of::<PoolEncoder>() {
                let target_size = target_frame.size();
                let target_height = target_size[2];
                let target_width = target_size[3];
                let repr_height = target_height / 4;
                let repr_width = target_width / 4;

                representation.repeat(&[1, 1, repr_height, repr_width])
            } else {
                panic!("bug");
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
        } = self
            .decoder
            .forward_t(&broadcast_repr, &query_params, &target_frame, train);

        let stds_target = pixel_std_annealing(&means_target.size(), step, self.device);
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
    }
}

fn pixel_std_annealing(shape: &[i64], step: usize, device: Device) -> Tensor {
    let begin = params::GENERATOR_SIGMA_BEGIN;
    let end = params::GENERATOR_SIGMA_END;
    let max_step = params::ANNEAL_SIGMA_MAX;
    let std = end + (begin - end) * (1.0 - step as f64 / max_step).max(0.0);

    Tensor::zeros(shape, (Kind::Float, device)).fill_(std)
}
