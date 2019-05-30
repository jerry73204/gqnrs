use std::any::TypeId;
use tch::{nn, nn::OptimizerConfig, Tensor, Kind, Device, Reduction};
use crate::encoder::{GqnEncoder, TowerEncoder, PoolEncoder};
use crate::decoder::{GqnDecoder, GqnDecoderOutput};
use crate::utils;
use crate::dist::{Rv, Normal};
use crate::objective::elbo;

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

impl<E: 'static> GqnModel<E> where
    E: GqnEncoder,
{
    pub fn new(path: &nn::Path, image_channels: i64) -> GqnModel<E> {
        let encoder = E::new(
            &(path / "encoder"),
            utils::ENC_CHANNELS,
            utils::POSE_CHANNELS,
        );

        let decoder = GqnDecoder::new(
            &(path / "decoder"),  // path
            utils::SEQ_LENGTH,  // num layers
            true,               // biases
            true,               // train
            utils::ENC_CHANNELS,
            utils::POSE_CHANNELS,
            utils::Z_CHANNELS,
            utils::LSTM_OUTPUT_CHANNELS,
            utils::LSTM_CANVAS_CHANNELS,
            image_channels,
            utils::LSTM_KERNEL_SIZE,
            utils::ETA_INTERNAL_KERNEL_SIZE,
            utils::ETA_EXTERNAL_KERNEL_SIZE,
        );

        let device = path.device();

        GqnModel {
            encoder,
            decoder,
            device,
        }
    }

    pub fn forward_t(
        &self,
        context_frames: &Tensor,
        context_poses: &Tensor,
        query_poses: &Tensor,
        target_frame: &Tensor,
        step: i64,
        train: bool,
    ) -> GqnModelOutput
    {
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
            let context_poses_size = context_poses.size();
            let batch_size2 = context_poses_size[0];
            let seq_size2 = context_poses_size[1];
            let n_params = context_poses_size[2];
            assert!(batch_size == batch_size2 && seq_size == seq_size2);
            n_params
        };

        let packed_context_frames = context_frames.view(&[batch_size * seq_size, channels, height, width]);
        let packed_context_poses = context_poses.view(&[batch_size * seq_size, n_params]);
        let packed_representation = self.encoder.forward_t(&packed_context_frames, &packed_context_poses, train);

        let representation = {
            let size = packed_representation.size();
            let repr_channels = size[1];
            let repr_height = size[2];
            let repr_width = size[3];
            let stacked_repr = packed_representation.view(&[batch_size, seq_size, repr_channels, repr_height, repr_width]);
            let repr = stacked_repr.sum2(&[1], false);
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
            }
            else if encoder_type == TypeId::of::<PoolEncoder>() {
                let target_size = target_frame.size();
                let target_height = target_size[2];
                let target_width = target_size[3];
                let repr_height = target_height / 4;
                let repr_width = target_width / 4;

                representation.repeat(&[1, 1, repr_height, repr_width])
            }
            else {
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
        } = self.decoder.forward_t(&broadcast_repr, query_poses, target_frame, train);

        let stds_target = pixel_std_annealing(&means_target.size(), step, self.device);
        let target_normal = Normal::new(&means_target, &stds_target);
        let target_sample = target_normal.sample();

        let target_frame_no_grad = target_frame.set_requires_grad(false);
        let target_mse = means_target.mse_loss(&target_frame_no_grad, Reduction::Mean);

        let elbo_loss = elbo(
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

fn pixel_std_annealing(shape: &[i64], step: i64, device: Device) -> Tensor {
    let sigma_i = utils::GENERATOR_SIGMA_ALPHA;
    let sigma_f = utils::GENERATOR_SIGMA_BETA;
    let anneal_max_step = utils::ANNEAL_SIGMA_TAU;
    let std = sigma_f.max(sigma_f + (sigma_i - sigma_f) * (1.0 - step as f64 / anneal_max_step));

    Tensor::zeros(shape, (Kind::Float, device))
        .fill_(std)
}
