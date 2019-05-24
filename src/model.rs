use std::any::TypeId;
use tch::{nn, Tensor};
use crate::encoder::{GqnEncoder, TowerEncoder, PoolEncoder};
use crate::decoder::{GqnDecoder, GqnDecoderOutput};
use crate::utils;

pub struct GqnModel<E: GqnEncoder>
{
    encoder: E,
    decoder: GqnDecoder,
}

impl<E: 'static> GqnModel<E> where
    E: GqnEncoder
{
    pub fn new(vs: &nn::Path) -> GqnModel<E> {
        let encoder = E::new(
            &(vs / "encoder"),
            utils::ENC_CHANNELS,
            utils::POSE_CHANNELS,
        );

        let decoder = GqnDecoder::new(
            &(vs / "decoder"),  // path
            utils::SEQ_LENGTH,  // num layers
            true,               // biases
            true,               // train
            utils::ENC_CHANNELS,
            utils::POSE_CHANNELS,
            utils::Z_CHANNELS,
            utils::LSTM_OUTPUT_CHANNELS,
            utils::LSTM_CANVAS_CHANNELS,
            utils::IMG_CHANNELS,
            utils::LSTM_KERNEL_SIZE,
            utils::ETA_INTERNAL_KERNEL_SIZE,
            utils::ETA_EXTERNAL_KERNEL_SIZE,
        );

        GqnModel {
            encoder,
            decoder,
        }
    }

    pub fn forward_t(
        &self,
        context_frames: &Tensor,
        context_poses: &Tensor,
        query_poses: &Tensor,
        target_frame: &Tensor,
        train: bool,
    ) -> GqnDecoderOutput
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

        let decoder_output = self.decoder.forward_t(&broadcast_repr, query_poses, target_frame, train);
        decoder_output
    }

    pub fn backward_fn(&self) {
        // TODO
    }
}
