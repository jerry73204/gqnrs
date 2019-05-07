use tch::{nn, Tensor};
use crate::encoder;
use crate::decoder;
use crate::utils;

pub struct GqnModel<E: encoder::GqnEncoder>
{
    encoder: E,
    decoder: decoder::GqnDecoder,
}

impl<E> GqnModel<E> where
    E: encoder::GqnEncoder {
    pub fn new(
        vs: &nn::Path,
    ) -> GqnModel<E> {
        let encoder = E::new(&(vs / "encoder"), utils::POSE_CHANNELS);
        let decoder = decoder::GqnDecoder::new(
            &(vs / "decoder"),
            utils::SEQ_LENGTH,
            true,
            true,
            utils::IMG_CHANNELS,
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
        target_frames: &Tensor,
        train: bool,
    ) -> decoder::GqnDecoderOutput {
        let representation = self.encoder.forward_t(context_frames, context_poses, train);
        println!("{:?}", representation.size());
        // TODO reshaping
        let decoder_output = self.decoder.forward_t(&representation, query_poses, target_frames, train);
        decoder_output
    }

    pub fn backward_fn(&self) {

    }
}
