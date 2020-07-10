use super::rnn::{GqnDecoderCell, GqnDecoderCellInit, GqnDecoderCellState, GqnNoise};
use crate::common::*;

#[derive(Debug)]
pub struct GqnDecoder {
    // model params
    repr_channels: i64,
    param_channels: i64,
    noise_channels: i64,
    cell_output_channels: i64,
    canvas_channels: i64,
    target_channels: i64,
    cell_kernel_size: i64,
    canvas_kernel_size: i64,
    target_kernel_size: i64,
    // modules & weights
    decoder_cells: Vec<GqnDecoderCell>,
    target_conv: nn::Conv2D,
}

impl GqnDecoder {
    pub fn new<'a, P>(
        path: P,
        // model params
        num_layers: i64,
        biases: bool,
        // channels
        repr_channels: i64,
        param_channels: i64,
        noise_channels: i64,
        cell_output_channels: i64,
        canvas_channels: i64,
        target_channels: i64,
        // kernel sizes
        cell_kernel_size: i64,
        canvas_kernel_size: i64,
        target_kernel_size: i64,
    ) -> GqnDecoder
    where
        P: Borrow<nn::Path<'a>>,
    {
        let path = path.borrow();

        let decoder_cells = (0..num_layers)
            .map(|step| {
                GqnDecoderCellInit {
                    target_channels,
                    repr_channels,
                    param_channels,
                    cell_output_channels,
                    noise_channels,
                    cell_kernel_size,
                    biases,
                    canvas_channels,
                    canvas_kernel_size,
                }
                .build(path / format!("decoder_cell_{}", step))
            })
            .collect::<Vec<_>>();

        let target_conv = nn::conv2d(
            path / "target_conv",
            canvas_channels,
            target_channels,
            target_kernel_size,
            nn::ConvConfig {
                stride: 1,
                padding: (target_kernel_size - 1) / 2,
                ..Default::default()
            },
        );

        GqnDecoder {
            // params
            repr_channels,
            param_channels,
            noise_channels,
            cell_output_channels,
            canvas_channels,
            target_channels,
            cell_kernel_size,
            canvas_kernel_size,
            target_kernel_size,
            // modules
            decoder_cells,
            target_conv,
        }
    }

    pub fn forward_t(
        &self,
        representation: &Tensor,
        query_poses: &Tensor,
        target_frame: &Tensor,
        train: bool,
    ) -> GqnDecoderOutput {
        // get batch size and sanity check
        let (batch_size, repr_channels, _h, _w) = representation.size4().unwrap();
        debug_assert_eq!(repr_channels, self.repr_channels);

        {
            let (b, _c, _h, _w) = target_frame.size4().unwrap();
            debug_assert_eq!(b, batch_size);
        }

        {
            let (b, n) = query_poses.size2().unwrap();
            debug_assert_eq!(b, batch_size);
            debug_assert_eq!(n, self.param_channels);
        }

        // pass thru decoder cells
        let init_decoder_cell_state =
            self.decoder_cells[0].zero_state(target_frame, representation);

        let (decoder_states, inf_noises, gen_noises) =
            self.decoder_cells
                .iter()
                .fold((vec![], vec![], vec![]), |mut args, cell| {
                    let (states, inf_noises, gen_noises) = &mut args;
                    let prev_state: &GqnDecoderCellState =
                        states.last().unwrap_or(&init_decoder_cell_state);

                    let (state, inf_noise, gen_noise) = cell.step(
                        target_frame,
                        representation,
                        &query_poses,
                        prev_state,
                        train,
                    );

                    states.push(state);
                    inf_noises.push(inf_noise);
                    gen_noises.push(gen_noise);

                    args
                });

        // compute target mean from last canvas
        let target_mean = decoder_states
            .last()
            .unwrap()
            .canvas
            .apply(&self.target_conv);

        GqnDecoderOutput {
            target_mean,
            decoder_states,
            inf_noises,
            gen_noises,
        }
    }
}

pub struct GqnDecoderOutput {
    pub target_mean: Tensor,
    pub decoder_states: Vec<GqnDecoderCellState>,
    pub inf_noises: Vec<GqnNoise>,
    pub gen_noises: Vec<GqnNoise>,
}
