use super::rnn::{GqnLSTM, GqnLSTMState};
use crate::common::*;

#[derive(Debug)]
pub struct GqnNoiseFactory {
    in_c: i64,
    out_c: i64,
    conv: nn::Conv2D,
}

impl GqnNoiseFactory {
    pub fn new<'p, P>(path: P, in_c: i64, out_c: i64, k: i64) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let conv = nn::conv2d(
            path / "conv",
            in_c,
            out_c * 2,
            k,
            nn::ConvConfig {
                padding: (k - 1) / 2,
                stride: 1,
                ..Default::default()
            },
        );

        Self { in_c, out_c, conv }
    }

    pub fn forward(&self, hidden: &Tensor) -> GqnNoise {
        // Eta function
        let conv_hidden = hidden.apply(&self.conv);
        let means = conv_hidden.narrow(1, 0, self.out_c);
        let stds = {
            let tmp = conv_hidden.narrow(1, self.out_c, self.out_c);
            (tmp + 0.5).softplus() + 1e-8
        };
        let scales = Tensor::randn(&means.size(), (Kind::Float, means.device()));

        // Compute noise
        let noise = &means + &stds * &scales;

        GqnNoise { means, stds, noise }
    }
}

#[derive(Debug)]
pub struct GqnNoise {
    pub means: Tensor,
    pub stds: Tensor,
    pub noise: Tensor,
}

#[derive(Debug, Clone)]
pub struct GqnDecoderCellInit {
    pub cell_output_channels: i64,
    pub noise_channels: i64,
    pub cell_kernel_size: i64,
    pub biases: bool,
    pub gen_input_channels: i64,
    pub canvas_channels: i64,
    pub canvas_kernel_size: i64,
    pub canvas_conv_input_channels: i64,
    pub inf_input_channels: i64,
}

impl GqnDecoderCellInit {
    pub fn build<'p, P>(self, path: P) -> GqnDecoderCell
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            cell_output_channels,
            noise_channels,
            cell_kernel_size,
            biases,
            gen_input_channels,
            canvas_channels,
            canvas_kernel_size,
            canvas_conv_input_channels,
            inf_input_channels,
        } = self;

        // noise part
        let inf_noise_factory = GqnNoiseFactory::new(
            path / "inf_noise",
            cell_output_channels,
            noise_channels,
            cell_kernel_size,
        );
        let gen_noise_factory = GqnNoiseFactory::new(
            path / "gen_noise",
            cell_output_channels,
            noise_channels,
            cell_kernel_size,
        );

        // generator part
        let gen_lstm = GqnLSTM::new(
            path / "generator_lstm",
            biases,
            gen_input_channels,
            cell_output_channels,
            cell_kernel_size,
            1.0,
        );

        let canvas_dconv = nn::conv_transpose2d(
            path / "canvas_dconv",
            cell_output_channels,
            canvas_channels,
            canvas_kernel_size,
            nn::ConvTransposeConfig {
                stride: 4,
                ..Default::default()
            },
        );

        // inference part
        let inf_lstm = GqnLSTM::new(
            path / "inference_lstm",
            biases,
            inf_input_channels,
            cell_output_channels,
            cell_kernel_size,
            1.0,
        );

        let canvas_conv = nn::conv2d(
            path / "canvas_conv",
            canvas_conv_input_channels,
            cell_output_channels,
            canvas_kernel_size,
            nn::ConvConfig {
                stride: 4,
                bias: false,
                padding: 1,
                ..Default::default()
            },
        );

        GqnDecoderCell {
            // params
            canvas_channels,
            // modules
            inf_noise_factory,
            gen_noise_factory,
            gen_lstm,
            canvas_dconv,
            inf_lstm,
            canvas_conv,
        }
    }
}

#[derive(Debug)]
pub struct GqnDecoderCell {
    // params
    canvas_channels: i64,
    // modules
    gen_lstm: GqnLSTM,
    inf_lstm: GqnLSTM,
    canvas_conv: nn::Conv2D,
    canvas_dconv: nn::ConvTranspose2D,
    inf_noise_factory: GqnNoiseFactory,
    gen_noise_factory: GqnNoiseFactory,
}

impl GqnDecoderCell {
    pub fn step(
        &self,
        target_frame: &Tensor,
        representation: &Tensor,
        broadcasted_poses: &Tensor,
        prev_state: &GqnDecoderCellState,
        train: bool,
    ) -> (GqnDecoderCellState, GqnNoise, GqnNoise) {
        let (target_height, target_width) = match target_frame.size().as_slice() {
            &[_batch_size_, _channels, height, width] => (height, width),
            _ => unreachable!(),
        };

        let GqnDecoderCellState {
            inf_state: prev_inf_state,
            gen_state: prev_gen_state,
            canvas: prev_canvas,
        } = prev_state;

        let inf_h_extra = Tensor::cat(&[target_frame, &prev_canvas], 1).apply(&self.canvas_conv);
        let inf_h_combined = &prev_inf_state.h + &inf_h_extra;
        // debug_assert!(prev_gen_state.h.size()[1] == self.cell_output_channels);

        let inf_input = Tensor::cat(&[representation, &broadcasted_poses, &prev_gen_state.h], 1);
        let inf_state = self.inf_lstm.step(
            &inf_input,
            &GqnLSTMState {
                h: inf_h_combined,
                c: prev_inf_state.c.shallow_clone(),
            },
        );

        // Create noise tensor
        // We have different random source for training/eval mode
        let inf_noise = self.inf_noise_factory.forward(&prev_inf_state.h);
        let gen_noise = self.gen_noise_factory.forward(&prev_gen_state.h);
        let input_noise = if train {
            &inf_noise.noise
        } else {
            &gen_noise.noise
        };

        // generator part
        let gen_input = Tensor::cat(&[representation, &broadcasted_poses, input_noise], 1);
        let gen_state = self.gen_lstm.step(&gen_input, &prev_gen_state);
        let gen_output = &gen_state.h;

        let canvas_extra = gen_output
            .apply(&self.canvas_dconv)
            .narrow(2, 0, target_height)
            .narrow(3, 0, target_width); // Crop out extra width/height due to deconvolution
        let canvas = prev_canvas + canvas_extra;

        (
            GqnDecoderCellState {
                canvas,
                gen_state,
                inf_state,
            },
            inf_noise,
            gen_noise,
        )
    }

    pub fn zero_state(
        &self,
        target_frame: &Tensor,
        representation: &Tensor,
    ) -> GqnDecoderCellState {
        let (batch_size, repr_height, repr_width) = match representation.size().as_slice() {
            &[batch_size, channels, height, width] => (batch_size, height, width),
            _ => unreachable!(),
        };
        let (target_height, target_width) = match target_frame.size().as_slice() {
            &[batch_size_, channels, height, width] => {
                debug_assert_eq!(batch_size, batch_size_);
                (height, width)
            }
            _ => unreachable!(),
        };

        let inf_state = self
            .inf_lstm
            .zero_state(batch_size, repr_height, repr_width);
        let gen_state = self
            .gen_lstm
            .zero_state(batch_size, repr_height, repr_width);
        let canvas = Tensor::zeros(
            &[
                batch_size,
                self.canvas_channels,
                target_height,
                target_width,
            ],
            (Kind::Float, representation.device()),
        );

        GqnDecoderCellState {
            inf_state,
            gen_state,
            canvas,
        }
    }
}

pub struct GqnDecoderCellState {
    pub inf_state: GqnLSTMState,
    pub gen_state: GqnLSTMState,
    pub canvas: Tensor,
}

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
    // misc
    device: Device,
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

        let canvas_conv_input_channels = target_channels + canvas_channels;
        let gen_input_channels = repr_channels + param_channels + noise_channels;
        let inf_input_channels = repr_channels + param_channels + cell_output_channels;

        let decoder_cells = (0..num_layers)
            .map(|step| {
                GqnDecoderCellInit {
                    cell_output_channels,
                    noise_channels,
                    cell_kernel_size,
                    biases,
                    gen_input_channels,
                    canvas_channels,
                    canvas_kernel_size,
                    canvas_conv_input_channels,
                    inf_input_channels,
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
            // misc
            device: path.device(),
        }
    }

    pub fn forward_t(
        &self,
        representation: &Tensor,
        query_poses: &Tensor,
        target_frame: &Tensor,
        train: bool,
    ) -> GqnDecoderOutput {
        let (batch_size, repr_height, repr_width) = match representation.size().as_slice() {
            &[batch_size, channels, height, width] => {
                debug_assert_eq!(channels, self.repr_channels);
                (batch_size, height, width)
            }
            _ => unreachable!(),
        };
        let (target_height, target_width) = match target_frame.size().as_slice() {
            &[batch_size_, channels, height, width] => {
                debug_assert_eq!(batch_size, batch_size_);
                (height, width)
            }
            _ => unreachable!(),
        };
        match query_poses.size().as_slice() {
            &[batch_size_, param_channels] => {
                debug_assert_eq!(batch_size_, batch_size);
                debug_assert_eq!(param_channels, self.param_channels);
            }
            _ => unreachable!(),
        }

        let broadcasted_poses = self.broadcast_poses(query_poses, repr_height, repr_width);
        let init_decoder_cell_state =
            self.decoder_cells[0].zero_state(target_frame, representation);

        let (states, inf_noises, gen_noises) =
            self.decoder_cells
                .iter()
                .fold((vec![], vec![], vec![]), |mut args, cell| {
                    let (states, inf_noises, gen_noises) = &mut args;
                    let prev_state: &GqnDecoderCellState =
                        states.last().unwrap_or(&init_decoder_cell_state);

                    let (state, inf_noise, gen_noise) = cell.step(
                        target_frame,
                        representation,
                        &broadcasted_poses,
                        prev_state,
                        train,
                    );

                    states.push(state);
                    inf_noises.push(inf_noise);
                    gen_noises.push(gen_noise);

                    args
                });

        let means_target = states.last().unwrap().canvas.apply(&self.target_conv);
        let canvases = Tensor::stack(
            &states.iter().map(|state| &state.canvas).collect::<Vec<_>>(),
            1,
        );
        let means_inf = Tensor::stack(
            &inf_noises
                .iter()
                .map(|noise| &noise.means)
                .collect::<Vec<_>>(),
            1,
        );
        let stds_inf = Tensor::stack(
            &inf_noises
                .iter()
                .map(|noise| &noise.stds)
                .collect::<Vec<_>>(),
            1,
        );
        let means_gen = Tensor::stack(
            &gen_noises
                .iter()
                .map(|noise| &noise.means)
                .collect::<Vec<_>>(),
            1,
        );
        let stds_gen = Tensor::stack(
            &gen_noises
                .iter()
                .map(|noise| &noise.stds)
                .collect::<Vec<_>>(),
            1,
        );
        let inf_states = states
            .iter()
            .map(|state| state.inf_state.shallow_clone())
            .collect::<Vec<_>>();
        let gen_states = states
            .iter()
            .map(|state| state.gen_state.shallow_clone())
            .collect::<Vec<_>>();

        GqnDecoderOutput {
            means_target,
            canvases,
            inf_states,
            gen_states,
            means_inf,
            stds_inf,
            means_gen,
            stds_gen,
        }
    }

    fn make_noise(&self, hidden: &Tensor, conv: &nn::Conv2D) -> (Tensor, Tensor, Tensor) {
        let (batch_size, hidden_height, hidden_width) = match hidden.size().as_slice() {
            &[batch_size, _channels, height, width] => (batch_size, height, width),
            _ => unreachable!(),
        };

        // Eta function
        let conv_hidden = hidden.apply(conv);
        let means = conv_hidden.narrow(1, 0, self.noise_channels);
        let stds_input = conv_hidden.narrow(1, self.noise_channels, self.noise_channels);
        let stds = (stds_input + 0.5).softplus() + 1e-8;

        // Compute noise
        let random_source = Tensor::randn(
            &[batch_size, self.noise_channels, hidden_height, hidden_width],
            (Kind::Float, self.device),
        );
        let noise = &means + &stds * random_source;

        (means, stds, noise)
    }

    fn broadcast_poses(&self, poses: &Tensor, height: i64, width: i64) -> Tensor {
        let batch_size = poses.size()[0];
        poses
            .reshape(&[batch_size, self.param_channels, 1, 1])
            .repeat(&[1, 1, height, width])
    }
}

pub struct GqnDecoderOutput {
    pub means_target: Tensor,
    pub canvases: Tensor,
    pub inf_states: Vec<GqnLSTMState>,
    pub gen_states: Vec<GqnLSTMState>,
    pub means_inf: Tensor,
    pub stds_inf: Tensor,
    pub means_gen: Tensor,
    pub stds_gen: Tensor,
}
