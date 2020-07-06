use super::rnn::{GqnLSTM, GqnLSTMState};
use crate::common::*;

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
    pub fn build<'p, P>(self, path: P, step: i64) -> GqnDecoderCell
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
        let inf_noise_conv = nn::conv2d(
            path / &format!("inf_noise_conv_{}", step),
            cell_output_channels,
            2 * noise_channels,
            cell_kernel_size,
            nn::ConvConfig {
                padding: (cell_kernel_size - 1) / 2,
                stride: 1,
                ..Default::default()
            },
        );

        let gen_noise_conv = nn::conv2d(
            path / &format!("gen_noise_conv_{}", step),
            cell_output_channels,
            2 * noise_channels,
            cell_kernel_size,
            nn::ConvConfig {
                padding: (cell_kernel_size - 1) / 2,
                stride: 1,
                ..Default::default()
            },
        );

        // generator part
        let gen_lstm = GqnLSTM::new(
            path / &format!("generator_lstm_{}", step),
            biases,
            gen_input_channels,
            cell_output_channels,
            cell_kernel_size,
            1.0,
        );

        let canvas_dconv = nn::conv_transpose2d(
            path / &format!("canvas_dconv_{}", step),
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
            path / &format!("inference_lstm_{}", step),
            biases,
            inf_input_channels,
            cell_output_channels,
            cell_kernel_size,
            1.0,
        );

        let canvas_conv = nn::conv2d(
            path / &format!("canvas_conv_{}", step),
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
            inf_noise_conv,
            gen_noise_conv,
            gen_lstm,
            canvas_dconv,
            inf_lstm,
            canvas_conv,
        }
    }
}

#[derive(Debug)]
pub struct GqnDecoderCell {
    gen_lstm: GqnLSTM,
    inf_lstm: GqnLSTM,
    canvas_conv: nn::Conv2D,
    canvas_dconv: nn::ConvTranspose2D,
    inf_noise_conv: nn::Conv2D,
    gen_noise_conv: nn::Conv2D,
}

#[derive(Debug)]
pub struct GqnDecoder {
    num_layers: i64,
    biases: bool,

    repr_channels: i64,
    param_channels: i64,
    noise_channels: i64,
    cell_output_channels: i64,
    canvas_channels: i64,
    target_channels: i64,

    cell_kernel_size: i64,
    canvas_kernel_size: i64,
    target_kernel_size: i64,

    decoder_cells: Vec<GqnDecoderCell>,

    target_conv: nn::Conv2D,

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
                .build(path, step)
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
            num_layers,
            biases,
            device: path.device(),

            repr_channels,
            param_channels,
            noise_channels,
            cell_output_channels,
            canvas_channels,
            target_channels,

            cell_kernel_size,
            canvas_kernel_size,
            target_kernel_size,

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
        let (batch_size, repr_height, repr_width) = {
            let repr_size = representation.size();
            let batch_size = repr_size[0];
            let repr_height = repr_size[2];
            let repr_width = repr_size[3];
            (batch_size, repr_height, repr_width)
        };

        let (target_height, target_width) = {
            let target_size = target_frame.size();
            let batch_size2 = target_size[0];
            let target_height = target_size[2];
            let target_width = target_size[3];
            assert!(batch_size == batch_size2);
            (target_height, target_width)
        };

        let broadcasted_poses = self.broadcast_poses(query_poses, repr_height, repr_width);

        let inf_init_state =
            self.decoder_cells[0]
                .inf_lstm
                .zero_state(batch_size, repr_height, repr_width);
        let gen_init_state =
            self.decoder_cells[0]
                .gen_lstm
                .zero_state(batch_size, repr_height, repr_width);
        let init_canvas = Tensor::zeros(
            &[
                batch_size,
                self.canvas_channels,
                target_height,
                target_width,
            ],
            (Kind::Float, self.device),
        );

        struct GqnDecoderCellState {
            inf_state: GqnLSTMState,
            gen_state: GqnLSTMState,
            canvas: Tensor,
            mean_inf: Tensor,
            std_inf: Tensor,
            mean_gen: Tensor,
            std_gen: Tensor,
        }

        let states: Vec<GqnDecoderCellState> =
            self.decoder_cells.iter().fold(vec![], |mut states, cell| {
                let GqnDecoderCell {
                    gen_lstm,
                    inf_lstm,
                    canvas_conv,
                    canvas_dconv,
                    inf_noise_conv,
                    gen_noise_conv,
                } = cell;

                // Extract tensors from previous step
                let (prev_inf_state, prev_gen_state, prev_canvas) = states
                    .last()
                    .map(|state| {
                        (
                            state.inf_state.shallow_clone(),
                            state.inf_state.shallow_clone(),
                            state.canvas.shallow_clone(),
                        )
                    })
                    .unwrap_or_else(|| {
                        (
                            inf_init_state.shallow_clone(),
                            gen_init_state.shallow_clone(),
                            init_canvas.shallow_clone(),
                        )
                    });

                // Inference part
                let inf_h_extra = Tensor::cat(&[target_frame, &prev_canvas], 1).apply(canvas_conv);
                let inf_h_combined = &prev_inf_state.h + &inf_h_extra;

                assert!(representation.size()[1] == self.repr_channels);
                assert!(broadcasted_poses.size()[1] == self.param_channels);
                assert!(prev_gen_state.h.size()[1] == self.cell_output_channels);
                let inf_input =
                    Tensor::cat(&[representation, &broadcasted_poses, &prev_gen_state.h], 1);
                let inf_state = inf_lstm.step(&inf_input, &inf_h_combined, &prev_inf_state.c);

                // Create noise tensor
                // We have different random source for training/eval mode
                let (mean_inf, std_inf, noise_inf) =
                    self.make_noise(&prev_inf_state.h, inf_noise_conv);
                let (mean_gen, std_gen, noise_gen) =
                    self.make_noise(&prev_gen_state.h, gen_noise_conv);
                let input_noise = if train { noise_inf } else { noise_gen };

                // generator part
                let gen_input = Tensor::cat(&[representation, &broadcasted_poses, &input_noise], 1);
                let gen_state = gen_lstm.step(&gen_input, &prev_gen_state.h, &prev_gen_state.c);
                let gen_output = &gen_state.h;

                let canvas_extra = gen_output
                    .apply(canvas_dconv)
                    .narrow(2, 0, target_height)
                    .narrow(3, 0, target_width); // Crop out extra width/height due to deconvolution
                let canvas = prev_canvas + canvas_extra;

                states.push(GqnDecoderCellState {
                    canvas,
                    gen_state,
                    inf_state,
                    mean_inf,
                    std_inf,
                    mean_gen,
                    std_gen,
                });
                states
            });

        let means_target = states.last().unwrap().canvas.apply(&self.target_conv);
        let canvases = Tensor::stack(
            &states.iter().map(|state| &state.canvas).collect::<Vec<_>>(),
            1,
        );
        let means_inf = Tensor::stack(
            &states
                .iter()
                .map(|state| &state.mean_inf)
                .collect::<Vec<_>>(),
            1,
        );
        let stds_inf = Tensor::stack(
            &states
                .iter()
                .map(|state| &state.std_inf)
                .collect::<Vec<_>>(),
            1,
        );
        let means_gen = Tensor::stack(
            &states
                .iter()
                .map(|state| &state.mean_gen)
                .collect::<Vec<_>>(),
            1,
        );
        let stds_gen = Tensor::stack(
            &states
                .iter()
                .map(|state| &state.std_gen)
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
        let hidden_size = hidden.size();
        let batch_size = hidden_size[0];
        let hidden_height = hidden_size[2];
        let hidden_width = hidden_size[3];

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
