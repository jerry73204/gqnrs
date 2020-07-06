use crate::common::*;

// gqn lstm cell

#[derive(Debug)]
pub struct GqnLSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl GqnLSTMState {
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            h: self.h.to_device(device),
            c: self.c.to_device(device),
        }
    }

    pub fn shallow_clone(&self) -> Self {
        Self {
            h: self.h.shallow_clone(),
            c: self.c.shallow_clone(),
        }
    }
}

#[derive(Debug)]
pub struct GqnLSTM {
    biases: bool,
    conv_ih: Conv2D,
    conv_hh: Conv2D,
    in_channels: i64,
    out_channels: i64,
    forget_bias: f64,
    device: Device,
}

impl GqnLSTM {
    pub fn new<'p, P>(
        path: P,
        biases: bool,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        forget_bias: f64,
    ) -> GqnLSTM
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let hidden_channels = 4 * out_channels;
        let conv_config = ConvConfig {
            stride: 1,
            padding: (kernel_size - 1) / 2,
            bias: biases,
            ..Default::default()
        };

        let conv_ih = nn::conv2d(
            path / "conv_ih",
            in_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let conv_hh = nn::conv2d(
            path / "conv_hh",
            out_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let device = path.device();

        GqnLSTM {
            biases,
            conv_ih,
            conv_hh,
            in_channels,
            out_channels,
            forget_bias,
            device,
        }
    }

    pub fn zero_state(&self, batch: i64, height: i64, width: i64) -> GqnLSTMState {
        let hidden_size = [batch, self.out_channels, height, width];

        let h = Tensor::zeros(&hidden_size, (Kind::Float, self.device));
        let c = Tensor::zeros(&hidden_size, (Kind::Float, self.device));

        GqnLSTMState { h, c }
    }

    pub fn step(&self, input: &Tensor, prev_state: &GqnLSTMState) -> GqnLSTMState {
        let GqnLSTMState { h: hx, c: cx } = prev_state;

        let gates = input.apply(&self.conv_ih) + hx.apply(&self.conv_hh);
        let mut in_gate = gates.narrow(1, 0 * self.out_channels, self.out_channels);
        let mut forget_gate = gates.narrow(1, 1 * self.out_channels, self.out_channels);
        let mut cell_gate = gates.narrow(1, 2 * self.out_channels, self.out_channels);
        let mut out_gate = gates.narrow(1, 3 * self.out_channels, self.out_channels);

        in_gate = in_gate.sigmoid();
        forget_gate = forget_gate.sigmoid() + self.forget_bias;
        cell_gate = cell_gate.tanh();
        out_gate = out_gate.sigmoid();

        let cy = forget_gate * cx + in_gate * cell_gate;
        let hy = out_gate + cy.tanh();

        GqnLSTMState { h: hy, c: cy }
    }
}

// decoder cell

#[derive(Debug, Clone)]
pub struct GqnDecoderCellInit {
    pub target_channels: i64,
    pub repr_channels: i64,
    pub param_channels: i64,
    pub cell_output_channels: i64,
    pub noise_channels: i64,
    pub cell_kernel_size: i64,
    pub biases: bool,
    pub canvas_channels: i64,
    pub canvas_kernel_size: i64,
}

impl GqnDecoderCellInit {
    pub fn build<'p, P>(self, path: P) -> GqnDecoderCell
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            target_channels,
            repr_channels,
            param_channels,
            noise_channels,
            cell_output_channels,
            cell_kernel_size,
            biases,
            canvas_channels,
            canvas_kernel_size,
        } = self;

        let canvas_conv_input_channels = target_channels + canvas_channels;
        let gen_input_channels = repr_channels + param_channels + noise_channels;
        let inf_input_channels = repr_channels + param_channels + cell_output_channels;

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
            target_channels,
            repr_channels,
            param_channels,
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
    target_channels: i64,
    repr_channels: i64,
    param_channels: i64,
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
        query_poses: &Tensor,
        prev_state: &GqnDecoderCellState,
        train: bool,
    ) -> (GqnDecoderCellState, GqnNoise, GqnNoise) {
        // get sizes and sanity check
        let (batch_size, target_height, target_width) = match target_frame.size().as_slice() {
            &[batch_size, channels, height, width] => {
                debug_assert_eq!(channels, self.target_channels);
                (batch_size, height, width)
            }
            _ => unreachable!(),
        };
        let (repr_height, repr_width) = match representation.size().as_slice() {
            &[batch_size_, channels, height, width] => {
                debug_assert_eq!(batch_size, batch_size_);
                debug_assert_eq!(channels, self.repr_channels);
                (height, width)
            }
            _ => unreachable!(),
        };
        match query_poses.size().as_slice() {
            &[batch_size_, channels] => {
                debug_assert_eq!(batch_size, batch_size_);
                debug_assert_eq!(channels, self.param_channels);
            }
            _ => unreachable!(),
        }

        // prepare cell inputs
        let GqnDecoderCellState {
            inf_state: prev_inf_state,
            gen_state: prev_gen_state,
            canvas: prev_canvas,
        } = prev_state;

        let broadcasted_poses =
            query_poses
                .reshape(&[batch_size, -1, 1, 1])
                .repeat(&[1, 1, repr_height, repr_width]);

        // run inference cell
        let inf_state = {
            let inf_h = &prev_inf_state.h
                + &Tensor::cat(&[target_frame, &prev_canvas], 1).apply(&self.canvas_conv);
            let inf_input =
                Tensor::cat(&[representation, &broadcasted_poses, &prev_gen_state.h], 1);
            self.inf_lstm.step(
                &inf_input,
                &GqnLSTMState {
                    h: inf_h,
                    c: prev_inf_state.c.shallow_clone(),
                },
            )
        };

        // generate noises
        let inf_noise = self.inf_noise_factory.forward(&prev_inf_state.h);
        let gen_noise = self.gen_noise_factory.forward(&prev_gen_state.h);

        // run generator cell
        let gen_state = {
            // We have different random source for training/eval mode
            let input_noise = if train {
                &inf_noise.noise
            } else {
                &gen_noise.noise
            };

            let gen_input = Tensor::cat(&[representation, &broadcasted_poses, input_noise], 1);
            self.gen_lstm.step(&gen_input, &prev_gen_state)
        };

        // create new canvas
        let canvas = {
            let addon = gen_state
                .h
                .apply(&self.canvas_dconv)
                .narrow(2, 0, target_height)
                .narrow(3, 0, target_width); // Crop out extra width/height due to deconvolution
            prev_canvas + addon
        };

        // output
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
            &[batch_size, _channels, height, width] => (batch_size, height, width),
            _ => unreachable!(),
        };
        let (target_height, target_width) = match target_frame.size().as_slice() {
            &[batch_size_, _channels, height, width] => {
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

#[derive(Debug)]
pub struct GqnDecoderCellState {
    pub inf_state: GqnLSTMState,
    pub gen_state: GqnLSTMState,
    pub canvas: Tensor,
}

impl GqnDecoderCellState {
    pub fn to_device(&self, device: Device) -> Self {
        let Self {
            inf_state,
            gen_state,
            canvas,
        } = self;

        Self {
            inf_state: inf_state.to_device(device),
            gen_state: gen_state.to_device(device),
            canvas: canvas.to_device(device),
        }
    }

    pub fn shallow_clone(&self) -> Self {
        let Self {
            inf_state,
            gen_state,
            canvas,
        } = self;

        Self {
            inf_state: inf_state.shallow_clone(),
            gen_state: gen_state.shallow_clone(),
            canvas: canvas.shallow_clone(),
        }
    }
}

// noise
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

impl GqnNoise {
    pub fn to_device(&self, device: Device) -> Self {
        let Self { means, stds, noise } = self;

        Self {
            means: means.to_device(device),
            stds: stds.to_device(device),
            noise: noise.to_device(device),
        }
    }

    pub fn shallow_clone(&self) -> Self {
        let Self { means, stds, noise } = self;

        Self {
            means: means.shallow_clone(),
            stds: stds.shallow_clone(),
            noise: noise.shallow_clone(),
        }
    }
}
