use tch::{nn, Device, Tensor, Kind};

pub struct GqnLSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

pub struct GqnLSTM {
    biases: bool,
    train: bool,
    conv_ih: nn::Conv2D,
    conv_hh: nn::Conv2D,
    in_channels: i64,
    out_channels: i64,
    forget_bias: f64,
    device: Device,
}

impl GqnLSTM {
    pub fn new(
        vs: &nn::Path,
        biases: bool,
        train: bool,
        in_channels: i64,
        out_channels: i64,
        kernel_size:i64,
        forget_bias: f64,
    ) -> GqnLSTM
    {
        let hidden_channels = 4 * out_channels;
        let conv_config = nn::ConvConfig {
            stride: 1,
            padding: (kernel_size - 1) / 2,
            bias: biases,
            ..Default::default()
        };

        let conv_ih = nn::conv2d(
            vs / "conv_ih",
            in_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let conv_hh = nn::conv2d(
            vs / "conv_hh",
            out_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let device = vs.device();

        GqnLSTM {
            biases,
            train,
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

    pub fn step(&self, input: &Tensor, hx: &Tensor, cx: &Tensor) -> GqnLSTMState {
        let gates = input.apply(&self.conv_ih) + hx.apply(&self.conv_hh);
        let mut in_gate = gates.narrow(
            1,
            0 * self.out_channels,
            self.out_channels,
        );
        let mut forget_gate = gates.narrow(
            1,
            1 * self.out_channels,
            self.out_channels,
        );
        let mut cell_gate = gates.narrow(
            1,
            2 * self.out_channels,
            self.out_channels,
        );
        let mut out_gate = gates.narrow(
            1,
            3 * self.out_channels,
            self.out_channels,
        );

        in_gate = in_gate.sigmoid();
        forget_gate = forget_gate.sigmoid() + self.forget_bias;
        cell_gate = cell_gate.tanh();
        out_gate = out_gate.sigmoid();

        let cy = forget_gate * cx + in_gate * cell_gate;
        let hy = out_gate + cy.tanh();

        GqnLSTMState { h: hy, c: cy }
    }
}
