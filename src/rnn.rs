
use std::borrow::Borrow;
use tch::{nn, Device, Tensor, Kind};

pub struct GqnLSTMState((Tensor, Tensor));

pub struct GqnLSTM
{
    conv_ih: nn::Conv2D,
    conv_hh: nn::Conv2D,
    in_channels: i64,
    out_channels: i64,
    forget_bias: f64,
    config: nn::RNNConfig,
    device: Device,
}

impl GqnLSTM
{
    pub fn new(
        p: &nn::Path,
        in_channels: i64,
        out_channels: i64,
        kernel_size:i64,
        forget_bias: f64,
        rnn_config: nn::RNNConfig,
    ) -> GqnLSTM
    {
        let hidden_channels = 4 * out_channels;
        let conv_config = nn::ConvConfig {
            stride: 1,
            padding: (kernel_size - 1) / 2,
            bias: rnn_config.has_biases,
            ..Default::default()
        };

        let conv_ih = nn::conv2d(
            p / "conv_ih",
            in_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let conv_hh = nn::conv2d(
            p / "conv_hh",
            out_channels,
            hidden_channels,
            kernel_size,
            conv_config,
        );
        let device = p.device();

        GqnLSTM {
            conv_ih,
            conv_hh,
            in_channels,
            out_channels,
            forget_bias,
            device,
            config: rnn_config,
        }
    }

    pub fn zero_state(&self, batch_dim: i64, height_dim: i64, width_dim: i64) -> GqnLSTMState
    {
        let hidden_size = [batch_dim, self.out_channels, height_dim, width_dim];

        let hx = Tensor::zeros(&hidden_size, (Kind::Float, self.device));
        let cx = Tensor::zeros(&hidden_size, (Kind::Float, self.device));

        GqnLSTMState((hx, cx))
    }

    pub fn step(&self, input: &Tensor, in_state: GqnLSTMState) -> GqnLSTMState
    {
        let GqnLSTMState((hx, cx)) = in_state;
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

        GqnLSTMState((hy, cy))
    }

    pub fn seq(&self, inputs: &Tensor) -> (Tensor, GqnLSTMState)
    {
        let inputs_size_vec = inputs.size();
        let [batch_dim, height_dim, width_dim] = if self.config.batch_first {
            [
                inputs_size_vec[0],
                inputs_size_vec[3],
                inputs_size_vec[4],
            ]

        } else {
            [
                inputs_size_vec[1],
                inputs_size_vec[3],
                inputs_size_vec[4],
            ]
        };

        let mut state = self.zero_state(batch_dim, height_dim, width_dim);

        for ind in 0..(self.config.num_layers)
        {
            let step_input = if self.config.batch_first {
                inputs.select(1, ind)
            } else {
                inputs.select(0, ind)
            };
            state = self.step(&step_input, state);
        }

        let GqnLSTMState((hy, cy)) = state;
        (hy.copy(), GqnLSTMState((hy, cy)))
    }
}
