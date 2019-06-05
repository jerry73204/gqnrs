use tch::{nn, Tensor, Kind, Device};
use crate::params;
use crate::rnn;

pub struct GqnDecoder {
    num_layers: i64,
    biases: bool,
    train: bool,

    repr_channels: i64,
    poses_channels: i64,
    noise_channels: i64,
    cell_output_channels: i64,
    canvas_channels: i64,
    target_channels: i64,

    cell_kernel_size: i64,
    canvas_kernel_size: i64,
    target_kernel_size: i64,

    generator_lstms: Vec<rnn::GqnLSTM>,
    inference_lstms: Vec<rnn::GqnLSTM>,
    canvas_convs: Vec<nn::Conv2D>,
    canvas_dconvs: Vec<nn::ConvTranspose2D>,
    inf_noise_convs: Vec<nn::Conv2D>,
    gen_noise_convs: Vec<nn::Conv2D>,
    target_conv: nn::Conv2D,

    device: Device,
}

pub struct GqnDecoderOutput {
    pub means_target: Tensor,
    pub canvases: Tensor,
    // pub inf_states: Vec<rnn::GqnLSTMState>,
    // pub gen_states: Vec<rnn::GqnLSTMState>,
    pub means_inf: Tensor,
    pub stds_inf: Tensor,
    pub means_gen: Tensor,
    pub stds_gen: Tensor,
}

impl GqnDecoder {
    pub fn new(
        path: &nn::Path,
        // model params
        num_layers: i64,
        biases: bool,
        train: bool,
        // channels
        repr_channels: i64,
        poses_channels: i64,
        noise_channels: i64,
        cell_output_channels: i64,
        canvas_channels: i64,
        target_channels: i64,
        // kernel sizes
        cell_kernel_size: i64,
        canvas_kernel_size: i64,
        target_kernel_size: i64,

    ) -> GqnDecoder {
        let canvas_conv_input_channels = target_channels + canvas_channels;
        let gen_input_channels = repr_channels + poses_channels + noise_channels;
        let inf_input_channels = repr_channels + poses_channels + cell_output_channels;

        let mut generator_lstms = Vec::new();
        let mut inference_lstms = Vec::new();
        let mut canvas_convs = Vec::new();
        let mut canvas_dconvs = Vec::new();
        let mut inf_noise_convs = Vec::new();
        let mut gen_noise_convs = Vec::new();

        for step in 0..num_layers {
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
            let gen_lstm = rnn::GqnLSTM::new(
                &(path / &format!("generator_lstm_{}", step)),
                biases,
                train,
                gen_input_channels,
                cell_output_channels,
                cell_kernel_size,
                1.0,
            );

            let dconv = nn::conv_transpose2d(
                path / &format!("canvas_dconv_{}", step),
                cell_output_channels,
                canvas_channels,
                canvas_kernel_size,
                nn::ConvTransposeConfig {
                    stride: 4,
                    ..Default::default()
                }
            );

            // inference part
            let inf_lstm = rnn::GqnLSTM::new(
                &(path / &format!("inference_lstm_{}", step)),
                biases,
                train,
                inf_input_channels,
                cell_output_channels,
                cell_kernel_size,
                1.0,
            );

            let conv = nn::conv2d(
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

            generator_lstms.push(gen_lstm);
            inference_lstms.push(inf_lstm);
            canvas_dconvs.push(dconv);
            canvas_convs.push(conv);
            inf_noise_convs.push(inf_noise_conv);
            gen_noise_convs.push(gen_noise_conv);
        }

        let target_conv = nn::conv2d(
            path / "target_conv",
            canvas_channels,
            target_channels,
            target_kernel_size,
            nn::ConvConfig {
                stride: 1,
                padding: (target_kernel_size - 1) / 2,
                ..Default::default()
            }
        );

        GqnDecoder {
            num_layers,
            biases,
            train,
            device: path.device(),

            repr_channels,
            poses_channels,
            noise_channels,
            cell_output_channels,
            canvas_channels,
            target_channels,

            cell_kernel_size,
            canvas_kernel_size,
            target_kernel_size,

            generator_lstms,
            inference_lstms,

            canvas_convs,
            canvas_dconvs,
            inf_noise_convs,
            gen_noise_convs,
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

        let broadcasted_poses = broadcast_poses(query_poses, repr_height, repr_width);

        let inf_init_state = self.inference_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let gen_init_state = self.generator_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let init_canvas = Tensor::zeros(
            &[batch_size, self.canvas_channels, target_height, target_width],
            (Kind::Float, self.device),
        );

        let mut inf_states = Vec::new();
        let mut gen_states = Vec::new();
        let mut canvases = Vec::new();
        let mut means_inf = Vec::new();
        let mut stds_inf = Vec::new();
        let mut means_gen = Vec::new();
        let mut stds_gen = Vec::new();

        // Chain the LSTM cells
        for step in 0..(self.num_layers) {
            // Extract tensors from previous step
            let prev_inf_state = match inf_states.last() {
                Some(ref prev) => prev,
                None => &inf_init_state,
            };
            let rnn::GqnLSTMState {
                h: ref prev_inf_h,
                c: ref prev_inf_c
            } = prev_inf_state;

            let prev_gen_state = match gen_states.last() {
                Some(ref prev) => prev,
                None => &gen_init_state,
            };
            let rnn::GqnLSTMState {
                h: ref prev_gen_h,
                c: ref prev_gen_c } = prev_gen_state;

            let prev_canvas = match canvases.last() {
                Some(ref prev) => prev,
                None => &init_canvas,
            };

            // Inference part
            let inf_lstm = &self.inference_lstms[step as usize];
            let canvas_conv = &self.canvas_convs[step as usize];

            let inf_h_extra = Tensor::cat(&[target_frame, prev_canvas], 1)
                .apply(canvas_conv);
            let inf_h_combined = prev_inf_h + inf_h_extra;

            assert!(representation.size()[1] == self.repr_channels);
            assert!(broadcasted_poses.size()[1] == self.poses_channels);
            assert!(prev_gen_h.size()[1] == self.cell_output_channels);
            let inf_input = Tensor::cat(&[representation, &broadcasted_poses, prev_gen_h], 1);
            let inf_state = inf_lstm.step(&inf_input, &inf_h_combined, prev_inf_c);

            // Create noise tensor
            // We have different random source for training/eval mode
            let (mean_inf, std_inf, noise_inf) = self.make_noise(
                prev_inf_h,
                &self.inf_noise_convs[step as usize],
            );
            let (mean_gen, std_gen, noise_gen) = self.make_noise(
                prev_gen_h,
                &self.gen_noise_convs[step as usize],
            );
            let input_noise = if self.train { noise_inf } else { noise_gen };

            // generator part
            let gen_lstm = &self.generator_lstms[step as usize];
            let canvas_dconv = &self.canvas_dconvs[step as usize];

            let gen_input = Tensor::cat(&[representation, &broadcasted_poses, &input_noise], 1);
            let gen_state = gen_lstm.step(&gen_input, prev_gen_h, prev_gen_c);
            let gen_output = &gen_state.h;

            let canvas_extra = gen_output.apply(canvas_dconv)
                .narrow(2, 0, target_height)
                .narrow(3, 0, target_width); // Crop out extra width/height due to deconvolution
            let canvas = prev_canvas + canvas_extra;

            canvases.push(canvas);
            gen_states.push(gen_state);
            inf_states.push(inf_state);

            means_inf.push(mean_inf);
            stds_inf.push(std_inf);

            means_gen.push(mean_gen);
            stds_gen.push(std_gen);
        }

        let means_target = canvases.last().unwrap().apply(&self.target_conv);
        let canvases_tensor = Tensor::stack(&canvases, 1);
        let means_inf_tensor = Tensor::stack(&means_inf, 1);
        let stds_inf_tensor = Tensor::stack(&stds_inf, 1);
        let means_gen_tensor = Tensor::stack(&means_gen, 1);
        let stds_gen_tensor = Tensor::stack(&stds_gen, 1);

        GqnDecoderOutput {
            means_target,
            canvases: canvases_tensor,
            // inf_states,
            // gen_states,
            means_inf: means_inf_tensor,
            stds_inf: stds_inf_tensor,
            means_gen: means_gen_tensor,
            stds_gen: stds_gen_tensor,
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
            (Kind::Float, self.device)
        );
        let noise = &means + &stds * random_source;

        (means, stds, noise)
    }

}

fn broadcast_poses(poses: &Tensor, height: i64, width: i64) -> Tensor {
    let batch_size = poses.size()[0];
    poses.reshape(&[batch_size, params::POSE_CHANNELS, 1, 1])
        .repeat(&[1, 1, height, width])
}
