use tch::{nn, Tensor, Kind, Device};
use crate::utils;
use crate::rnn;

pub struct GqnDecoder
{
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

pub struct GqnDecoderOutput
{
    target: Tensor,
    canvases: Vec<Tensor>,

    inf_states: Vec<rnn::GqnLSTMState>,
    gen_states: Vec<rnn::GqnLSTMState>,

    means_inf: Vec<Tensor>,
    vars_inf: Vec<Tensor>,

    means_gen: Vec<Tensor>,
    vars_gen: Vec<Tensor>,

}

impl GqnDecoder
{
    pub fn new(
        vs: &nn::Path,

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

    ) -> GqnDecoder
    {
        let hidden_channels = 4 * cell_output_channels; // TODO remove this hard-code
        let canvas_conv_input_channels = target_channels + canvas_channels;
        let gen_input_channels = repr_channels + poses_channels + noise_channels;
        let inf_input_channels = repr_channels + poses_channels + hidden_channels;

        let mut generator_lstms = Vec::new();
        let mut inference_lstms = Vec::new();
        let mut canvas_convs = Vec::new();
        let mut canvas_dconvs = Vec::new();
        let mut inf_noise_convs = Vec::new();
        let mut gen_noise_convs = Vec::new();

        for step in 0..num_layers
        {
            // noise part
            let inf_noise_conv = nn::conv2d(
                vs / &format!("inf_noise_conv_{}", step),
                hidden_channels,
                2 * noise_channels,
                cell_kernel_size,
                nn::ConvConfig {
                    padding: (cell_kernel_size - 1) / 2,
                    stride: 1,
                    ..Default::default()
                },
            );

            let gen_noise_conv = nn::conv2d(
                vs / &format!("gen_noise_conv_{}", step),
                hidden_channels,
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
                &(vs / &format!("generator_lstm_{}", step)),
                biases,
                train,
                gen_input_channels,
                cell_output_channels,
                cell_kernel_size,
                1.0,
            );

            let dconv = nn::conv_transpose2d(
                vs / &format!("canvas_dconv_{}", step),
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
                &(vs / &format!("inference_lstm_{}", step)),
                biases,
                train,
                inf_input_channels,
                cell_output_channels,
                cell_kernel_size,
                1.0,
            );

            let conv = nn::conv2d(
                vs / &format!("canvas_conv_{}", step),
                canvas_conv_input_channels,
                hidden_channels,
                canvas_kernel_size,
                nn::ConvConfig {
                    stride: 4,
                    bias: false,
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
            vs / "target_conv",
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
            device: vs.device(),

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

    pub fn forward(
        &self,
        representation: &Tensor,
        query_poses: &Tensor,
        target_frames: &Tensor
    ) -> GqnDecoderOutput
    {
        let repr_size = representation.size();
        let batch_size = repr_size[0];
        let repr_height = repr_size[2];
        let repr_width = repr_size[3];

        let broadcasted_poses = broadcast_poses(query_poses, repr_height, repr_width);

        let inf_init_state = self.inference_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let gen_init_state = self.generator_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let init_canvas = Tensor::zeros(
            &[batch_size, self.canvas_channels, repr_height, repr_width],
            (Kind::Float, self.device),
        );

        let mut inf_states = Vec::new();
        let mut gen_states = Vec::new();
        let mut canvases = Vec::new();
        let mut means_inf = Vec::new();
        let mut vars_inf = Vec::new();
        let mut means_gen = Vec::new();
        let mut vars_gen = Vec::new();

        for step in 0..(self.num_layers)
        {
            let prev_inf_state_ref = match inf_states.last() {
                Some(ref prev) => prev,
                None => &inf_init_state,
            };
            let rnn::GqnLSTMState {
                h: ref prev_inf_h_ref,
                c: ref prev_inf_c_ref } = prev_inf_state_ref;

            let prev_gen_state_ref = match gen_states.last() {
                Some(ref prev) => prev,
                None => &gen_init_state,
            };
            let rnn::GqnLSTMState {
                h: ref prev_gen_h_ref,
                c: ref prev_gen_c_ref } = prev_gen_state_ref;

            let prev_canvas_ref = match canvases.last() {
                Some(ref prev) => prev,
                None => &init_canvas,
            };

            // Inference part
            let inf_lstm = &self.inference_lstms[step as usize];
            let canvas_conv = &self.canvas_convs[step as usize];

            let inf_state_addon = Tensor::cat(&[target_frames, prev_canvas_ref], 1);

            let inf_merge_h = prev_inf_h_ref + inf_state_addon.apply(canvas_conv);

            let inf_input = Tensor::cat(&[representation, &broadcasted_poses, prev_gen_h_ref], 1);
            let inf_state = inf_lstm.step(&inf_input, &inf_merge_h, prev_inf_c_ref);

            // noise
            // We have different random source for training/eval mode

            let (mean_inf, var_inf, noise_inf) = self.make_noise(
                prev_inf_h_ref,
                &self.inf_noise_convs[step as usize],
            );
            let (mean_gen, var_gen, noise_gen) = self.make_noise(
                prev_gen_h_ref,
                &self.gen_noise_convs[step as usize],
            );
            let input_noise = if self.train { noise_inf } else { noise_gen };

            // generator part
            let gen_lstm = &self.generator_lstms[step as usize];
            let canvas_dconv = &self.canvas_dconvs[step as usize];

            let gen_input = Tensor::cat(&[representation, &broadcasted_poses, &input_noise], 1);
            let gen_state = gen_lstm.step(&gen_input, prev_gen_h_ref, prev_gen_c_ref);
            let gen_output = &gen_state.h;
            let canvas = match canvases.last() {
                Some(prev) => prev + gen_output.apply(canvas_dconv),
                None => gen_output.apply(canvas_dconv),
            };

            canvases.push(canvas);
            gen_states.push(gen_state);
            inf_states.push(inf_state);

            means_inf.push(mean_inf);
            vars_inf.push(var_inf);

            means_gen.push(mean_gen);
            vars_gen.push(var_gen);
        }

        let target = canvases.last().unwrap().apply(&self.target_conv);

        GqnDecoderOutput {
            target,
            canvases,

            inf_states,
            gen_states,

            means_inf,
            vars_inf,

            means_gen,
            vars_gen,
        }
    }

    fn make_noise(&self, hidden: &Tensor, conv: &nn::Conv2D) -> (Tensor, Tensor, Tensor)
    {
        let hidden_size = hidden.size();
        let batch_size = hidden_size[0];
        let hidden_channels = hidden_size[1];
        let hidden_height = hidden_size[2];
        let hidden_width = hidden_size[3];

        // Eta function
        let conv_hidden = hidden.apply(conv);
        let mu = conv_hidden.narrow(1, 0, self.noise_channels);
        let raw_sigma = conv_hidden.narrow(1, self.noise_channels, self.noise_channels);
        let sigma = (raw_sigma + 0.5).softplus() + 1e-8;

        // Compute noise
        let random_source = Tensor::randn(
            &[batch_size, self.noise_channels, hidden_height, hidden_width],
            (Kind::Float, self.device)
        );
        let noise = &sigma * random_source;

        (mu, sigma, noise)
    }

}

fn broadcast_poses(poses: &Tensor, height: i64, width: i64) -> Tensor
{
    let batch_size = poses.size()[0];
    poses.reshape(&[batch_size, utils::POSE_CHANNELS, 1, 1])
        .repeat(&[1, 1, height, width])
}
