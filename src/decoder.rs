use tch::{nn, Tensor, Kind};
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
    target_conv: nn::Conv2D,
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

        for step in 0..num_layers
        {
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
                vs / &format!("canvas+_conv_{}", step),
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
            target_conv,
        }
    }

    pub fn forward(&self, vs: &nn::Path, representation: &Tensor, query_poses: &Tensor, target_frames: &Tensor) -> GqnDecoderOutput
    {
        let repr_size = representation.size();
        let batch_size = repr_size[0];
        let repr_height = repr_size[2];
        let repr_width = repr_size[3];

        let broadcasted_poses = broadcast_poses(vs, query_poses, repr_height, repr_width);

        let inf_init_state = self.inference_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let gen_init_state = self.generator_lstms[0].zero_state(batch_size, repr_height, repr_width);
        let init_canvas = Tensor::zeros(
            &[batch_size, self.canvas_channels, repr_height, repr_width],
            (Kind::Float, vs.device()),
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
            let (mean_gen, var_gen, noise_gen) = make_noise(&vs, prev_gen_h_ref);
            let (mean_inf, var_inf, noise_inf) = make_noise(&vs, prev_inf_h_ref);
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
}

fn broadcast_poses(vs: &nn::Path, poses: &Tensor, height: i64, width: i64) -> Tensor
{
    let batch_size = poses.size()[0];
    poses.reshape(&[batch_size, utils::POSE_CHANNELS, 1, 1])
        .repeat(&[1, 1, height, width])
}

fn make_noise(vs: &nn::Path, hidden: &Tensor) -> (Tensor, Tensor, Tensor)
{
    let device = hidden.device();
    let z_channels = utils::Z_CHANNELS;
    let hidden_size = hidden.size();
    let batch_size = hidden_size[0];
    let hidden_channels = hidden_size[1];
    let hidden_height = hidden_size[2];
    let hidden_width = hidden_size[3];

    // Eta function
    let conv_config = nn::ConvConfig {
        padding: (utils::LSTM_KERNEL_SIZE - 1) / 2,
        stride: 1,
        ..Default::default()
    };
    let conv = hidden.apply(&nn::conv2d(
        vs / "noise_conv2d",
        hidden_channels,
        2 * z_channels,
        utils::LSTM_KERNEL_SIZE,
        conv_config
    ));


    let mu = conv.narrow(1, 0, z_channels);
    let raw_sigma = conv.narrow(1, z_channels, z_channels);
    let sigma = (raw_sigma + 0.5).softplus() + 1e-8;

    // Compute noise
    let random_source = Tensor::randn(
        &[batch_size, z_channels, hidden_height, hidden_width],
        (Kind::Float, device)
    );
    let noise = &sigma * random_source;

    (mu, sigma, noise)
}
