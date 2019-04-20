use tch::{nn, Tensor, Kind};
use crate::utils;

pub fn generator_decoder(vs: &nn::Path, representation: &Tensor, query_poses: &Tensor) -> Tensor
{
    let repr_size = representation.size();
    // let batch_size = repr_size[0];
    let repr_height = repr_size[2];
    let repr_width = repr_size[3];

    let broadcasted_poses = broadcast_poses(vs, query_poses, repr_height, repr_width);
    broadcasted_poses
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
