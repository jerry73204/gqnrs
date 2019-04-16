extern crate tch;

use super::utils;

use tch::{nn, Tensor};

pub fn tower_encoder(p: &nn::Path, frames: &Tensor, poses: &Tensor) -> Tensor
{
    let conv_config = |padding, stride| {
        nn::ConvConfig {padding: padding, stride: stride, ..Default::default()}
    };

    let mut net = frames.apply(&nn::conv2d(p / "conv1", 3, 256, 2, conv_config(0, 2)));
    let mut skip = net.apply(&nn::conv2d(p / "conv2", 256, 128, 1, conv_config(0, 1)));
    net = net.apply(&nn::conv2d(p / "conv3", 256, 128, 3, conv_config(1, 1)));
    net = net + skip;

    net = net.apply(&nn::conv2d(p / "conv4", 128, 256, 2, conv_config(0, 2)));

    let net_size = net.size();
    let batch_size = net_size[0];
    let broadcast_poses = poses
        .reshape(&[batch_size, utils::POSE_CHANNELS, 1, 1])
        .repeat(&[1, 1, net_size[2], net_size[3]]);
    net = Tensor::cat(&[net, broadcast_poses], 1);

    skip = net.apply(&nn::conv2d(p / "conv5", 263, 128, 1, conv_config(0, 1)));
    net = net.apply(&nn::conv2d(p / "conv6", 263, 128, 3, conv_config(1, 1)));
    net = net + skip;

    net = net.apply(&nn::conv2d(p / "conv7", 128, 256, 3, conv_config(1, 1)));
    net = net.apply(&nn::conv2d(p / "conv8", 256, 256, 1, conv_config(0, 1)));

    net
}
