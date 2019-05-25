use tch::{nn, Tensor};

pub trait GqnEncoder {
    fn new(path: &nn::Path, repr_channels: i64, pose_channels: i64) -> Self;
    fn forward_t(&self, frames: &Tensor, poses: &Tensor, train: bool) -> Tensor;
}

pub struct TowerEncoder {
    pose_channels: i64,
    repr_channels: i64,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    conv4: nn::Conv2D,
    conv5: nn::Conv2D,
    conv6: nn::Conv2D,
    conv7: nn::Conv2D,
    conv8: nn::Conv2D,
}

impl GqnEncoder for TowerEncoder {
    fn new(path: &nn::Path, repr_channels: i64, pose_channels: i64) -> TowerEncoder {
        let conv_config = |padding, stride| {
            nn::ConvConfig {padding: padding, stride: stride, ..Default::default()}
        };

        let conv1 = nn::conv2d(path / "conv1", 3, 256, 2, conv_config(0, 2));
        let conv2 = nn::conv2d(path / "conv2", 256, 128, 1, conv_config(0, 1));
        let conv3 = nn::conv2d(path / "conv3", 256, 128, 3, conv_config(1, 1));
        let conv4 = nn::conv2d(path / "conv4", 128, 256, 2, conv_config(0, 2));
        let conv5 = nn::conv2d(path / "conv5", 263, 128, 1, conv_config(0, 1));
        let conv6 = nn::conv2d(path / "conv6", 263, 128, 3, conv_config(1, 1));
        let conv7 = nn::conv2d(path / "conv7", 128, 256, 3, conv_config(1, 1));
        let conv8 = nn::conv2d(path / "conv8", 256, repr_channels, 1, conv_config(0, 1));

        TowerEncoder {
            pose_channels,
            repr_channels,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
        }
    }

    fn forward_t(&self, frames: &Tensor, poses: &Tensor, train: bool) -> Tensor {
        let mut net = frames.apply(&self.conv1);
        let mut skip = net.apply(&self.conv2);
        net = net.apply(&self.conv3);
        net = net + skip;

        net = net.apply(&self.conv4);

        let net_size = net.size();
        let batch_size = net_size[0];
        let broadcast_poses = poses
            .reshape(&[batch_size, self.pose_channels, 1, 1])
            .repeat(&[1, 1, net_size[2], net_size[3]]);
        net = Tensor::cat(&[net, broadcast_poses], 1);

        skip = net.apply(&self.conv5);
        net = net.apply(&self.conv6);
        net = net + skip;

        net = net.apply(&self.conv7);
        net = net.apply(&self.conv8);

        net
    }
}

pub struct PoolEncoder {
    tower_encoder: TowerEncoder,
}

impl GqnEncoder for PoolEncoder {
    fn new(path: &nn::Path, repr_channels: i64, pose_channels: i64) -> PoolEncoder {
        let tower_encoder = TowerEncoder::new(path, repr_channels, pose_channels);

        PoolEncoder {
            tower_encoder,
        }
    }

    fn forward_t(&self, frames: &Tensor, poses: &Tensor, train: bool) -> Tensor {
        let mut net = self.tower_encoder.forward_t(frames, poses, train);
        let net_size = net.size();
        let batch_size = net_size[0];
        let n_channels = net_size[1];
        let height = net_size[2];
        let width = net_size[3];

        // reduce mean of height, width dimension
        net = net.view(&[batch_size, n_channels, height * width])
            .mean2(&[2], false)
            .view(&[batch_size, n_channels, 1, 1]);

        net
    }
}
