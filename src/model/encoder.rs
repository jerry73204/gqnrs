use crate::common::*;

pub fn tower_encoder<'p, P>(
    path: P,
    repr_channels: i64,
    param_channels: i64,
) -> Box<dyn Fn(&Tensor, &Tensor, bool) -> Tensor + Send>
where
    P: Borrow<nn::Path<'p>>,
{
    let path = path.borrow();

    let conv_config = |padding, stride| nn::ConvConfig {
        padding: padding,
        stride: stride,
        ..Default::default()
    };

    let conv1 = nn::conv2d(path / "conv1", 3, 256, 2, conv_config(0, 2));
    let conv2 = nn::conv2d(path / "conv2", 256, 128, 1, conv_config(0, 1));
    let conv3 = nn::conv2d(path / "conv3", 256, 128, 3, conv_config(1, 1));
    let conv4 = nn::conv2d(path / "conv4", 128, 256, 2, conv_config(0, 2));
    let conv5 = nn::conv2d(
        path / "conv5",
        256 + param_channels,
        128,
        1,
        conv_config(0, 1),
    );
    let conv6 = nn::conv2d(
        path / "conv6",
        256 + param_channels,
        128,
        3,
        conv_config(1, 1),
    );
    let conv7 = nn::conv2d(path / "conv7", 128, 256, 3, conv_config(1, 1));
    let conv8 = nn::conv2d(path / "conv8", 256, repr_channels, 1, conv_config(0, 1));

    Box::new(move |frames, poses, _train| {
        let mut net = frames.apply(&conv1);
        let mut skip = net.apply(&conv2);
        net = net.apply(&conv3);
        net = net + skip;

        net = net.apply(&conv4);

        let net_size = net.size();
        let batch_size = net_size[0];
        let broadcast_poses = poses.reshape(&[batch_size, param_channels, 1, 1]).repeat(&[
            1,
            1,
            net_size[2],
            net_size[3],
        ]);
        net = Tensor::cat(&[net, broadcast_poses], 1);

        skip = net.apply(&conv5);
        net = net.apply(&conv6);
        net = net + skip;

        net = net.apply(&conv7);
        net = net.apply(&conv8);

        net
    })
}

pub fn pool_encoder<'p, P>(
    path: P,
    repr_channels: i64,
    param_channels: i64,
) -> Box<dyn Fn(&Tensor, &Tensor, bool) -> Tensor + Send>
where
    P: Borrow<nn::Path<'p>>,
{
    let path = path.borrow();
    let tower_encoder = tower_encoder(path / "tower_encoder", repr_channels, param_channels);

    Box::new(move |frames, poses, train| {
        let mut net = tower_encoder(frames, poses, train);
        let net_size = net.size();
        let batch_size = net_size[0];
        let n_channels = net_size[1];
        let height = net_size[2];
        let width = net_size[3];

        // reduce mean of height, width dimension
        net = net
            .view(&[batch_size, n_channels, height * width][..])
            .mean1(&[2], false, Kind::Float)
            .view(&[batch_size, n_channels, 1, 1][..]);

        net
    })
}
