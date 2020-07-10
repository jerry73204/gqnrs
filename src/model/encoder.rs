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

    let conv1 = nn::conv2d(
        path / "conv1",
        3,
        256,
        2,
        nn::ConvConfig {
            padding: 0,
            stride: 2,
            ..Default::default()
        },
    );
    let conv2 = nn::conv2d(
        path / "conv2",
        256,
        128,
        1,
        nn::ConvConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        },
    );
    let conv3 = nn::conv2d(
        path / "conv3",
        256,
        128,
        3,
        nn::ConvConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        },
    );
    let conv4 = nn::conv2d(
        path / "conv4",
        128,
        256,
        2,
        nn::ConvConfig {
            padding: 0,
            stride: 2,
            ..Default::default()
        },
    );
    let conv5 = nn::conv2d(
        path / "conv5",
        256 + param_channels,
        128,
        1,
        nn::ConvConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        },
    );
    let conv6 = nn::conv2d(
        path / "conv6",
        256 + param_channels,
        128,
        3,
        nn::ConvConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        },
    );
    let conv7 = nn::conv2d(
        path / "conv7",
        128,
        256,
        3,
        nn::ConvConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        },
    );
    let conv8 = nn::conv2d(
        path / "conv8",
        256,
        repr_channels,
        1,
        nn::ConvConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        },
    );

    Box::new(move |frames, poses, _train| {
        let x1 = frames.apply(&conv1);
        let x2 = x1.apply(&conv3);
        let skip1 = x1.apply(&conv2);
        let x3 = x2 + skip1;

        let x4 = x3.apply(&conv4);

        let broadcast_poses = {
            let (b, _c, h, w) = x4.size4().unwrap();
            poses
                .reshape(&[b, param_channels, 1, 1])
                .repeat(&[1, 1, h, w])
        };

        let x5 = Tensor::cat(&[x4, broadcast_poses], 1);
        let skip2 = x5.apply(&conv5);
        let x6 = x5.apply(&conv6);
        let x7 = x6 + skip2;

        let x8 = x7.apply(&conv7);
        let x9 = x8.apply(&conv8);

        x9
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
        let tower_output = tower_encoder(frames, poses, train);
        let (b, c, _h, _w) = tower_output.size4().unwrap();

        // reduce mean of height, width dimension
        let encoder_output = tower_output
            .mean1(&[2, 3], false, Kind::Float)
            .view([b, c, 1, 1]);

        encoder_output
    })
}
