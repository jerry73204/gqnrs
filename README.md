# gqnrs

DeepMind's [Generative Query Network (GQN)](https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf) implementation in Rust. It relies on [tch-rs](https://github.com/LaurentMazare/tch-rs), the libtorch binding in Rust, and [tfrecord-rs](https://github.com/jerry73204/tfrecord-rs/tree/master) for data serving. The repo is mainly done by jerry73204 for [NTU NEWSLAB](https://newslabcpsgroupwebsite.wordpress.com/)'s research project.

I take [tf-gqn](https://github.com/ogroth/tf-gqn) as reference implementation, and is not guaranteed to be 100% correct. Feel free to report issues if you find any glitches.

## Build

### Prerequisites

The program requires CUDA 9.0 and libtorch to work.

- Rust 2021 edition

  Install Rust package on your OS/distribution through package manager or something else. I would recommend [rustup](https://rustup.rs/).

- PyTorch 1.10.2

  Install PyTorch using `pip` if you're using Ubuntu 20.04.

  ```sh
  pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- CUDA 11.5

  Visit NVIDIA [CUDA Archive page](https://developer.nvidia.com/cuda-toolkit-archive). Click and install CUDA 11.5.2.



### Compile

Supposed that you are using Ubuntu 20.04, set the following environment variables.

```sh
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export LIBTORCH="$HOME/.local/lib/python3.8/site-packages/torch"
export LIBTORCH_CXX11_ABI=0
export RUST_LOG=info
```

To build the project,

```sh
cargo build --release
```

## Dataset

Obtain [gqn-dataset](https://github.com/deepmind/gqn-datasets). The overall dataset has about 2TB size. Make sure you have enough storage.

## Usage

### Example Usage

Suppose we want to train on `rooms_free_camera_with_object_rotations` dataset, located at `/path/to/gqn-dataset/rooms_free_camera_with_object_rotations`. We save the model file `model.zip` during training.

```sh
cargo run --release -- \
                        -n rooms_free_camera_with_object_rotations \
                        -i /path/to/gqn-dataset/rooms_free_camera_with_object_rotations \
                        --model-file model.zip
```

### Advanced Usage

The program provides several advanced features, including

- **Logging**: Save the losses and model outputs in `logs` directory with `--log-dir logs`.
- **Multi-GPU training**: Specify allocated devices by `--devices 'cuda(0),cuda(1)'`, and set the batch size per device by `--batch-size 4`.
- **Tweak logging**: Save model every 1000 steps by `--save-steps 1000`, and write logs every 100 steps by `--log-steps 100`. You can save output images by `--save-image`.


```sh
cargo run --release -- \
                        -n rooms_free_camera_with_object_rotations \
                        -i /path/to/gqn-dataset/rooms_free_camera_with_object_rotations \
                        --model-file model.zip \
                        --log-dir logs \
                        --batch-size 4 \
                        --devices 'cuda(0),cuda(1)' \
                        --save-steps 1000 \
                        --log-steps 100 \
                        --save-image
```


## Visualization

You can plot the loss curve using Facebooks' [Visdom](https://github.com/facebookresearch/visdom). The helper script `plot.py` visualizes the learning progress.

Check the instructions on official site to install Visdom, and start the Visdom server.

```sh
visdom
```

Let gqnrs saves the logs by `--log-dir logs` option. Run `./plot.py` to push to logs to visdom server.

```sh
./plot.py logs
```
