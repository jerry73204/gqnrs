// input parameters
pub const POSE_CHANNELS: i64 = 7;

// hyper-parameters: scene representation
pub const ENC_CHANNELS: i64 = 256;

// hyper-parameters: generator LSTM
pub const LSTM_OUTPUT_CHANNELS: i64 = 256;
pub const LSTM_CANVAS_CHANNELS: i64 = 256;
pub const LSTM_KERNEL_SIZE: i64 = 5;
pub const Z_CHANNELS: i64 = 64;  // latent space size per image generation step
pub const SEQ_LENGTH: i64 = 8;  // number image generation steps; orig.: 12

// hyper-parameters: eta functions
pub const ETA_INTERNAL_KERNEL_SIZE: i64 = 5;  // internal projection of states to means and variances
pub const ETA_EXTERNAL_KERNEL_SIZE: i64 = 1;  // kernel size for final projection of canvas to mean image

// hyper-parameters: ADAM optimization
pub const ANNEAL_SIGMA_TAU: f64 = 200000.0;  // annealing interval for global noise
pub const GENERATOR_SIGMA_ALPHA: f64 = 2.;  // start value for global generation variance
pub const GENERATOR_SIGMA_BETA: f64 = 0.7;  // final value for global generation variance
pub const ANNEAL_LR_TAU: f64 = 1600000.0;  // annealing interval for learning rate
pub const ADAM_LR_ALPHA: f64 = 5.0 * 10e-6;  // start learning rate of ADAM optimizer; orig.: 5 * 10e-4
pub const ADAM_LR_BETA: f64 = 1.0 * 10e-6;  // final learning rate of ADAM optimizer; orig.: 5 * 10e-5
