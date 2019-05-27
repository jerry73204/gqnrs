mod math;

use tch::Tensor;

pub trait Rv {
    fn sample(&self) -> Tensor;
    fn log_prob(&self, value: &Tensor) -> Tensor;
    fn log_cdf(&self, value: &Tensor) -> Tensor;
    fn cdf(&self, value: &Tensor) -> Tensor;
}

pub trait KLDiv<R> where
    R: Rv {
    fn kl_div(&self, other: &R) -> Tensor;
}

pub struct Normal<'a> {
    mean: &'a Tensor,
    std: &'a Tensor,
}

impl<'a> Normal<'a> {
    pub fn new(mean: &'a Tensor, std: &'a Tensor) -> Normal<'a> {
        Normal {
            mean,
            std,
        }
    }

    fn z(&self, x: &Tensor) -> Tensor {
        (x - self.mean) /  self.std
    }

    fn log_unnormalized_prob(&self, x: &Tensor) -> Tensor {
        -0.5 * self.z(x).pow(2)
    }

    fn log_normalization(&self) -> Tensor {
        0.5 * (2.0 * std::f64::consts::PI).ln() + self.std.log()
    }
}

impl<'a> Rv for Normal<'a> {
    fn sample(&self) -> Tensor {
        Tensor::normal2(&self.mean, &self.std)
    }

    // References
    // https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/normal.py
    // https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/internal/special_math.py
    fn log_prob(&self, value: &Tensor) -> Tensor {
        self.log_unnormalized_prob(value) - self.log_normalization()
    }

    fn log_cdf(&self, value: &Tensor) -> Tensor {
        let z = self.z(value);
        math::log_ndtr(&z)
    }

    fn cdf(&self, value: &Tensor) -> Tensor {
        let z = self.z(value);
        math::ndtr(&z)
    }
}

impl<'a> KLDiv<Normal<'a>> for Normal<'a> {
    fn kl_div(& self, other: & Normal) -> Tensor {
        let var_a = self.std.pow(2);
        let var_b = other.std.pow(2);
        let ratio = &var_a / &var_b;

        (self.mean - other.mean).pow(2) / (2. * var_b) + 0.5 * (&ratio - 1 - ratio.log())
    }
}
