use crate::common::*;

mod math;

pub trait Rv {
    fn sample(&self) -> Tensor;
    fn log_prob(&self, value: &Tensor) -> Tensor;
    fn log_cdf(&self, value: &Tensor) -> Tensor;
    fn cdf(&self, value: &Tensor) -> Tensor;
}

pub trait KLDiv<R>
where
    R: Rv,
{
    fn kl_div(&self, other: &R) -> Tensor;
}

pub struct Normal {
    mean: Tensor,
    std: Tensor,
}

impl Normal {
    pub fn new(mean: &Tensor, std: &Tensor) -> Normal {
        let mean = mean.shallow_clone();
        let std = std.shallow_clone();
        assert_eq!(mean.size(), std.size());
        Normal { mean, std }
    }

    fn z(&self, x: &Tensor) -> Tensor {
        (x - &self.mean) / &self.std
    }

    fn log_unnormalized_prob(&self, x: &Tensor) -> Tensor {
        -0.5 * self.z(x).pow(2)
    }

    fn log_normalization(&self) -> Tensor {
        0.5 * (2. * std::f64::consts::PI).ln() + self.std.log()
    }
}

impl Rv for Normal {
    fn sample(&self) -> Tensor {
        tch::no_grad(|| {
            let out = Tensor::zeros(
                self.mean.size().as_slice(),
                (Kind::Float, self.mean.device()),
            );
            Tensor::normal_out2(&out, &self.mean, &self.std);
            out
        })
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

impl KLDiv<Normal> for Normal {
    fn kl_div(&self, other: &Normal) -> Tensor {
        let var_a = self.std.pow(2);
        let var_b = other.std.pow(2);
        let ratio = &var_a / &var_b;

        (&self.mean - &other.mean).pow(2) / (2. * var_b) + 0.5 * (&ratio - 1 - ratio.log())
    }
}

#[cfg(test)]
mod tests {
    extern crate rv;

    use super::{KLDiv, Normal, Rv};
    use rv::traits::{Cdf, ContinuousDistr};
    use tch::Tensor;

    #[test]
    fn test_kl_normal_normal() {
        let n_steps = 100;

        for n in 0..n_steps {
            let mean1: f64 = 0.;
            let std1: f64 = 1.;
            let mean2: f64 = 0.;
            let std2: f64 = 1. * n as f64 + 1e-6;

            let mean1_tensor: Tensor = mean1.into();
            let std1_tensor: Tensor = std1.into();
            let mean2_tensor: Tensor = mean2.into();
            let std2_tensor: Tensor = std2.into();

            let normal1 = Normal::new(&mean1_tensor, &std1_tensor);
            let normal2 = Normal::new(&mean2_tensor, &std2_tensor);
            let kl = normal1.kl_div(&normal2);

            let true_kl = (std2 / std1).ln()
                + (std1.powf(2.) + (mean1 - mean2).powf(2.)) * 0.5 / std2.powf(2.)
                - 0.5;
            let diff = true_kl - kl.double_value(&[]).abs();
            assert!(diff <= 1e-7);
        }
    }

    #[test]
    fn test_normal() {
        let n_steps = 100;

        let mean1: f64 = 1.;
        let std1: f64 = 1.;
        let mean1_tensor: Tensor = mean1.into();
        let std1_tensor: Tensor = std1.into();
        let normal1 = Normal::new(&mean1_tensor, &std1_tensor);

        let gauss = rv::dist::Gaussian::new(mean1, std1).unwrap();

        for n in 0..n_steps {
            let x = 2. * n as f64 / n_steps as f64;
            let x_tensor: Tensor = x.into();

            let lp = normal1.log_prob(&x_tensor);
            let true_lp = gauss.ln_pdf(&x);
            let diff_lp = (lp.double_value(&[]) - true_lp).abs();

            let cdf = normal1.cdf(&x_tensor);
            let true_cdf = gauss.cdf(&x);
            let diff_cdf = (cdf.double_value(&[]) - true_cdf).abs();

            let log_cdf = normal1.log_cdf(&x_tensor);
            let diff_log_cdf = log_cdf.double_value(&[]).exp() - true_cdf;

            assert!(diff_lp <= 1e-7);
            assert!(diff_cdf <= 1e-7);
            assert!(diff_log_cdf <= 1e-7);
        }
    }
}
