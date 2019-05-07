use tch::{self, nn, Tensor};

pub trait Rv {
    fn sample(&self) -> Tensor;
    fn log_prob(&self, value: &Tensor) -> Tensor;
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
}

impl<'a> Rv for Normal<'a> {
    fn sample(&self) -> Tensor {
        Tensor::normal2(&self.mean, &self.std)
    }

    fn log_prob(&self, value: &Tensor) -> Tensor {
        let var = self.std.pow(2);
        let log_scale = self.std.log();
        -(value - self.mean).pow(2) / (2 * var) - log_scale - (2_f64 * std::f64::consts::PI).sqrt().ln()
    }

    fn cdf(&self, value: &Tensor) -> Tensor {
        0.5_f64 * (1_f64 + Tensor::erf(
            &( (value - self.mean) * self.std.reciprocal() / (2_f64).sqrt()) )
        )
    }
}

impl<'a> KLDiv<Normal<'a>> for Normal<'a> {
    fn kl_div(& self, other: & Normal) -> Tensor {

        let var_ratio = (self.std / other.std).pow(2);
        let t1 = ((self.mean - other.mean) / other.std).pow(2);
        0.5_f64 * (&var_ratio + t1 - 1_f64 - &var_ratio.log())
    }
}
