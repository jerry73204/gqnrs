use crate::common::*;

pub fn log_ndtr(x: &Tensor) -> Tensor {
    log_ndtr_ext(x, 3)
}

pub fn log_ndtr_ext(x: &Tensor, series_order: i64) -> Tensor {
    // lower_segment = { -20,  x.dtype=float64
    //                 { -10,  x.dtype=float32
    // upper_segment = {   8,  x.dtype=float64
    //                 {   5,  x.dtype=float32
    let (lower_segment, upper_segment) = match x.kind() {
        Kind::Float => (Tensor::of_slice(&[-10_f32]), Tensor::of_slice(&[5_f32])),
        Kind::Double => (Tensor::of_slice(&[-20_f64]), Tensor::of_slice(&[8_f64])),
        _ => panic!("Unsupported tensor kind"),
    };

    let above_upper = -ndtr(&-x); // log(1-x) ~= -x, x << 1
    let between = ndtr(&x.max_other(&lower_segment)).log();
    let below_lower = log_ndtr_lower(&x.min_other(&lower_segment), series_order);

    above_upper.where_self(
        &x.gt_tensor(&upper_segment),
        &between.where_self(&x.gt_tensor(&lower_segment), &below_lower),
    )
}

pub fn ndtr(x: &Tensor) -> Tensor {
    let half_sqrt_2: Tensor = (0.5 * 2_f64.sqrt()).into();
    let w = x * &half_sqrt_2;
    let z = w.abs();
    let y = (w.erf() + 1.).where_self(
        &z.lt_tensor(&half_sqrt_2),
        &(-z.erfc() + 2.).where_self(&w.gt_tensor(&w.zeros_like()), &z.erfc()),
    );
    y * 0.5
}

pub fn log_ndtr_lower(x: &Tensor, series_order: i64) -> Tensor {
    let x_2 = x.pow_tensor_scalar(2);
    let log_scale = -0.5 * x_2 - (-x).log() - 0.5 * (2. * std::f64::consts::PI).ln();
    log_scale + log_ndtr_asymptotic_series(x, series_order).log()
}

pub fn log_ndtr_asymptotic_series(x: &Tensor, series_order: i64) -> Tensor {
    if series_order <= 0 {
        return Tensor::ones(&[], (Kind::Float, x.device()));
    }

    let x_2 = x.pow_tensor_scalar(2);
    let mut evem_sum = x.zeros_like();
    let mut odd_sum = x.zeros_like();
    let mut x_2n = x_2.copy();
    let mut prod: i64 = 1;

    for n in 1..(series_order + 1) {
        prod *= 2 * n - 1;
        let f_tensor: Tensor = (prod as f64).into();

        let y = f_tensor / &x_2n;

        match n % 2 {
            1 => odd_sum += y,
            _ => evem_sum += y,
        }

        x_2n *= &x_2;
    }

    return 1. + evem_sum + odd_sum;
}
