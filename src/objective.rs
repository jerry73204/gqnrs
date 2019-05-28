use tch::{nn, Tensor};
use crate::dist::{Normal, Rv, KLDiv};

pub fn elbo(
    means_target: &Tensor,
    stds_target: &Tensor,
    means_inf: &Tensor,
    stds_inf: &Tensor,
    means_gen: &Tensor,
    stds_gen: &Tensor,
    target_frame: &Tensor,
) -> Tensor {
    let seq_len = {
        let size = means_inf.size();
        assert!(size.len() == 5);
        size[1]
    };

    // Assume batch first and equal seq length
    assert!(
        seq_len == means_inf.size()[1] &&
            seq_len == stds_inf.size()[1] &&
            seq_len == means_gen.size()[1] &&
            seq_len == stds_gen.size()[1]
    );

    let normal_target = Normal::new(means_target, stds_target);
    let target_llh = -normal_target.log_prob(target_frame)
        .mean2(&[0], false)
        .sum2(&[0, 1, 2], false);

    let mut kl_div_sum = None;
    for ind in 0..seq_len {
        let means_inf_l = means_inf.select(1, ind);
        let stds_inf_l = stds_inf.select(1, ind);
        let normal_inf_l = Normal::new(&means_inf_l, &stds_inf_l);

        let means_gen_l = means_gen.select(1, ind);
        let stds_gen_l = stds_gen.select(1, ind);
        let normal_gen_l = Normal::new(&means_gen_l, &stds_gen_l);

        let kl_div_l = normal_inf_l.kl_div(&normal_gen_l);

        kl_div_sum = match kl_div_sum {
            Some(sum) => Some(sum + kl_div_l),
            None => Some(kl_div_l),
        };
    }

    let kl_regularizer = kl_div_sum.unwrap()
        .mean2(&[0], false)
        .sum2(&[0, 1, 2], false);

    let elbo = target_llh + kl_regularizer;

    elbo
}

fn has_nan(tensor: &Tensor) -> bool {
    tensor.isnan().any().int64_value(&[]) == 1
}
