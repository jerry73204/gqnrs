use tch::{nn, Tensor, Kind};
use crate::dist::{self, Normal, Rv, KLDiv};

pub fn elbo(
    means_target: &Tensor,
    stds_target: &Tensor,
    means_q: &Tensor,
    stds_q: &Tensor,
    means_pi: &Tensor,
    stds_pi: &Tensor,
    target_frame: &Tensor,
    seq_len: i64,
) -> Tensor {
    // Assume sequence first (not batch first)
    assert!(
        means_target.size()[0] == seq_len &&
            stds_target.size()[0] == seq_len &&
            means_q.size()[0] == seq_len &&
            stds_q.size()[0] == seq_len &&
            means_pi.size()[0] == seq_len &&
            stds_pi.size()[0] == seq_len
    );

    let normal_target = Normal::new(means_target, stds_target);
    let target_llh = normal_target.log_prob(target_frame).mean3(0, Kind::Float);

    let mut kl_div_sum = None;
    for ind in 0..seq_len {
        let means_q_l = means_q.select(0, ind);
        let stds_q_l = stds_q.select(0, ind);
        let normal_q_l = Normal::new(&means_q_l, &stds_q_l);

        let means_pi_l = means_pi.select(0, ind);
        let stds_pi_l = stds_pi.select(0, ind);
        let normal_pi_l = Normal::new(&means_pi_l, &stds_pi_l);

        let kl_div_l = normal_q_l.kl_div(&normal_pi_l);
        kl_div_sum = match kl_div_sum {
            Some(sum) => Some(sum + kl_div_l),
            None => Some(kl_div_l),
        };
    }

    let kl_regularizer = kl_div_sum.unwrap().mean3(0, Kind::Float);

    let elbo = target_llh + kl_regularizer;

    elbo
}
