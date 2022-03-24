use crate::common::*;
use crate::{
    dist::{KLDiv, Normal, Rv},
    model::rnn::GqnNoise,
};

pub fn elbo<N1, N2>(
    target_frame: &Tensor,
    target_mean: &Tensor,
    target_std: &Tensor,
    inf_noises: &[N1],
    gen_noises: &[N2],
) -> Tensor
where
    N1: Borrow<GqnNoise>,
    N2: Borrow<GqnNoise>,
{
    let target_llh = -Normal::new(target_mean, target_std)
        .log_prob(target_frame)
        .sum_dim_intlist(&[1, 2, 3], false, Kind::Float);

    let kl_regularizer = {
        let kl_div_sum = inf_noises
            .iter()
            .zip_eq(gen_noises.iter())
            .map(|(inf_noise, gen_noise)| {
                let inf_noise = inf_noise.borrow();
                let gen_noise = gen_noise.borrow();

                let inf_normal = Normal::new(&inf_noise.means, &inf_noise.stds);
                let gen_normal = Normal::new(&gen_noise.means, &gen_noise.stds);

                inf_normal.kl_div(&gen_normal)
            })
            .reduce(|lhs, rhs| lhs + rhs)
            .unwrap();

        kl_div_sum.sum_dim_intlist(&[1, 2, 3], false, Kind::Float)
    };

    target_llh + kl_regularizer
}
