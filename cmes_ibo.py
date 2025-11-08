"""
Constrained Max-value Entropy Search with Information-Based Optimization (CMES-IBO).

Information-theoretic acquisition function that maximizes information gain
about the constrained optimum by sampling feasible functions.
"""
import torch
import numpy as np


from utils import gaussian_cdf, gaussian_log_pdf
from gp import GP


class CMES_IBO:
    """
    Constrained Max-value Entropy Search acquisition function.
    
    Uses information-theoretic approach to select queries that maximize
    information about the optimal value under constraints. Samples
    feasible functions and computes entropy reduction.
    
    Attributes
    ----------
    estimator_idx : int or None
        Index of current best estimate.
    estimator : torch.Tensor or None
        Current best point estimate.
    adaptive : bool
        Always False for CMES-IBO (coupled queries only).
    
    Notes
    -----
    Queries both target and constraints together (2 evaluations per iteration).
    Uses Monte Carlo sampling to estimate the distribution of the constrained optimum.
    """
    def __init__(self):
        self.estimator_idx = None
        self.estimator = None
        # for EIC use input query as the estimator (or the best observation so far?)
        self.adaptive = False  # only coupled queries for CMES_IPO

    @staticmethod
    def sample_f_star(
        n_sample,
        xz_domain,
        target_model,
        constraint_model_dict,
        constraint_info_list,
        cidx_to_moidx,
        threshold_list,
    ):
        # sample n target and constraint functions
        # NOTE: the number of samples returning can be < n_sample
        #       because the sampled constraints may lead to no feasible solution
        f_preds = GP.predict_f(target_model, xz_domain)
        target_samples = f_preds.sample(sample_shape=torch.Size((n_sample,)))

        constraint_sample_dict = {}
        for i, constraint_model in constraint_model_dict.items():
            f_preds = GP.predict_f(constraint_model, xz_domain)
            constraint_sample_dict[i] = f_preds.sample(
                sample_shape=torch.Size((n_sample,))
            )

        # find n samples of f_star
        f_star_samples = []
        for i in range(n_sample):
            target_sample = target_samples[i]

            feasible_cond = torch.ones((xz_domain.shape[0],), dtype=torch.bool)

            for j, threshold in enumerate(threshold_list):
                constraint_sample = constraint_sample_dict[cidx_to_moidx[j]][i]
                if constraint_info_list[j]["inequality"] == "less_than_equal_to":
                    constraint_sample = -constraint_sample

                feasible_cond = torch.logical_and(
                    feasible_cond, constraint_sample >= threshold
                )

            feasible_idxs = feasible_cond.nonzero()
            if len(feasible_idxs) > 0:
                f_star_samples.append(torch.max(target_sample[feasible_idxs]))

        return f_star_samples

    @staticmethod
    def get_estimator_idx(
        target_mean_f,
        constraint_mean_f_list,
        constraint_std_f_list,
        threshold_list,
    ):
        n_constraint = len(threshold_list)
        feasible_cond = torch.ones_like(target_mean_f, dtype=torch.bool)

        for i, threshold in enumerate(threshold_list):
            constraint_mean_f = constraint_mean_f_list[i]
            constraint_std_f = constraint_std_f_list[i]

            constraint_satisfaction_prob = 1.0 - gaussian_cdf(
                (threshold - constraint_mean_f) / constraint_std_f
            )
            #
            feasible_cond = torch.logical_and(
                feasible_cond,
                constraint_satisfaction_prob >= np.power(0.95, 1.0 / n_constraint),
            )
        #             feasible_cond = torch.logical_and(
        #                 feasible_cond, constraint_mean_f - 1.96 * constraint_std_f >= threshold
        #             )

        feasible_idxs = feasible_cond.nonzero()
        if len(feasible_idxs) > 0:
            return feasible_idxs[torch.argmax(target_mean_f[feasible_idxs])]
        else:
            # set to the maximizer of target_mean_f
            return torch.argmax(target_mean_f)

        return None

    @staticmethod
    def cmes_ibo(
        target_mean_f,
        target_std_f,
        constraint_mean_f_list,
        constraint_std_f_list,
        f_star,
        threshold_list,
    ):
        barZ = 1.0 - gaussian_cdf(
            (f_star - target_mean_f) / target_std_f
        )  # defined after equation (6) in Takeno et al., 2022

        for i, threshold in enumerate(threshold_list):
            constraint_mean_f = constraint_mean_f_list[i]
            constraint_std_f = constraint_std_f_list[i]
            barZ *= 1.0 - gaussian_cdf(
                (threshold - constraint_mean_f) / constraint_std_f
            )

        barZ = 1.0 - barZ
        return -torch.log(barZ)

    def get_query(
        self,
        threshold_list,
        target,  # {'upper_f', 'lower_f', 'mean_f', 'std_f'}
        constraint,  # {'upper_f_list', 'lower_f_list', 'mean_f_list', 'std_f_list'}
        other_info,
        # {'n_sample',
        #  'xz_domain', 'target_model',
        #  'constraint_model_dict',
        #  'constraint_info_list',
        #  'cidx_to_moidx'}
    ):
        f_star_samples = CMES_IBO.sample_f_star(
            other_info["n_sample"],
            other_info["xz_domain"],
            other_info["target_model"],
            other_info["constraint_model_dict"],
            other_info["constraint_info_list"],
            other_info["cidx_to_moidx"],
            threshold_list,
        )

        if len(f_star_samples) == 0:
            print("Cannot sample any feasible solutions! Set f_star sample to be 0.0.")
            f_star_samples = [0.0]

        cmes_ibo_vals = 0.0
        for f_star in f_star_samples:
            cmes_ibo_vals += CMES_IBO.cmes_ibo(
                target["mean_f"],
                target["std_f"],
                constraint["mean_f_list"],
                constraint["std_f_list"],
                f_star,
                threshold_list,
            ) / len(f_star_samples)

        query_idx = torch.argmax(cmes_ibo_vals)
        query_type = [0] + list(
            range(1, len(threshold_list) + 1)
        )  # query both target and constraints
        method = "CMES-IBO"

        # assign estimator of the optimal solution
        # recommend argmax_x target_mean_f
        #           s.t. for all i, Pr(c_i(x) >= lambda_i) >= 0.95
        self.estimator_idx = CMES_IBO.get_estimator_idx(
            target["mean_f"],
            constraint["mean_f_list"],
            constraint["std_f_list"],
            threshold_list,
        )

        return (
            query_idx,
            query_type,
            method,
            {"lower_margin_list": None, "upper_margin_list": None},
        )
