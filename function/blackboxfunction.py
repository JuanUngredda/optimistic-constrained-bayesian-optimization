import os
import json
import numpy as np
import torch


class BlackBoxFunction:
    def __init__(
        self, xdim, zdim, xsize, zsize, task_identifier="target", noise_std=None
    ):
        # domain is normalized to range [0,1]
        # task_identifier is used to generate existing tasks as variants of the target task
        self.xdim = xdim
        self.zdim = zdim
        self.xsize = xsize
        self.zsize = zsize
        self.dim = self.xdim + self.zdim
        self.task_identifier = task_identifier
        self.noise_std = noise_std

        self.xz_domain = None
        self.maximizer = None
        self.maximum_value = None

    @staticmethod
    def generate_discrete_points(n, dim=1, low=0.0, high=1.0):
        if dim == 1:
            return torch.linspace(low, high, n).reshape(-1, 1)
        elif dim > 1:
            rand01 = np.loadtxt("rand01-seed0.txt")
            return torch.from_numpy(rand01[: n * dim]).reshape(n, dim)
        else:
            raise Exception("Dimension must be positive!")

    def get_GP_hyperparameters(self, task_identifier):
        hyperparameter_filename = (
            f"optimization_results/hyperparameters/{self.module_name}__{task_identifier}.json"
        )

        with open(hyperparameter_filename, "r") as f:
            hyperparameters = json.load(f)
        return hyperparameters

    def get_task_identifier_info(self, task_identifier):
        """
        A task is created by 2 modifications:
          * transformation of the blackbox function (scaling, vertical/x-horizontal shift)
            * horizontal shifting should be in direction of decision variable x
              (instead of the environmental varible z)
              since we want to change the location of the maximizer
          * distribution of environmental random variable z

          task_identifier = "[zdist_identifier]-[transformation_identifier]"
          transformation_identifier: scale_[value], vshift_[value], xhshift_[value]
            value: 1_5 -> 1.5, 0_5 -> 0.1
          e.g., "unimode_10-scale_1_5"
        """
        if "-" in task_identifier:
            zdist_identifier = task_identifier.split("-")[0]
            transform_identifier = "-".join(task_identifier.split("-")[1:])
            transform_info = transform_identifier.split("_")
            transform_type = transform_info[0]  # scale, vshift,
            transform_val = float(".".join(transform_info[1:3]))  # if xhshift we shift
            info = {
                "zdist": zdist_identifier,
                "scale": 1.0,
                "vshift": 0.0,
                "xhshift": 0.0,
                "xhshiftonly": 0.0,
            }
            info[
                transform_type
            ] = transform_val  # currently each task only has 1 transformation
            return info

        zdist_identifier = task_identifier
        return {
            "zdist": zdist_identifier,
            "scale": 1.0,
            "vshift": 0.0,
            "xhshift": 0.0,
            "xhshiftonly": 0.0,
        }

    def get_params(self, task_identifier):
        raise Exception("To be implemented in child class!")

    def get_discrete_x_domain(self):
        # return tensor of shape (nx, xdim)
        raise Exception("To be implemented in child class!")

    def get_discrete_z_domain(self):
        # return tensor of shape (nz, zdim)
        raise Exception("To be implemented in child class!")

    def get_discrete_xz_domain(self):
        # return a domain x in the first dimensions, z in the last dimensions
        #   [x0,z0], [x0,z1],..., [x0,zn], [x1,z0], [x1,z1],..., [x1,zn], ...
        if self.xz_domain is not None:
            return self.xz_domain

        with torch.no_grad():
            x_domain = self.get_discrete_x_domain()  # (nx, xdim)
            z_domain = self.get_discrete_z_domain()  # (nz, zdim)
            repeat_interleave_x_domain = x_domain.repeat_interleave(
                z_domain.shape[0], dim=0
            )
            repeat_z_domain = z_domain.repeat(x_domain.shape[0], 1)
            self.xz_domain = torch.concat(
                [repeat_interleave_x_domain, repeat_z_domain],
                dim=1,
            )
            # (nx*nz, xdim + zdim)
        return self.xz_domain

    def get_init_observations(self, n, seed=0):
        torch.manual_seed(seed)

        with torch.no_grad():
            init_idxs = torch.randint(low=0, high=self.xz_domain.shape[0], size=(n,))
            init_xz = self.xz_domain[init_idxs]
            init_y = self.get_noisy_observation(init_xz)

        return init_xz, init_y

    def get_maximizer(self):
        if self.maximizer is not None:
            return self.maximizer

        with torch.no_grad():
            func_range = self.get_noiseless_observation(self.xz_domain)
            max_idx = torch.argmax(func_range)

            self.maximizer = self.xz_domain[max_idx]
            self.maximum_value = func_range[max_idx].squeeze()

        return self.maximizer

    def get_maximum(self):
        if self.maximum_value is None:
            self.get_maximizer()
        return self.maximum_value

    def get_max_risk_measure(self, risk_measure, alpha):
        if risk_measure == "VaR":
            return self.get_max_VaR(alpha)
        elif risk_measure == "CVaR":
            return self.get_max_CVaR(alpha)
        else:
            raise Exception(f"Unknown {risk_measure}")

    def get_max_VaR(self, alpha):
        with torch.no_grad():
            func_range = self.get_noiseless_observation(self.xz_domain)
            _, VaR_vals = self.get_value_at_risk(
                func_range.reshape(self.xsize, self.zsize), alpha
            )  # (nz,)
            max_idx = torch.argmax(VaR_vals)

            VaR_maximizer = self.x_domain[max_idx]
            VaR_maximum_value = VaR_vals[max_idx].squeeze()

        return VaR_maximizer, VaR_maximum_value

    def get_max_CVaR(self, alpha):
        # TESTED
        with torch.no_grad():
            func_range = self.get_noiseless_observation(self.xz_domain)
            _, CVaR_vals = self.get_conditional_value_at_risk(
                func_range.reshape(self.xsize, self.zsize), alpha
            )  # (nz,)
            max_idx = torch.argmax(CVaR_vals)

            CVaR_maximizer = self.x_domain[max_idx]
            CVaR_maximum_value = CVaR_vals[max_idx].squeeze()

        return CVaR_maximizer, CVaR_maximum_value

    def get_noiseless_observation(self, xz):
        # return tensor of shape (x.shape[0],1)
        # applied transformation: scale, xhshift, vshift
        with torch.no_grad():
            xhshiftonly_amount = torch.tensor(
                [self.params["xhshiftonly"]] * self.xdim + [0] * self.zdim
            )
            xz = xz.reshape(-1, self.dim) + xhshiftonly_amount
            return (
                self.params["scale"]
                * self.get_noiseless_notransformed_observation(
                    xz + self.params["xhshift"]
                )
                + self.params["vshift"]
            )

    def get_max_func_eval(self):
        with torch.no_grad():
            func_evals = self.get_noiseless_observation(self.xz_domain)
        return torch.max(func_evals)

    def get_max_constrained_func_eval(self, constraint_func, threshold):
        # subject to the constraint_func(x) >= threshold
        with torch.no_grad():
            constraint_range = constraint_func.get_noiseless_observation(self.xz_domain)
            penalty = - (constraint_range < threshold) * 1e9
            constrained_func_evals = self.get_noiseless_observation(self.xz_domain) + penalty
        return torch.max(constrained_func_evals)


    def get_noisy_observation(self, xz):
        # return tensor of shape (x.shape[0],1)
        if self.noise_std is None:
            raise Exception("Unknown noise")
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            n_obs = xz.shape[0]
        return self.get_noiseless_observation(xz) + torch.randn(n_obs) * self.noise_std

    def get_z_domain_probability(self):
        # return probability mass for all points in z domain
        #        a tensor of shape (self.zsize,)
        raise Exception("To be implemented in child class!")

    def get_risk_measure_at_x_func(self, risk_measure):
        if risk_measure == "VaR":
            return self.get_value_at_risk_at_x
        elif risk_measure == "CVaR":
            return self.get_conditional_value_at_risk_at_x
        else:
            raise Exception(f"Unknown {risk_measure}.")

    def get_value_at_risk_at_x(self, x, alpha):
        # to compute the groundtruth VaR to compute the regret
        # concat x with z_domain
        x = x.reshape(-1, self.xdim)
        n = x.shape[0]

        repeat_interleave_x = x.repeat_interleave(self.zsize, dim=0)
        # (n * zsize, xdim)
        repeat_z_domain = self.z_domain.repeat(n, 1)
        # (n * zsize, zdim)

        xz = torch.concat([repeat_interleave_x, repeat_z_domain], dim=1)
        # (n * zsize, xdim + zdim)
        fvals = self.get_noiseless_observation(xz)  # (n * zsize)
        fvals = fvals.reshape(n, self.zsize)

        quantile_zs, quantile_vals = self.get_value_at_risk(fvals, alpha)
        return quantile_zs, quantile_vals

    def get_conditional_value_at_risk_at_x(self, x, alpha):
        # to compute the groundtruth VaR to compute the regret
        # concat x with z_domain
        x = x.reshape(-1, self.xdim)
        n = x.shape[0]

        repeat_interleave_x = x.repeat_interleave(self.zsize, dim=0)
        # (n * zsize, xdim)
        repeat_z_domain = self.z_domain.repeat(n, 1)
        # (n * zsize, zdim)

        xz = torch.concat([repeat_interleave_x, repeat_z_domain], dim=1)
        # (n * zsize, xdim + zdim)
        fvals = self.get_noiseless_observation(xz)  # (n * zsize)
        fvals = fvals.reshape(n, self.zsize)

        z_quatiles, CVaRs = self.get_conditional_value_at_risk(fvals, alpha)
        return z_quatiles, CVaRs  # (n,)

    def get_all_VaRs_less_than_alpha(self, val1d, alpha):
        # TESTED
        # NOTE: val1d is 1-d
        # val1d: (self.zsize,)
        # alpha: the risk
        #   (it is difficult to impleted for 2d vals)
        # returns: all VaR w.r.t alpha' such that alpha' <= alpha
        val1d = val1d.squeeze()  # (self.zsize,)
        assert len(val1d) == self.zsize

        with torch.no_grad():
            sorted_val1d_idxs = torch.argsort(val1d, descending=False)
            # (zsize,)
            sorted_val1d_probs = self.get_z_domain_probability()[sorted_val1d_idxs]
            # (zsize,)

            sorted_val1d_cumprobs = torch.cumsum(sorted_val1d_probs, dim=0)
            # (zsize)

            sorted_val1d_quantile_idx = torch.searchsorted(
                sorted_val1d_cumprobs, alpha, side="left"
            )
            # side='left' to ensure it is the smallest cdf >= alpha
            # scalar this index is based on zsorted_val1d_idxs

            risks_le_alpha = torch.clone(
                sorted_val1d_cumprobs[: sorted_val1d_quantile_idx + 1]
            )
            risks_le_alpha[
                -1
            ] = alpha  # the largest risk should be alpha NOTE: modify this way also modifies sorted_val1d_cumprobs, so we need torch.clone in the previous step
            risk_le_alpha_idxs = sorted_val1d_idxs[: sorted_val1d_quantile_idx + 1]
            VaRs_of_risk_le_alpha = val1d[risk_le_alpha_idxs]

        return risks_le_alpha, VaRs_of_risk_le_alpha

    def get_conditional_value_at_risk(self, vals, alpha):
        # TESTED
        # vals: (nx, self.zsize)
        # alpha: the risk
        # CVaR: expected of value <= VaR

        # need to compute VaR
        (
            quantile_zs,
            VaR,
            zsorted_vals_probs,
            zsorted_vals_cumprobs,
            zsorted_vals_quantile_idxs,
            zsorted_vals,
        ) = self.get_value_at_risk(vals, alpha, get_full_info=True)
        VaR = VaR.reshape(-1, 1)  # (nx,1)
        # zsorted_vals_probs: (nx, zsize)
        # zsorted_vals_cumprobs: (nx, zsize)
        # zsorted_vals_quantile_idxs, # (nx, 1)
        # zsorted_vals: (nx, zsize)

        cdf_at_VaR = torch.gather(
            zsorted_vals_cumprobs, dim=1, index=zsorted_vals_quantile_idxs
        )
        # (nx, 1)

        le_VaR_indicator = zsorted_vals <= VaR  # (nx, self.zsize)

        CVaR = (
            torch.sum(
                le_VaR_indicator * zsorted_vals * zsorted_vals_probs,
                dim=1,
                keepdim=True,
            )
            - (cdf_at_VaR - alpha) * VaR
        ) / alpha  # (nx,1)
        return quantile_zs, CVaR

    def get_value_at_risk(self, vals, alpha, get_full_info=False):
        # return value-at-risk for all x
        # x: (nx, self.zsize)
        # alpha: the risk
        # value-at-risk is defined as
        #   where the smallest cdf > alpha
        assert len(vals.shape) == 2
        assert vals.shape[1] == self.zsize
        #
        with torch.no_grad():
            n = vals.shape[0]
            # zsorted_vals_idxs = torch.argsort(vals, dim=1, descending=False)
            # (nx, zsize)
            zsorted_vals, zsorted_vals_idxs = torch.sort(
                vals, dim=1, descending=False
            )  # (nx, zsize)
            zsorted_vals_probs = self.get_z_domain_probability()[
                zsorted_vals_idxs.flatten()
            ]
            # (nx * zsize,)
            zsorted_vals_probs = zsorted_vals_probs.reshape(
                n, self.zsize
            )  # TODO: check if this is correct after flatten and reshape! checked!
            # (nx, zsize)

            zsorted_vals_cumprobs = torch.cumsum(
                zsorted_vals_probs, dim=1
            )  # (nx, zsize)
            zsorted_vals_quantile_idxs = torch.searchsorted(
                zsorted_vals_cumprobs, torch.ones(n, 1) * alpha, side="left"
            )  # side='left' to ensure it is the smallest cdf >= alpha
            # (nx, 1) this index is based on zsorted_vals_idxs

            # convert to the index of vals
            vals_quantile_idxs = torch.gather(
                zsorted_vals_idxs, dim=1, index=zsorted_vals_quantile_idxs
            )
            # (nx, 1)

            quantile_zs = self.get_discrete_z_domain()[vals_quantile_idxs.squeeze(), :]
            # (nx, zdim)
            quantile_vals = torch.gather(vals, dim=1, index=vals_quantile_idxs)
            # (nx, 1)

        if get_full_info:
            return (
                quantile_zs,
                quantile_vals.squeeze(),  # (nx,)
                zsorted_vals_probs,  # (nx, zsize)
                zsorted_vals_cumprobs,  # (nx,zsize)
                zsorted_vals_quantile_idxs,  # (nx, 1)
                zsorted_vals,  # (nx,zsize)
            )

        return quantile_zs, quantile_vals.squeeze()

    def get_risk_measure_func(self, risk_measure):
        if risk_measure == "VaR":
            return self.get_value_at_risk
        elif risk_measure == "CVaR":
            return self.get_conditional_value_at_risk
        else:
            raise Exception(f"Unknown {risk_measure}!")
