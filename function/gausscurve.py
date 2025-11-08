import os
import json
import numpy as np
import torch


from .blackboxfunction import BlackBoxFunction


class GaussCurve(BlackBoxFunction):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        task_identifier="unimode_10",
        noise_std=0.01,
    ):
        print("All input dimensions in [0,1]")
        xdim = 1
        zdim = 1

        super(GaussCurve, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "gausscurve"
        self.xsize = xsize
        self.zsize = zsize
        # self.x_domain = torch.linspace(0.0, 1.0, xsize).reshape(-1, 1)
        # self.z_domain = torch.linspace(0.0, 1.0, zsize).reshape(-1, 1)
        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim)
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim)
        self.xz_domain = self.get_discrete_xz_domain()

        self.z_probabilities = None

    # when we change task_identifier
    # the property params changed
    @property
    def params(self):
        return self.get_params(self.task_identifier)

    def get_params(self, task_identifier):
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
        task_info = self.get_task_identifier_info(task_identifier)
        zdist_identifier = task_info["zdist"]

        if zdist_identifier == "unimode_alpha10":  # should use alpha = 0.1
            params = {
                "amp": torch.Tensor([5.0]),
                "mu": torch.Tensor([[0.1, 0.1]]),
                "sigma": torch.Tensor([0.05, 0.5]),
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        elif zdist_identifier == "unimode_10":  # should use alpha = 0.1
            params = {
                "amp": torch.Tensor([0.5]),
                "mu": torch.Tensor([[0.1, 0.7]]),
                "sigma": torch.Tensor([0.5]),
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        elif zdist_identifier == "unimode1_10":  # should use alpha = 0.1
            params = {
                "amp": torch.Tensor([0.5]),
                "mu": torch.Tensor([[0.9, 0.2]]),
                "sigma": torch.Tensor([0.5]),
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        elif (
            zdist_identifier == "bimode_90"
        ):  # should use alpha=0.9 to generate VaR with 2 modes
            params = {
                "amp": torch.Tensor([0.43, 0.5]),
                "mu": torch.Tensor([[0.1, 0.7], [0.9, 0.0]]),
                "sigma": torch.Tensor([0.2, 0.2]),
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        elif (
            zdist_identifier == "bimode1_90"
        ):  # should use alpha=0.9 to generate VaR with 2 modes
            params = {
                "amp": torch.Tensor([0.45, 0.8]),
                "mu": torch.Tensor([[0.1, 0.7], [0.9, 0.0]]),
                "sigma": torch.Tensor([0.5, 0.5]),
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        else:
            raise Exception(f"GaussCurve unknown zdist identifier: {zdist_identifier}")

        params["scale"] = task_info["scale"]
        params["vshift"] = task_info["vshift"]
        params["xhshift"] = task_info["xhshift"]
        params["xhshiftonly"] = task_info["xhshiftonly"]
        return params

    def get_discrete_x_domain(self):
        # return tensor of shape (nx, xdim)
        return self.x_domain

    def get_discrete_z_domain(self):
        # return tensor of shape (nz, zdim)
        return self.z_domain

    def get_beta_t(self, t):
        domain_size = self.xsize * self.zsize
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 5

    def get_noiseless_notransformed_observation(self, xz):
        # return tensor of shape (x.shape[0],)
        # haven't applied any transformation
        with torch.no_grad():
            xz = xz.reshape(-1, 1, self.dim)

            amp = self.params["amp"].reshape(1, -1)
            mu = self.params["mu"].reshape(1, -1, self.dim)
            sigma = self.params["sigma"].reshape(1, -1)

            diff = (xz - mu) / sigma  # (n, n_param, dim)
            square_dist = torch.sum(diff * diff, dim=2)  # (n, n_param)
            val = torch.sum(amp * torch.exp(-square_dist), dim=1)
        return val

    def get_z_domain_probability(self):
        # return probability mass for all points in z domain
        #        a tensor of shape (self.zsize,)
        if self.z_probabilities is not None:
            return self.z_probabilities

        with torch.no_grad():
            mu = self.params["z_distribution"]["mu"]
            sigma = self.params["z_distribution"]["sigma"]
            probabilities = (
                1.0
                / np.sqrt(2.0 * np.pi)
                / sigma
                * torch.exp(-0.5 * (self.z_domain - mu) ** 2 / sigma**2)
            ).squeeze()

            self.z_probabilities = probabilities / torch.sum(probabilities)
        return self.z_probabilities
