import os
import numpy as np
import torch

from .blackboxfunction import BlackBoxFunction


class Hartmann6D(BlackBoxFunction):
    def __init__(
        self,
        xsize=600,
        zsize=100,
        task_identifier="original_1_1",
        noise_std=0.01,
    ):
        print("All input dimensions in [0,1]")
        xdim = 5
        zdim = 1

        super(Hartmann6D, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "hartmann6d"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim)
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim)
        self.xz_domain = self.get_discrete_xz_domain()

        self.z_probabilities = None

    # when we change task_identifier
    # the property params changed
    @property
    def params(self):
        return self.get_params(self.task_identifier)

    # only for choosing a distribution of z such that
    # VaR and CVaR are not too easy-to-optimize functions
    def get_params(self, task_identifier):
        task_info = self.get_task_identifier_info(task_identifier)
        zdist_identifier = task_info["zdist"]
        #
        if zdist_identifier == "original_1_1":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.1},
            }
        elif zdist_identifier == "original_1_2":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.2},
            }
        elif zdist_identifier == "original_1_3":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.3},
            }
        elif zdist_identifier == "original_1_4":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 0.4},
            }
        elif zdist_identifier == "original_2_1":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.1, "sigma": 0.1},
            }
        elif zdist_identifier == "original_2_2":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.1, "sigma": 0.2},
            }
        elif zdist_identifier == "original_2_3":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.1, "sigma": 0.3},
            }
        elif zdist_identifier == "original_2_4":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.1, "sigma": 0.4},
            }
        elif zdist_identifier == "original_3_1":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.9, "sigma": 0.1},
            }
        elif zdist_identifier == "original_3_2":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.9, "sigma": 0.2},
            }
        elif zdist_identifier == "original_3_3":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.9, "sigma": 0.3},
            }
        elif zdist_identifier == "original_3_4":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.9, "sigma": 0.4},
            }
        else:
            raise Exception(
                f"{self.module_name} unknown zdist identifier: {zdist_identifier}"
            )
        #
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
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 20

    def get_noiseless_notransformed_observation(self, xz):
        # negative of Hartmann 6D
        # return tensor of shape (x.shape[0],)
        with torch.no_grad():
            A = torch.tensor(
                [
                    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
                ]
            )

            alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])

            P = 1e-4 * torch.tensor(
                [
                    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
                    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
                ]
            )

            xz = torch.tile(xz.reshape(-1, 1, self.dim), dims=(1, 4, 1))
            val = (
                2.58
                + torch.sum(
                    alpha * torch.exp(-torch.sum(A * (xz - P) ** 2, axis=2)), axis=1
                )
            ) / 1.94

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
