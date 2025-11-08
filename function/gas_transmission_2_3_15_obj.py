import os
import numpy as np
import torch

from .blackboxfunction import BlackBoxFunction


class GasTransmission_2_3_15_OBJ(BlackBoxFunction):
    # 2.3.15 Gas Transmission Compressor Design [40]
    # in A Test-suit of Non-Convex Constrained Optimization Problems from the Real-World and Some Baseline Results
    # by Abhishek Kumar, 2019
    def __init__(
        self,
        xsize=10,
        zsize=10,
        task_identifier="original_1_1",
        noise_std=0.01,
    ):
        xdim = 2
        zdim = 2  # the last dimension

        super(GasTransmission_2_3_15_OBJ, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "gas_transmission_2_3_15_obj"
        self.xsize = xsize
        self.zsize = zsize

        # self.x_domain = torch.linspace(0.0, 1.0, xsize).reshape(-1, 1)
        # self.z_domain = torch.linspace(0.0, 1.0, zsize).reshape(-1, 1)
        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim).float()
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim).float()
        self.xz_domain = self.get_discrete_xz_domain()

        self.z_probabilities = None

    # when we change task_identifier
    # the property params changed
    @property
    def params(self):
        return self.get_params(self.task_identifier)

    def get_params(self, task_identifier):
        return {
            "z_distribution": {"mu": 0.0, "sigma": 0.1},
            "scale": 1.0,
            "vshift": 0.0,
            "xhshift": 0.0,
            "xhshiftonly": 0.0,
        }

    def get_discrete_x_domain(self):
        # return tensor of shape (nx, xdim)
        return self.x_domain

    def get_discrete_z_domain(self):
        # return tensor of shape (nz, zdim)
        return self.z_domain

    def get_beta_t(self, t):
        domain_size = self.xsize * self.zsize
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 100

    def get_noiseless_notransformed_observation(self, xz):
        # transform range [0,1] to
        # x1 in [20,50]
        # x2 in [1,10]
        # x3 in [20,50]
        # x4 in [0.1, 60]

        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            xz = xz * torch.tensor(
                [50.0 - 20.0, 10.0 - 1.0, 50.0 - 20.0, 60.0 - 0.1]
            ) + torch.tensor([20.0, 1.0, 20.0, 0.1])

            val = (
                8.61
                * 1e5
                * torch.sqrt(xz[:, 0])
                * xz[:, 1]
                * torch.pow(xz[:, 2], -2.0 / 3.0)
                * torch.pow(xz[:, 3], -0.5)
                + 3.69 * 1e4 * xz[:, 2]
                + 7.72 * 1e8 / xz[:, 0] * torch.pow(xz[:, 1], 0.219)
                - 765.43 * 1e6 / xz[:, 0]
            ) / 1e4

            # normalize to range [-1., 1.]
            val = (val - 173.9976) / (3553.9161 - 173.9976)
            val = (val - 0.5) * 2.0

            val = val.reshape(
                -1,
            )
        return val.float()

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
