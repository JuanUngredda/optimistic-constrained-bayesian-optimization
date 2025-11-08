import os
import numpy as np
import torch

from .blackboxfunction import BlackBoxFunction


class WeldedBeamDesign_2_3_5_CST5(BlackBoxFunction):
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
        xdim = 3
        zdim = 1  # the last dimension

        super(WeldedBeamDesign_2_3_5_CST5, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "welded_beam_design_2_3_5_cst5"
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
        # x1 in [0.125, 2]
        # x2 in [0.1, 10]
        # x3 in [0.1, 10]
        # x4 in [0.1, 2]

        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            xz = xz * torch.tensor(
                [2.0 - 0.125, 10.0 - 0.1, 10.0 - 0.1, 2.0 - 1.0]
            ) + torch.tensor([0.125, 0.1, 0.1, 0.1])

            P = 6000.0
            L = 14.0
            E = 30 * 1e6
            G = 12 * 1e6

            sigma = 6.0 * P * L / xz[:,3] / torch.square(xz[:,2])
            sigma_max = 30000.0

            val = sigma - sigma_max

            val = val.reshape(
                -1,
            )

            max_val = 62737638.2036
            min_val = -25407.7036
            val = (val - min_val) / (max_val - min_val)
            val = (val - 0.5) * 2.0
            #
            threshold = (0.0 - min_val) / (max_val - min_val)
            threshold = (threshold - 0.5) * 2.0
            print(f"Threshold <= {threshold}")  # -0.9991903610402348

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
