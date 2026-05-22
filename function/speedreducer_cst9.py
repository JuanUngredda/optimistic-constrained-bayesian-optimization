import numpy as np
import torch

from .blackboxfunction import BlackBoxFunction
from juan_benchmark.speedreducer_math import bilog, constraint_raw


class SpeedReducer_CST9(BlackBoxFunction):
    def __init__(self, xsize=10, zsize=10, task_identifier="original_1_1", noise_std=0.01):
        xdim = 6
        zdim = 1
        super(SpeedReducer_CST9, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )
        self.module_name = "speedreducer_cst9"
        self.xsize = xsize
        self.zsize = zsize
        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim).float()
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim).float()
        self.xz_domain = self.get_discrete_xz_domain()
        self.z_probabilities = None

    @property
    def params(self):
        return self.get_params(self.task_identifier)

    def get_params(self, task_identifier):
        return {
            "z_distribution": {"mu": 0.5, "sigma": 0.01},
            "scale": 1.0,
            "vshift": 0.0,
            "xhshift": 0.0,
            "xhshiftonly": 0.0,
        }

    def get_discrete_x_domain(self):
        return self.x_domain

    def get_discrete_z_domain(self):
        return self.z_domain

    def get_beta_t(self, t):
        domain_size = self.xsize * self.zsize
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 100

    def get_noiseless_notransformed_observation(self, xz):
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            val = bilog(constraint_raw(xz, 9)).reshape(-1)
        return val.float()

    def get_z_domain_probability(self):
        if self.z_probabilities is not None:
            return self.z_probabilities
        with torch.no_grad():
            mu = self.params["z_distribution"]["mu"]
            sigma = self.params["z_distribution"]["sigma"]
            probabilities = (
                1.0
                / np.sqrt(2.0 * np.pi)
                / sigma
                * torch.exp(-0.5 * (self.z_domain - mu) ** 2 / sigma ** 2)
            ).squeeze()
            self.z_probabilities = probabilities / torch.sum(probabilities)
        return self.z_probabilities
