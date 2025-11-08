import json
import pickle
import os
import numpy as np
import torch
from gp import GP


from .blackboxfunction import BlackBoxFunction


class Yacht(BlackBoxFunction):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        task_identifier="original_1_1",
        noise_std=0.01,
    ):
        print("All input dimensions in [0,1]")
        xdim = 5
        zdim = 1

        super(Yacht, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "yacht"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim)
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim)

        self.xz_domain = self.get_discrete_xz_domain()

        self.z_probabilities = None

        # load GP hyperparameters and the yacht hydrodynamics data
        with open("data/yacht-hydrodynamics/data.json", "r") as f:
            data = json.load(f)
            train_x = torch.tensor(data["X"]).double()
            ys = torch.tensor(data["Y"]).double()
            hyperparameters = data["hyperparameters"]
            self.gp = GP(train_x, ys, initialization=hyperparameters, ard=True)

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

        if zdist_identifier == "original_unif":
            params = {
                # distribution of z
                "z_distribution": {"mu": 0.5, "sigma": 100.0},
            }
        elif zdist_identifier == "original_1_1":
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
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 25

    def get_noiseless_notransformed_observation(self, xz):
        # return tensor of shape (x.shape[0],)
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)

        with torch.no_grad():
            f_preds = GP.predict_f(self.gp, xz)
            f_preds_mean = f_preds.mean

        return f_preds_mean.reshape(
            -1,
        )

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
