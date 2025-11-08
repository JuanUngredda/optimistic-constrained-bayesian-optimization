import json
import pickle
import os
import numpy as np
import torch
from gp import GP

from .blackboxfunction import BlackBoxFunction


class Portfolio(BlackBoxFunction):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        task_identifier="original_1_1",
        noise_std=0.01,
    ):
        print("All input dimensions in [0,1]")
        xdim = 3
        zdim = 2

        super(Portfolio, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "portfolio"
        self.xsize = xsize
        self.zsize = zsize

        self.x_domain = BlackBoxFunction.generate_discrete_points(xsize, xdim)
        self.z_domain = BlackBoxFunction.generate_discrete_points(zsize, zdim)

        self.xz_domain = self.get_discrete_xz_domain()

        self.z_probabilities = None

        # load GP hyperparameters and the portfolio data
        with open("data/portfolio/data.json", "r") as f:
            data = json.load(f)

        task_info = self.get_task_identifier_info(task_identifier)
        identifier = task_info["zdist"]

        if identifier == "task_1":
            train_x = torch.tensor(data["X"]).double()[:700, :]
            ys = torch.tensor(data["Y"]).double()[:700]
        elif identifier == "task_2":
            train_x = torch.tensor(data["X"]).double()[700:1400, :]
            ys = torch.tensor(data["Y"]).double()[700:1400]
        elif identifier == "task_3":
            train_x = torch.tensor(data["X"]).double()[1400:2100, :]
            ys = torch.tensor(data["Y"]).double()[1400:2100]
        elif identifier == "task_4":
            train_x = torch.tensor(data["X"]).double()[2100:2800, :]
            ys = torch.tensor(data["Y"]).double()[2100:2800]
        else:
            raise Exception("Unknown task: {identifier}.")

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

        # always use uniform distribution for z in the portfolio optimization experiments
        params = {"z_distribution": "uniform"}

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
        return 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / 32

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
            if self.params["z_distribution"] == "uniform":
                probabilities = torch.ones(self.zsize)

            self.z_probabilities = probabilities / torch.sum(probabilities)
        return self.z_probabilities
