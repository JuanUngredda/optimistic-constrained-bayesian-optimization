import os
import pickle
import numpy as np
import torch

from .blackboxfunction import BlackBoxFunction


class Cifar10CNN_CST3(BlackBoxFunction):
    def __init__(
        self,
        xsize=10,
        zsize=10,
        task_identifier="original_1_1",
        noise_std=0.01,
    ):
        xdim = 4
        zdim = 1

        super(Cifar10CNN_CST3, self).__init__(
            xdim, zdim, xsize, zsize, task_identifier, noise_std=noise_std
        )

        self.module_name = "cifar10_cnn_cst3"
        func_data = Cifar10CNN_CST3.get_objfunc_and_constraints()

        self.xz_domain = torch.from_numpy(func_data["normalized_input"]).float()
        self.vals = torch.from_numpy(func_data["10_constraints"][2]).squeeze().float()

        self.xzsize = self.xz_domain.shape[0]
        self.z_probabilities = None


    @staticmethod
    def get_objfunc_and_constraints():
        C = 10
        with open("function/cnn_CIFAR10_data/cnn_CIFAR10_data.pickle", "rb") as f:
            data = pickle.load(f)

        g_thresholds = 0.5 * np.ones(C)
        feasible_index = np.where(
            np.all(data[:, 5:15] >= g_thresholds, axis=1) == True
        )[0]

        X = data[:, 0:5]
        minX = np.min(X, axis=0, keepdims=True)
        maxX = np.max(X, axis=0, keepdims=True)
        normalized_X = (X - minX) / (maxX - minX) # normalize to range [0,1]
        Y = [data[:, 15]]
        Y.extend([data[:, 5 + i] for i in range(C)])

        # logit transformation
        Y = [np.where(Y <= 0, 1e-5, Y) for Y in Y]
        Y = [np.where(Y >= 1, 1 - 1e-5, Y) for Y in Y]
        Y = [np.log(Y / (1 - Y)) for Y in Y]
        g_thresholds = np.log(g_thresholds / (1 - g_thresholds))

        return {
            "input": X,
            "normalized_input": normalized_X,
            "objective_function": Y[0],
            "10_constraints": Y[1:],
            "ge_thresholds": g_thresholds,
        }

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

    def get_beta_t(self, t):
        return 2.0 * np.log(self.xzsize * (t + 1) ** 2 / 6 / 0.1) / 100

    def get_noiseless_notransformed_observation(self, xz):
        # return tensor of shape (x.shape[0],)
        with torch.no_grad():
            xz = xz.reshape(-1, self.dim)
            diff = xz.unsqueeze(1) - self.xz_domain
            diff = torch.sum(diff * diff, 2)
            idxs = (diff < 1e-6).nonzero()[:,1]
            return self.vals[idxs]

