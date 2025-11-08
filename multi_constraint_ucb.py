import os
import numpy as np
import torch
from torch.distributions.normal import Normal

import gpytorch
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from gp import GP

reserved_colors = ["tab:purple", "tab:red", "tab:pink"]
colors = [c for c in list(mcolors.TABLEAU_COLORS.keys()) if c not in reserved_colors]
cmap = "Purples"

transform_prediction_and_threshold = {
    "greater_than_equal_to": lambda y: y,
    "less_than_equal_to": lambda y: -y,
}


class MultiConstraintUCB(object):
    def __init__(self, filename_prefix=None):
        self.target_model = None
        self.constraint_model_dict = None
        self.beta_t = None
        self.xz_domain = None
        self.regrets = None  # a dictionary {'instantaneous': instantaneous_regrets,
        # of the latest executation of run

        if filename_prefix is not None:
            self.load(filename_prefix)

    @staticmethod
    def standardize(y):
        return (y - torch.mean(y)) / torch.std(y)

    @staticmethod
    def un_standardize(standardized_y, mean, std):
        return standardized_y * std + mean

    @staticmethod
    def build_gp(func_info):
        if func_info["is_hyperparameter_trainable"]:
            model = GP(
                func_info["init_xz"],
                MultiConstraintUCB.standardize(func_info["init_y"])
                if func_info["use_standardization"]
                else func_info["init_y"],
                prior=func_info["prior"],
                ard=func_info["ard"],
            )
        else:
            # fixed hyperparameters
            model = GP(
                func_info["init_xz"],
                MultiConstraintUCB.standardize(func_info["init_y"])
                if func_info["use_standardization"]
                else func_info["init_y"],
                initialization={
                    "likelihood.noise_covar.noise": func_info["hyperparameters"][
                        "likelihood.noise_covar.noise"
                    ],
                    "covar_module.base_kernel.lengthscale": func_info[
                        "hyperparameters"
                    ]["covar_module.base_kernel.lengthscale"],
                    "covar_module.outputscale": func_info["hyperparameters"][
                        "covar_module.outputscale"
                    ],
                    "mean_module.constant": func_info["hyperparameters"][
                        "mean_module.constant"
                    ],
                },
                ard=func_info["ard"],
            )

        return model

    @staticmethod
    def retrain_gp(model, xz, y, func_info):
        model.set_train_data(
            xz,
            MultiConstraintUCB.standardize(y)
            if func_info["use_standardization"]
            else y,
            strict=False,
        )

        if func_info["is_hyperparameter_trainable"]:
            GP.optimize_hyperparameters(
                model,
                xz,
                MultiConstraintUCB.standardize(y)
                if func_info["use_standardization"]
                else y,
                learning_rate=0.1,
                training_iter=50,
                verbose=False,
            )

        return model

    @staticmethod
    def plot(
        target_func_eval,
        xsize,
        zsize,
        levelsets,  # list of {vals, level, linestyle, color}
        points,  # list of {coord, marker}
        title="Untitled",
        plot_legend=True,
        filename=None,
    ):
        # plot the ground truth function and the constraint
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.imshow(
            np.reshape(target_func_eval.numpy(), (xsize, zsize), order="F"),
            cmap=cmap,  # plt.cm.YlOrBr,
            interpolation="bilinear",
            extent=[0.0, 1.0, 0.0, 1.0],
            origin="lower",
        )

        X, Z = np.meshgrid(np.linspace(0, 1, xsize), np.linspace(0, 1, zsize))
        # CS = ax.contour(X, Z, np.reshape(constraint_func_eval.numpy(), (self.target_info["func"].xsize, self.target_info["func"].zsize), order='F'), levels=[0.0, 0.2, 0.5, 0.7, 0.8, 1.0, constraint_threshold])

        for levelset in levelsets:
            CS = ax.contour(
                X,
                Z,
                np.reshape(levelset["vals"], (xsize, zsize), order="F"),
                levels=[levelset["level"]],
                linestyles=levelset["linestyle"],
                colors=levelset["color"],
                zorder=10,
            )
            ax.clabel(CS, inline=True, fontsize=10, zorder=10)

        for j, point in enumerate(points):
            for i, coord in enumerate(point["coord"]):
                ax.scatter(
                    coord[0],
                    coord[1],
                    marker=point["marker"],
                    c=point["color"][i],
                    s=60 + point["s_inc"],
                    label=point["label"] if i == 0 else None,
                    zorder=20 + j,
                    edgecolor=point["edgecolors"]
                    if "edgecolors" in point
                    else point["color"][i],
                )
        if plot_legend:
            ax.legend(bbox_to_anchor=(1.02, 0.5))
        ax.set_title(title)
        if filename is not None:
            fig.savefig(filename)

    @staticmethod
    def visualize_groundtruth(
        xsize,
        zsize,
        init_target_xz,
        init_constraint_xz_dict,
        target_func_eval,
        constraint_func_eval_dict,
        thresholds,
        maximizer,
        filename=None,
    ):
        constraint_query_colors = []
        for i, constraint_xz in init_constraint_xz_dict.items():
            constraint_query_colors.extend([colors[i + 1]] * constraint_xz.shape[0])

        plot_points = [
            {
                "coord": torch.cat(
                    [
                        constraint_xz
                        for constraint_xz in init_constraint_xz_dict.values()
                    ],
                    dim=0,
                ),
                "marker": "X",
                "color": constraint_query_colors,
                "s_inc": 0,
                "label": "constraint queries",
            },
            {
                "coord": init_target_xz,
                "marker": "P",
                "color": [colors[0]] * init_target_xz.shape[0],
                "label": "objective queries",
                "s_inc": 0,
                "edgecolors": "k",
            },
            {
                "coord": [maximizer],
                "marker": "*",
                "color": "y",
                "label": "maximizer",
                "s_inc": 60,
                "edgecolors": "k",
            },
        ]

        MultiConstraintUCB.plot(
            target_func_eval,
            xsize,
            zsize,
            [
                {
                    "vals": constraint_func_eval,
                    "level": thresholds[i],
                    "linestyle": "-",
                    "color": [colors[i + 1]],
                }
                for i, constraint_func_eval in constraint_func_eval_dict.items()
            ],
            plot_points,
            title="Initial observations",
            plot_legend=False,
            filename=filename,
        )

    @staticmethod
    def visualize_bo_iteration(
        xsize,
        zsize,
        target_xz,
        constraint_xz_dict,
        target_upper_f,
        thresholds,
        lower_margin_list,
        upper_margin_list,
        constraint_func_eval_dict,
        constraint_upper_f_list,
        constraint_lower_f_list,
        inequality_type_list,
        is_constraint_query_t,
        maximizer,
        t,
        method,
        regret,
        query,
        estimator=None,
        filename=None,
    ):
        constraint_query_colors = []
        for i, constraint_xz in constraint_xz_dict.items():
            constraint_query_colors.extend([colors[i + 1]] * constraint_xz.shape[0])

        plot_points = [
            {
                "coord": torch.cat(
                    [constraint_xz for constraint_xz in constraint_xz_dict.values()],
                    dim=0,
                ),
                "marker": "X",
                "color": constraint_query_colors,
                "s_inc": 0,
                "label": "constraint queries",
            },
            {
                "coord": target_xz,
                "marker": "P",
                "color": [colors[0]] * target_xz.shape[0],
                "label": "objective queries",
                "s_inc": 0,
                "edgecolors": "k",
            },
            {
                "coord": [query.squeeze()],
                "marker": "X" if is_constraint_query_t else "P",
                "color": "y",
                "s_inc": 0,
                "label": f"{t+1}-th query",
            },
            {
                "coord": [estimator],
                "marker": "o",
                "color": "y",
                "label": "estimator",
                "s_inc": 0,
                "edgecolors": "k",
            },
            {
                "coord": [maximizer],
                "marker": "*",
                "color": "y",
                "label": "maximizer",
                "s_inc": 60,
                "edgecolors": "k",
            },
        ]
        #
        title = f"Iteration {t+1}: {method}, regret: {regret:.2f}"
        #
        MultiConstraintUCB.plot(
            target_upper_f,
            xsize,
            zsize,
            [
                {
                    "vals": constraint_func_eval,
                    "level": thresholds[i],
                    "linestyle": "-",
                    "color": [colors[i + 1]],
                }
                for i, constraint_func_eval in constraint_func_eval_dict.items()
            ]
            + [
                {
                    "vals": -constraint_lower_f_list[i]
                    if inequality_type_list[i] == "NEGATE"
                    else constraint_upper_f,
                    "level": -lower_margin_list[i]
                    if inequality_type_list[i] == "NEGATE"
                    else upper_margin_list[i],
                    "linestyle": "--" if inequality_type_list[i] == "NEGATE" else "--",
                    "color": [colors[i + 1]],
                }
                for i, constraint_upper_f in enumerate(constraint_upper_f_list)
            ],
            # + [
            # {
            # "vals": -constraint_upper_f_list[i]
            # if inequality_type_list[i] == "NEGATE"
            # else constraint_lower_f,
            # "level": -upper_margin_list[i]
            # if inequality_type_list[i] == "NEGATE"
            # else lower_margin_list[i],
            # "linestyle": "--" if inequality_type_list[i] == "NEGATE" else ":",
            # "color": [colors[i + 1]],
            # }
            # for i, constraint_lower_f in enumerate(constraint_lower_f_list)
            # ],
            plot_points,
            title,
            plot_legend=True,
            filename=filename,
        )

    @staticmethod
    def gaussian_cdf(val, mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        return 0.5 * (1 + torch.erf((val - mean) / std / torch.sqrt(2.0)))

    @staticmethod
    def gaussian_log_pdf(val, mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        return (
            -0.5 * torch.square((val - mean) / std)
            - torch.log(std)
            - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        )

    @staticmethod
    def expected_improvement(max_obs, posterior_mean, posterior_std):
        standardized = (posterior_mean - max_obs) / posterior_std
        return (posterior_mean - max_obs) * MultiConstraintUCB.gaussian_cdf(
            standardized
        ) + posterior_std * torch.exp(MultiConstraintUCB.gaussian_log_pdf(standardized))

    @staticmethod
    def expected_improvement_with_constraints(
        max_obs,
        posterior_mean,
        posterior_std,
        constraint_mean_f_list,
        constraint_std_f_list,
        inequality_type_list,
        threshold_list,
    ):
        ei_vals = MultiConstraintUCB.expected_improvement(
            max_obs, posterior_mean, posterior_std
        )

        eic_vals = ei_vals
        for i, inequality in enumerate(inequality_type_list):
            constraint_mean_f = constraint_mean_f_list[i]
            constraint_std_f = constraint_std_f_list[i]
            threshold = threshold_list[i]

            eic_vals *= 1.0 - MultiConstraintUCB.gaussian_cdf(
                transform_prediction_and_threshold[inequality](threshold),
                constraint_mean_f,
                constraint_std_f,
            )

        return eic_vals

    @staticmethod
    def get_upper_lower_bounds(gp_model, input_domain, beta):
        with torch.no_grad():
            preds = GP.predict_f(gp_model, input_domain)
            means = preds.mean
            stds = torch.sqrt(preds.variance)

            upper = means + beta * stds  # (400,)
            lower = means - beta * stds  # (400,)

        return lower, upper

    @staticmethod
    def get_Sminus(
        constraint_lower_f_list,
        constraint_upper_f_list,
        lower_margin_list,
        upper_margin_list,
    ):
        Sminus_cond = torch.ones_like(constraint_lower_f_list[0], dtype=torch.bool)

        for i in range(len(constraint_lower_f_list)):
            constraint_lower_f = constraint_lower_f_list[i]
            constraint_upper_f = constraint_upper_f_list[i]
            lower_margin = lower_margin_list[i]
            upper_margin = upper_margin_list[i]

            Sminus_cond_i = torch.logical_and(
                constraint_lower_f >= lower_margin, constraint_upper_f >= upper_margin
            )
            Sminus_cond = torch.logical_and(Sminus_cond, Sminus_cond_i)

        Sminus_idxs = Sminus_cond.nonzero()
        return Sminus_idxs, Sminus_cond

    @staticmethod
    def get_Sunion(constraint_upper_f_list, upper_margin_list):
        Sunion_cond = torch.ones_like(constraint_upper_f_list[0], dtype=torch.bool)
        for constraint_upper_f, upper_margin in zip(
            constraint_upper_f_list, upper_margin_list
        ):
            Sunion_cond = torch.logical_and(
                Sunion_cond, constraint_upper_f >= upper_margin
            )
        Sunion_idxs = Sunion_cond.nonzero()
        return Sunion_idxs, Sunion_cond

    @staticmethod
    def get_maximizer(ref_vals, other_vals, set_idxs=None):
        """
        max_{i in set_idxs} ref_vals[i]
        """
        if set_idxs is None:
            set_idxs = list(range(len(ref_vals)))

        ref_vals_in_set = ref_vals[set_idxs]
        maximizer_idx_in_set = torch.argmax(ref_vals_in_set)
        maximizer_idx = set_idxs[maximizer_idx_in_set]
        ref_val_at_maximizer = ref_vals[maximizer_idx]
        other_val_at_maximizer = None
        if other_vals is not None:
            other_val_at_maximizer = other_vals[maximizer_idx]

        return maximizer_idx, ref_val_at_maximizer, other_val_at_maximizer

    @staticmethod
    def get_constraint_idx_to_model_obs_idx(constraint_info_list):
        cidx_to_moidx = dict(
            zip(
                list(range(len(constraint_info_list))),
                list(range(len(constraint_info_list))),
            )
        )

        for i, constraint_info in enumerate(constraint_info_list):
            if (
                "share_observation_with_constraint" in constraint_info
                and constraint_info["share_observation_with_constraint"] > 0
            ):
                cidx_to_moidx[constraint_info["share_observation_with_constraint"]] = i

        return cidx_to_moidx

    def run(
        self,
        xz_domain,
        target_info,
        constraint_info_list,
        #   list of {func, init_xz, init_y, is_hyperparameter_trainable,
        #   prior, hyperparameters, use_standardization, ard, get_beta_t}
        #   NOTE: move the threshold to constraint_info
        n_bo_iter,
        experiment_identifier,
    ):
        self.dim = xz_domain.shape[1]
        self.xz_domain = xz_domain

        self.target_info = target_info
        self.constraint_info_list = constraint_info_list
        self.n_constraint = len(self.constraint_info_list)

        self.xsize = self.target_info["func"].xsize
        self.zsize = self.target_info["func"].zsize

        self.cidx_to_moidx = MultiConstraintUCB.get_constraint_idx_to_model_obs_idx(
            constraint_info_list
        )

        self.target_xz = self.target_info["init_xz"]
        self.target_y = self.target_info["init_y"]

        self.constraint_xz_dict = {}
        self.constraint_y_dict = {}

        for i in range(self.n_constraint):
            self.constraint_xz_dict[self.cidx_to_moidx[i]] = self.constraint_info_list[
                self.cidx_to_moidx[i]
            ]["init_xz"]
            self.constraint_y_dict[self.cidx_to_moidx[i]] = self.constraint_info_list[
                self.cidx_to_moidx[i]
            ]["init_y"]

        self.betas = []
        self.is_constraint_query = np.zeros(
            n_bo_iter, dtype=int
        )  # 0 if querying target, > 0: (index of constraint to query - 1)
        self.constraint_approx_margins = []

        # save regrets for plotting
        instantaneous_regrets = [None] * n_bo_iter
        instantaneous_target_regrets = [None] * n_bo_iter
        instantaneous_constraint_regrets = [None] * n_bo_iter

        lowest_bound_regrets = [None] * n_bo_iter
        lowest_bound_target_regrets = [None] * n_bo_iter
        lowest_bound_constraint_regrets = [None] * n_bo_iter

        # get the ground truth optimal value f*
        target_func_eval = self.target_info["func"].get_noiseless_observation(
            self.xz_domain
        )

        constraint_func_eval_dict = {}
        for i in range(self.n_constraint):
            constraint_func_eval_dict[
                self.cidx_to_moidx[i]
            ] = self.constraint_info_list[i]["func"].get_noiseless_observation(
                self.xz_domain
            )

        inequality_type_list = [
            constraint_info["inequality"]
            for constraint_info in self.constraint_info_list
        ]

        thresholds = [
            constraint_info["threshold"]
            for constraint_info in self.constraint_info_list
        ]

        max_func_eval_idx = MultiConstraintUCB.get_max_constraint_func_eval_idx(
            target_func_eval,
            constraint_func_eval_dict,
            thresholds,
            inequality_type_list,
        ).squeeze()
        max_func_eval = target_func_eval[max_func_eval_idx]
        maximizer = self.xz_domain[max_func_eval_idx].numpy().squeeze()
        print(f"Optimal value: {max_func_eval.squeeze()} at {maximizer}")

        MultiConstraintUCB.visualize_groundtruth(
            self.xsize,
            self.zsize,
            self.target_xz,
            self.constraint_xz_dict,
            target_func_eval,
            constraint_func_eval_dict,
            thresholds,
            maximizer,
            filename=f"img/{experiment_identifier}_groundtruth.pdf",
        )

        # build and train gp
        self.target_model = MultiConstraintUCB.build_gp(self.target_info)

        self.constraint_model_dict = {}
        for i in range(self.n_constraint):
            self.constraint_model_dict[
                self.cidx_to_moidx[i]
            ] = MultiConstraintUCB.build_gp(constraint_info_list[self.cidx_to_moidx[i]])

        self.target_model = MultiConstraintUCB.retrain_gp(
            self.target_model, self.target_xz, self.target_y, self.target_info
        )

        for i in self.constraint_model_dict:
            self.constraint_model_dict[i] = MultiConstraintUCB.retrain_gp(
                self.constraint_model_dict[i],
                self.constraint_xz_dict[i],
                self.constraint_y_dict[i],
                self.constraint_info_list[i],
            )

        lowest_bound = float("inf")
        estimator = None
        for t in range(n_bo_iter):
            print(f"\nIteration {t}:")
            self.betas.append(self.target_info["get_beta_t"](t))
            print(f"beta_t: {self.betas[-1]}")

            method = "Unspecified"

            with torch.no_grad():
                (
                    target_lower_f,
                    target_upper_f,
                ) = MultiConstraintUCB.get_upper_lower_bounds(
                    self.target_model, self.xz_domain, self.betas[-1]
                )

                constraint_lower_f_list = []
                constraint_upper_f_list = []

                for i in range(self.n_constraint):
                    (
                        constraint_lower_f,
                        constraint_upper_f,
                    ) = MultiConstraintUCB.get_upper_lower_bounds(
                        self.constraint_model_dict[self.cidx_to_moidx[i]],
                        self.xz_domain,
                        self.betas[-1],
                    )

                    if constraint_info_list[i]["inequality"] == "less_than_equal_to":
                        constraint_lower_f_list.append(-constraint_upper_f)
                        constraint_upper_f_list.append(-constraint_lower_f)
                    elif (
                        constraint_info_list[i]["inequality"] == "greater_than_equal_to"
                    ):
                        constraint_lower_f_list.append(constraint_lower_f)
                        constraint_upper_f_list.append(constraint_upper_f)
                    else:
                        raise Exception(
                            f"Unknown inequality type {constraint_info_list[i]['inequality']}"
                        )

                upper_margin_list = [
                    (
                        -thresholds[i]
                        if inequality_type_list[i] == "less_than_equal_to"
                        else thresholds[i]
                    )
                    for i in range(len(thresholds))
                ]

                # compute S- and S+ w.r.t. self.constraint_approx_margins[-1]
                # \tilde{S}^+ union S-
                Sunion_idxs, Sunion_cond = MultiConstraintUCB.get_Sunion(
                    constraint_upper_f_list, upper_margin_list
                )

                current_bound = float("inf")

                if len(Sunion_idxs) == 0:
                    # doing BO on the constraint
                    print("Doing BO on the constraint that is violated")
                    query_idx = None
                    for i in range(len(constraint_upper_f_list)):
                        constraint_upper_f = constraint_upper_f_list[i]
                        upper_margin = upper_margin_list[i]

                        max_idx = torch.argmax(constraint_upper_f)

                        if constraint_upper_f[max_idx] < upper_margin:
                            query_idx = max_idx
                            self.is_constraint_query[t] = i + 1
                            break

                    if query_idx is None:
                        raise Exception(
                            "Strange: S+ is empty while cannot do BO on the constraint!"
                        )
                    method = "BO - CONSTRAINT"

                else:  # len(Sunion_idxs) > 0
                    print(f"Non empty Sunion: len(Sunion_idxs) = {len(Sunion_idxs)}")

                    xplus_splus_idx = Sunion_idxs[
                        torch.argmax(target_upper_f[Sunion_idxs])
                    ]

                    margin = torch.squeeze(
                        target_upper_f[xplus_splus_idx]
                        - target_lower_f[xplus_splus_idx]
                    )
                    print("Margin:", margin)

                    lower_margin_list = [
                        upper_margin - margin
                        for i, upper_margin in enumerate(upper_margin_list)
                    ]

                    Sminus_idxs, Sminus_cond = MultiConstraintUCB.get_Sminus(
                        constraint_lower_f_list,
                        constraint_upper_f_list,
                        lower_margin_list,
                        upper_margin_list,
                    )

                    # check if xplus_splus in S^+ / S^-
                    Sunion_diff_Sminus_cond = torch.logical_and(
                        Sunion_cond, torch.logical_not(Sminus_cond)
                    )

                    is_xplus_splus_in_diffset = Sunion_diff_Sminus_cond[xplus_splus_idx]
                    query_idx = xplus_splus_idx

                    if len(Sminus_idxs) == 0 or is_xplus_splus_in_diffset:
                        print(
                            f"Doing LSE on the constraint: len(Sminus_idxs) = {len(Sminus_idxs)}, {is_xplus_splus_in_diffset}"
                        )

                        min_constraint_lower_f_at_xplus_splus_idx = None
                        min_constraint_lower_f_at_xplus_splus = 1e9

                        for i, constraint_lower_f in enumerate(constraint_lower_f_list):
                            if (
                                constraint_lower_f[xplus_splus_idx]
                                < min_constraint_lower_f_at_xplus_splus
                            ):
                                min_constraint_lower_f_at_xplus_splus = (
                                    constraint_lower_f[xplus_splus_idx]
                                )
                                min_constraint_lower_f_at_xplus_splus_idx = i

                        current_bound = (
                            constraint_upper_f_list[
                                min_constraint_lower_f_at_xplus_splus_idx
                            ][xplus_splus_idx]
                            - constraint_lower_f_list[
                                min_constraint_lower_f_at_xplus_splus_idx
                            ][xplus_splus_idx]
                        )

                        self.is_constraint_query[t] = (
                            min_constraint_lower_f_at_xplus_splus_idx + 1
                        )
                        method = "LSE - CONSTRAINT"

                    else:
                        print(
                            f"Doing BO on the objective: len(Sminus_idxs) = {len(Sminus_idxs)}, {is_xplus_splus_in_diffset}"
                        )
                        self.is_constraint_query[t] = 0
                        current_bound = (
                            target_upper_f[xplus_splus_idx]
                            - target_lower_f[xplus_splus_idx]
                        )
                        method = "BO - objective"

                # update the target or the constraint with new observation
                print("  Query idx:", query_idx)
                query = self.xz_domain[query_idx].reshape(1, self.dim)
                print(f"  Query = {query}")

                if self.is_constraint_query[t] == 0:
                    query = self.xz_domain[query_idx].reshape(1, self.dim)

                    obs = (
                        self.target_info["func"]
                        .get_noisy_observation(query)
                        .reshape(
                            1,
                        )
                    )

                    self.target_xz = torch.cat([self.target_xz, query], dim=0)
                    self.target_y = torch.cat([self.target_y, obs], dim=0)

                    self.target_model = MultiConstraintUCB.retrain_gp(
                        self.target_model,
                        self.target_xz,
                        self.target_y,
                        self.target_info,
                    )

                else:
                    query_constraint_idx = self.cidx_to_moidx[
                        self.is_constraint_query[t] - 1
                    ]
                    obs = (
                        self.constraint_info_list[query_constraint_idx]["func"]
                        .get_noisy_observation(query)
                        .reshape(
                            1,
                        )
                    )

                    self.constraint_xz_dict[query_constraint_idx] = torch.cat(
                        [self.constraint_xz_dict[query_constraint_idx], query],
                        dim=0,
                    )
                    self.constraint_y_dict[query_constraint_idx] = torch.cat(
                        [self.constraint_y_dict[query_constraint_idx], obs],
                        dim=0,
                    )

                    self.constraint_model_dict[
                        query_constraint_idx
                    ] = MultiConstraintUCB.retrain_gp(
                        self.constraint_model_dict[query_constraint_idx],
                        self.constraint_xz_dict[query_constraint_idx],
                        self.constraint_y_dict[query_constraint_idx],
                        self.constraint_info_list[query_constraint_idx],
                    )

                # compute the regret
                ## instantaneous regret
                func_eval_at_query = self.target_info["func"].get_noiseless_observation(
                    query
                )
                instantaneous_target_regret = max(
                    0, (max_func_eval - func_eval_at_query).numpy().squeeze()
                )

                instantaneous_constraint_regret = 0
                for constraint_info in self.constraint_info_list:
                    constraint_at_query = constraint_info[
                        "func"
                    ].get_noiseless_observation(query)
                    diff = constraint_info["threshold"] - constraint_at_query
                    if constraint_info["inequality"] == "less_than_equal_to":
                        diff = -diff

                    instantaneous_constraint_regret += max(0, diff.numpy().squeeze())

                instantaneous_regret = (
                    instantaneous_target_regret + instantaneous_constraint_regret
                )
                print(
                    f"  Instantaneous regret = {instantaneous_regret} = target {instantaneous_target_regret} + constraint {instantaneous_constraint_regret}"
                )
                instantaneous_regrets[t] = instantaneous_regret
                instantaneous_target_regrets[t] = instantaneous_target_regret
                instantaneous_constraint_regrets[t] = instantaneous_constraint_regret

                if current_bound < lowest_bound:
                    print(f"Update lowest bound to {current_bound}")
                    lowest_bound = current_bound
                    lowest_bound_regrets[t] = instantaneous_regret
                    lowest_bound_target_regrets[t] = instantaneous_target_regret
                    lowest_bound_constraint_regrets[t] = instantaneous_constraint_regret
                    estimator = query.numpy().squeeze()
                else:
                    print(f"NOT Update lowest bound {lowest_bound}")
                    lowest_bound_regrets[t] = lowest_bound_regrets[t - 1]
                    lowest_bound_target_regrets[t] = lowest_bound_target_regrets[t - 1]
                    lowest_bound_constraint_regrets[
                        t
                    ] = lowest_bound_constraint_regrets[t - 1]

                print(
                    f"  lowest_bound regret = {lowest_bound_regrets[t]} = target {lowest_bound_target_regrets[t]} + constraint {lowest_bound_constraint_regrets[t]}"
                )

                if t >= n_bo_iter - 1:
                    MultiConstraintUCB.visualize_bo_iteration(
                        self.xsize,
                        self.zsize,
                        self.target_xz,
                        self.constraint_xz_dict,
                        target_upper_f,
                        thresholds,
                        lower_margin_list,
                        upper_margin_list,
                        constraint_func_eval_dict,
                        constraint_upper_f_list,
                        constraint_lower_f_list,
                        inequality_type_list,
                        self.is_constraint_query[t],
                        maximizer,
                        t,
                        method,
                        lowest_bound_regrets[t],
                        query.numpy().squeeze(),
                        estimator,
                        filename=f"img/{experiment_identifier}_bo_iter_{t}.pdf",
                    )

        self.regrets = {
            "instantaneous_regrets": instantaneous_regrets,
            "instantaneous_target_regrets": instantaneous_target_regrets,
            "instantaneous_constraint_regrets": instantaneous_constraint_regrets,
            "lowest_bound_regrets": lowest_bound_regrets,
            "lowest_bound_target_regrets": lowest_bound_target_regrets,
            "lowest_bound_constraint_regrets": lowest_bound_constraint_regrets,
        }
        return self.regrets

    @staticmethod
    def get_max_constraint_func_eval(
        func_eval, constraint_eval_dict, thresholds, inequality_type_list
    ):
        with torch.no_grad():
            idx = MultiConstraintUCB.get_max_constraint_func_eval_idx(
                func_eval, constraint_eval_dict, thresholds, inequality_type_list
            )
            return func_eval[idx]

    @staticmethod
    def get_max_constraint_func_eval_idx(
        func_eval, constraint_eval_dict, thresholds, inequality_type_list
    ):
        with torch.no_grad():
            feasible_cond = torch.ones_like(func_eval, dtype=torch.bool)

            for i, inequality in enumerate(inequality_type_list):
                constraint_eval = constraint_eval_dict[i]

                feasible_cond = torch.logical_and(
                    feasible_cond,
                    transform_prediction_and_threshold[inequality](constraint_eval)
                    >= transform_prediction_and_threshold[inequality](thresholds[i]),
                )

            feasible_idxs = feasible_cond.nonzero()
            return feasible_idxs[torch.argmax(func_eval[feasible_idxs])]

    def predict(self, xz):
        xz = xz.reshape(-1, self.dim)
        f_preds = GP.predict_f(self.target_model, xz)

        with torch.no_grad():
            f_means = f_preds.mean
            f_vars = f_preds.variance
            f_stds = torch.sqrt(f_vars)
            lower = f_means - self.beta_t * f_stds
            upper = f_means + self.beta_t * f_stds

        return f_means, f_stds, lower, upper

    def save(self, filename_prefix):
        """
        domain
        observations
        beta_t
        GP hyperparameters
        """
        if self.target_model is None or self.beta_t is None:
            raise Exception("Haven't trained GP model")
        #
        data = {
            "xz_domain": self.xz_domain,
            "dim": self.dim,
            "observed_target_xz": self.observed_target_xz,
            "observed_target_y": self.observed_target_y,
            "beta_t": self.beta_t,
            "has_prior": self.has_prior,
            "use_standardization": self.use_standardization,
            "ard": self.ard,
            # regrets of the latest runs (None if haven't run)
            "regrets": self.regrets,
        }
        gp_filename = filename_prefix + "__GP.pth"
        self.target_model.save(gp_filename)
        #
        BO_filename = filename_prefix + "__BO_data.pkl"
        with open(VaR_filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save BO results to {gp_filename}")
        print(f"               and {BO_filename}")

    #
    def load(self, filename_prefix):
        with open(filename_prefix + "__BO_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.xz_domain = data["xz_domain"]

            self.dim = data["dim"]

            self.observed_target_xz = data["observed_target_xz"]
            self.observed_target_y = data["observed_target_y"]
            self.beta_t = data["beta_t"]
            self.has_prior = data["has_prior"]
            self.use_standardization = data["use_standardization"]
            self.ard = data["ard"]
            self.regrets = data["regrets"] if "regrets" in data else None

            ard_num_dims = self.dim if self.ard else None

        self.target_model = GP(
            self.observed_target_xz, self.observed_target_y, ard=self.ard
        )
        print(
            f"DEBUG: lengthscale {self.target_model.covar_module.base_kernel.raw_lengthscale}"
        )
        if self.has_prior:
            self.target_model.likelihood.noise_covar.register_prior(
                "noise_std_prior",
                gpytorch.priors.NormalPrior(0.0, 1.0),  # dummy prior, will be loaded
                lambda module: module.noise.sqrt(),
            )

            self.target_model.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.GammaPrior(
                        2.0, 4.0
                    ),  # dummy prior, will be loaded
                    ard_num_dims=ard_num_dims,
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(
                    2.0, 0.15
                ),  # dummy prior, will be loaded
            )

        GP.load(self.target_model, filename_prefix + "__GP.pth")
        print(
            f"DEBUG: lengthscale {self.target_model.covar_module.base_kernel.raw_lengthscale}"
        )
