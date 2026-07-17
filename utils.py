"""
Utility functions for constrained Bayesian optimization.

Provides helper functions for GP posterior updates, regret computation,
constraint transformations, and statistical utilities.
"""
import numpy as np
import torch
import gpytorch

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


def standardize(y):
    return (y - torch.mean(y)) / torch.std(y)


def un_standardize(standardized_y, mean, std):
    return standardized_y * std + mean


def update_gp_posterior(
    query, gp_model, function_info, xz, y, update_gp_hyperparameters=True, training_iter=30
):
    obs = (
        function_info["func"]
        .get_noisy_observation(query)
        .reshape(
            1,
        )
    )

    xz = torch.cat([xz, query], dim=0)
    y = torch.cat([y, obs], dim=0)

    gp_model = retrain_gp(gp_model, xz, y, function_info, update_gp_hyperparameters, training_iter)

    return gp_model, xz, y


def update_target_and_constraint_posterior(
    query_type,
    query,
    target_info,
    target_xz,
    target_y,
    target_model,
    constraint_info_list,
    cidx_to_moidx,
    constraint_xz_dict,
    constraint_y_dict,
    constraint_model_dict,
    update_gp_hyperparameters=True,
    training_iter=30,
):
    for t in query_type:
        if t == 0:  # target
            print(f"  ==> Query target")
            target_model, target_xz, target_y = update_gp_posterior(
                query,
                target_model,
                target_info,
                target_xz,
                target_y,
                update_gp_hyperparameters,
                training_iter,
            )

        elif t > 0:  # constraint
            query_constraint_idx = cidx_to_moidx[t - 1]
            print(f"  ==> Query constraint model (may shared among constraints): {query_constraint_idx}")
            (
                constraint_model_dict[query_constraint_idx],
                constraint_xz_dict[query_constraint_idx],
                constraint_y_dict[query_constraint_idx],
            ) = update_gp_posterior(
                query,
                constraint_model_dict[query_constraint_idx],
                constraint_info_list[query_constraint_idx],
                constraint_xz_dict[query_constraint_idx],
                constraint_y_dict[query_constraint_idx],
                update_gp_hyperparameters,
                training_iter,
            )

    return (
        target_xz,
        target_y,
        target_model,
        constraint_xz_dict,
        constraint_y_dict,
        constraint_model_dict,
    )


def get_prior(gp_prior_config):
    gp_prior = {}

    for hyperparam, prior in gp_prior_config.items():
        if "gamma" in prior:
            gp_prior[hyperparam] = gpytorch.priors.GammaPrior(
                prior["gamma"]["concentration"], prior["gamma"]["rate"]
            )
        elif "gaussian" in prior:
            gp_prior[hyperparam] = gpytorch.priors.NormalPrior(
                prior["gaussian"]["loc"], prior["gaussian"]["scale"]
            )

    return gp_prior


def build_gp(func_info):
    if func_info["is_hyperparameter_trainable"]:
        model = GP(
            func_info["init_xz"],
            standardize(func_info["init_y"])
            if func_info["use_standardization"]
            else func_info["init_y"],
            prior=get_prior(func_info["gp_prior"]),
            ard=func_info["ard"],
            fix_mean_at=func_info["fix_mean_at"],
            learn_noise=func_info.get("learn_noise", False),
        )
    else:
        # fixed hyperparameters
        model = GP(
            func_info["init_xz"],
            standardize(func_info["init_y"])
            if func_info["use_standardization"]
            else func_info["init_y"],
            initialization={
                "likelihood.noise_covar.noise": func_info["hyperparameters"][
                    "likelihood.noise_covar.noise"
                ],
                "covar_module.base_kernel.lengthscale": func_info["hyperparameters"][
                    "covar_module.base_kernel.lengthscale"
                ],
                "covar_module.outputscale": func_info["hyperparameters"][
                    "covar_module.outputscale"
                ],
                "mean_module.constant": func_info["hyperparameters"][
                    "mean_module.constant"
                ],
            },
            ard=func_info["ard"],
            fix_mean_at=None,
        )

    return model


def retrain_gp(model, xz, y, func_info, update_gp_hyperparameters=True, training_iter=30):
    model.set_train_data(
        xz,
        standardize(y) if func_info["use_standardization"] else y,
        strict=False,
    )

    if func_info["is_hyperparameter_trainable"] and update_gp_hyperparameters:
        print(f"Update GP hyperparameters in {training_iter} iterations.")

        GP.optimize_hyperparameters(
            model,
            xz,
            standardize(y) if func_info["use_standardization"] else y,
            learning_rate=0.1,
            training_iter=training_iter,
            verbose=False,
        )

    return model


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
                [constraint_xz for constraint_xz in init_constraint_xz_dict.values()],
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

    plot(
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
        # # t is a list of constraint_type (can query multiple functions in 1 iteration)
        #         {
        #             "coord": [query.squeeze()],
        #             "marker": "X" if is_constraint_query_t else "P",
        #             "color": "y",
        #             "s_inc": 0,
        #             "label": f"{t+1}-th query",
        #         },
        {
            "coord": [maximizer.squeeze()],
            "marker": "*",
            "color": "y",
            "label": "maximizer",
            "s_inc": 60,
            "edgecolors": "k",
        },
    ]
    if estimator is not None:
        plot_points.append(
            {
                "coord": [estimator.squeeze()],
                "marker": "o",
                "color": "y",
                "label": "estimator",
                "s_inc": 0,
                "edgecolors": "k",
            }
        )

    #
    if regret is not None:
        title = f"Iteration {t+1}: {method}, regret: {regret:.2f}"
    else:
        title = f"Iteration {t+1}: {method}"
    #
    plot_levelsets = [
        {
            "vals": constraint_func_eval,
            "level": thresholds[i],
            "linestyle": "-",
            "color": [colors[i + 1]],
        }
        for i, constraint_func_eval in constraint_func_eval_dict.items()
    ]

    if lower_margin_list and upper_margin_list:
        plot_levelsets += [
            {
                # plot lower margin (for Sminus)
                "vals": -constraint_lower_f
                if inequality_type_list[i] == "less_than_equal_to"
                else constraint_lower_f,
                "level": -lower_margin_list[i]
                if inequality_type_list[i] == "less_than_equal_to"
                else lower_margin_list[i],
                "linestyle": "--"
                if inequality_type_list[i] == "less_than_equal_to"
                else "--",
                "color": [colors[i + 1]],
            }
            for i, constraint_lower_f in enumerate(constraint_lower_f_list)
        ]

    plot(
        target_upper_f,
        xsize,
        zsize,
        plot_levelsets,
        plot_points,
        title,
        plot_legend=True,
        filename=filename,
    )

    for i, constraint_upper_f in enumerate(constraint_upper_f_list):
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.imshow(
            np.reshape(constraint_upper_f.numpy(), (xsize, zsize), order="F"),
            cmap=cmap,  # plt.cm.YlOrBr,
            interpolation="bilinear",
            extent=[0.0, 1.0, 0.0, 1.0],
            origin="lower",
        )
        ax.set_title(f"Constraint {i}")
        fig.savefig(filename + f"_constraint_{i}.pdf")


def get_regret(xz, max_func_eval, target_info, constraint_info_list):
    func_eval_at_xz = target_info["func"].get_noiseless_observation(xz)
    instantaneous_target_regret = max(
        0, (max_func_eval - func_eval_at_xz).numpy().squeeze()
    )

    instantaneous_constraint_regret = 0
    for constraint_info in constraint_info_list:
        constraint_at_xz = constraint_info["func"].get_noiseless_observation(xz)
        diff = constraint_info["threshold"] - constraint_at_xz
        if constraint_info["inequality"] == "less_than_equal_to":
            diff = -diff

        instantaneous_constraint_regret += max(0, diff.numpy().squeeze())

    instantaneous_regret = instantaneous_target_regret + instantaneous_constraint_regret
    return (
        instantaneous_regret,
        instantaneous_target_regret,
        instantaneous_constraint_regret,
    )


def gaussian_cdf(val, mean=torch.tensor(0.0), std=torch.tensor(1.0)):
    return 0.5 * (1 + torch.erf((val - mean) / std / torch.sqrt(torch.tensor(2.0))))


def gaussian_log_pdf(val, mean=torch.tensor(0.0), std=torch.tensor(1.0)):
    return (
        -0.5 * torch.square((val - mean) / std)
        - torch.log(std)
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    )


def expected_improvement(max_obs, posterior_mean, posterior_std):
    standardized = (posterior_mean - max_obs) / posterior_std
    return (posterior_mean - max_obs) * gaussian_cdf(
        standardized
    ) + posterior_std * torch.exp(gaussian_log_pdf(standardized))

