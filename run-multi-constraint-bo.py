from pathlib import Path
import sys
import os
import json
import numpy as np
import pickle
import torch
import argparse

import gpytorch

from matplotlib import pyplot as plt
from gp import GP
from multi_constraint_bo import MultiConstraintBO, ACQUISITION_FUNCTIONS

import convertpath


def standardize(ys):
    mean_y = ys.mean()
    std_y = ys.std()
    return (ys - ys.mean()) / ys.std()


parser = argparse.ArgumentParser()
parser.add_argument("--num-rand", dest="nrand", type=int, help="Number of random runs")
parser.add_argument(
    "--plot", dest="isplot", type=int, help="1 if plotting BO iterations, 0 otherwise"
)
parser.add_argument("path", type=str, help="Path to experiments config")
parser.add_argument(
    "--acqfunc",
    choices=ACQUISITION_FUNCTIONS,
    default=ACQUISITION_FUNCTIONS[0],
    help="acquisition_function_name",
)

args = parser.parse_args()

experiment_config_filename = args.path
experiment_identifier = Path(args.path).stem
acquisition_function_name = args.acqfunc
nrand = args.nrand if args.nrand else 1
is_visualize = bool(args.isplot) if args.isplot else False

print(f"Number or random runs: {nrand}")
print(f"Path to config: {experiment_config_filename}")

with open(experiment_config_filename, "r") as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))

n_bo_iterations = data["n_bo_iterations"]
n_training_iter_gp_hyper = data["n_training_iter_gp_hyper"]
update_gp_hyper_every_iter = data["update_gp_hyper_every_iter"]


xsize = data["xsize"]
zsize = data["zsize"]
domain_size = xsize * zsize

# if acquisition_function_name.endswith("decoupled"):
#     data["beta"] *= len(data["constraints"])

target_get_beta_t = (
    lambda t: 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / data["beta"]
)
constraint_get_beta_t = (
    lambda t: 2.0 * np.log(domain_size * (t + 1) ** 2 / 6 / 0.1) / data["beta"]
)

target = {
    "name": "target",
    "module_name": data["target"]["module_name"],
    "identifier": data["target"]["identifier"],
    "noise_std": data["target"]["noise_std"],
    "n_init_observations": data["target"]["n_init_observations"],
    "func": None,
    "init_xz": None,
    "init_y": None,
    "is_hyperparameter_trainable": data["target"]["is_hyperparameter_trainable"],
    "gp_prior": data["target"]["gp_prior"],
    "hyperparameters": None,
    "use_standardization": data["target"]["use_standardization"],
    "noise_std": data["target"]["noise_std"],
    "ard": data["target"]["ard"],
    "get_beta_t": target_get_beta_t,
    "fix_mean_at": None,
}

constraint_list = [
    {
        "name": "constraint",
        "share_observation_with_constraint": constraint_info[
            "share_observation_with_constraint"
        ],
        "inequality": constraint_info["inequality"],
        "module_name": constraint_info["module_name"],
        "identifier": constraint_info["identifier"],
        "noise_std": constraint_info["noise_std"],
        "n_init_observations": constraint_info["n_init_observations"],
        "func": None,
        "init_xz": None,
        "init_y": None,
        "standardized_init_y": None,
        "is_hyperparameter_trainable": constraint_info["is_hyperparameter_trainable"],
        "gp_prior": constraint_info["gp_prior"],
        "hyperparameters": None,
        "use_standardization": constraint_info["use_standardization"],
        "noise_std": constraint_info["noise_std"],
        "ard": constraint_info["ard"],
        "threshold": constraint_info["threshold"],
        "get_beta_t": constraint_get_beta_t,
        "fix_mean_at": constraint_info["fix_mean_at"]
        if (
            "fix_mean_at"
            in data["target"]
            # and acquisition_function_name in ["ucbc2_mixed", "eic"]
        )
        else None,
    }
    for constraint_info in data["constraints"]
]

# to save the progress of BO's regret
save_result_filename_prefix = f"out/{experiment_identifier}_{acquisition_function_name}"
regrets = {}
query_types = []

for nr in range(nrand):
    # different random runs have different initial observations
    print("##########################")
    print(f"# Random Experiment {nr} #")

    # initialization
    for i, func_info in enumerate([target] + constraint_list):
        func_info["func"] = convertpath.importfunction(func_info["module_name"])(
            xsize=xsize,
            zsize=zsize,
            task_identifier=func_info["identifier"],
            noise_std=func_info["noise_std"],
        )

        if func_info["is_hyperparameter_trainable"]:
            # func_info["prior"] = GP.get_default_hyperparameter_prior()
            func_info["hyperparameters"] = None
        else:
            # func_info["prior"] = None
            func_info["hyperparameters"] = func_info["func"].get_GP_hyperparameters(
                func_info["func"].task_identifier
            )
            func_info["hyperparameters"]["likelihood.noise_covar.noise"] = max(
                0.0001, func_info["noise_std"] ** 2
            )

        # need to ensure same input for both target and constraints
        # so that the estimation of EIC and other methods are correct!
        # by setting seed = nr instead seed = i + nr
        func_info["init_xz"], func_info["init_y"] = func_info[
            "func"
        ].get_init_observations(func_info["n_init_observations"], seed=nr)

    xz_domain = target["func"].get_discrete_xz_domain()

    ucb = MultiConstraintBO()

    regrets_nr, query_types_nr = ucb.run(
        xz_domain,
        target,
        constraint_list,
        acquisition_function_name,
        n_bo_iterations,
        experiment_identifier + f"_rand_{nr}",
        n_training_iter_gp_hyper=n_training_iter_gp_hyper,
        update_gp_hyper_every_iter=update_gp_hyper_every_iter,
        is_visualize=is_visualize
    )

    query_types.append(query_types_nr)

    if len(regrets) == 0:
        regrets = regrets_nr
        for key in regrets:
            regrets[key] = np.expand_dims(regrets[key], axis=0)

    else:
        for key in regrets:
            regrets[key] = np.concatenate(
                [regrets[key], np.expand_dims(regrets_nr[key], axis=0)], axis=0
            )

    sys.stdout.flush()

    # save regret
    regret_filename = f"{save_result_filename_prefix}_regrets.pkl"
    with open(regret_filename, "wb") as f:
        pickle.dump(regrets, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save regrets to {regret_filename}.")

    # save query_types, input queries
    bo_info_filename = f"{save_result_filename_prefix}_bo_info.pkl"
    with open(bo_info_filename, "wb") as f:
        pickle.dump(
            {
                "query_types": query_types,
                "target_xz": ucb.target_xz,
                "target_y": ucb.target_y,
                "constraint_xz_dict": ucb.constraint_xz_dict,
                "constraint_y_dict": ucb.constraint_y_dict,
                "cidx_to_moidx": ucb.cidx_to_moidx,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
