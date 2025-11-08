import json
import glob
import importlib


def importfunction(function_module_name):
    module = get_import_path(function_module_name)
    classname = get_function_class_name(function_module_name)
    return getattr(importlib.import_module(module), classname)


def get_save_path_prefix(function_module_name, existing_task_identifier):
    return f"optimization_results/{function_module_name}__{existing_task_identifier}"


def get_import_path(function_module_name):
    return f"function.{function_module_name}"


def get_function_class_name(function_module_name):
    with open("config.json", "r") as f:
        config = json.load(f)

    module2class = config["module2class"]

    if function_module_name not in module2class:
        raise Exception(
            f"Function module {function_module_name} not found in module2class"
        )
    return module2class[function_module_name]


def get_save_paths_of_module(function_module_name, risk_measure="VaR"):
    # risk_measure in ["CVaR", "VaR"]
    filenames = glob.glob(f"optimization_results/{function_module_name}*.pkl")

    if risk_measure == "VaR":
        return [filename[:-9] for filename in filenames if "CVaR" not in filename]

    return [
        filename[: filename.find("CVaR") - 1]
        for filename in filenames
        if "CVaR" in filename
    ]


def get_filename_prefix_from_task_identifiers(function_module_name, task_identifiers):
    return [
        f"optimization_results/{function_module_name}__{task_identifier}"
        for task_identifier in task_identifiers
    ]
