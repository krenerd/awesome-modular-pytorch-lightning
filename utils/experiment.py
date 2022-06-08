import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import data.transforms.vision as DT_V
import yaml
from data.dataset.util import torchvision_dataset
from data.transforms.base import ApplyDataTransformations, ComposeTransforms

from .verbose import set_verbose

""" Implement utilities used in `main.py`.
"""


def find_transform_from_name(f_name):
    TRANSFORM_DECLARATIONS = [DT_V]  # list of modules to serach for.
    if type(f_name) == str:
        # find transform name that matches `name` from TRANSFORM_DECLARATIONS
        is_name_in = [hasattr(file, f_name) for file in TRANSFORM_DECLARATIONS]
        assert (
            sum(is_name_in) == 1
        ), f"Transform `{f_name}` was found in `{sum(is_name_in)} files."
        file = TRANSFORM_DECLARATIONS[is_name_in.index(True)]
        print(
            f"Transform {f_name} --> {getattr(file, f_name)}: found in {file.__name__}"
        )
        return getattr(file, f_name)
    else:
        print(f"{f_name} might already be a function.")
        return f_name


def find_dataset_from_name(d_name):
    DATASET_DECLARATIONS = []  # list of modules to serach for.
    if type(d_name) == str:
        # find transform name that matches `name` from TRANSFORM_DECLARATIONS
        is_name_in = [hasattr(file, d_name) for file in DATASET_DECLARATIONS]
        assert (
            sum(is_name_in) == 1
        ), f"Dataset `{d_name}` was found in `{sum(is_name_in)} files."
        file = DATASET_DECLARATIONS[is_name_in.index(True)]
        print(f"Dataset {d_name} --> {getattr(file, d_name)}: found in {file.__name__}")
        return getattr(file, d_name)
    else:
        print(f"{d_name} might already be a function.")
        return d_name


def build_dataset(dataset_cfg, transform_cfg, const_cfg):
    # 1. build initial dataset to read data.
    dataset_mode = dataset_cfg["MODE"]
    if dataset_mode == "torchvision":
        datasets = torchvision_dataset(dataset_cfg["NAME"], dataset_cfg)
    elif dataset_mode == "from-directory":
        raise NotImplementedError("TODO!")
    else:
        raise ValueError(f"Invalid dataset type: `{dataset_mode}`")
    # datasets: dict{subset_key: torch.utils.data.Dataset, ...}

    # 2.1. build list of transformations using `transform` defined in config.
    # transforms: dict{subset_key: [t1, t2, ...], ...}
    transforms = {subset: [] for subset in datasets.keys()}
    for subsets, t_configs in transform_cfg:
        t = []
        # for each element of transforms,
        for t_config in t_configs:
            f_name, kwargs = t_config["name"], t_config.get("args", {})
            # find transform from name
            transform_f = find_transform_from_name(f_name)
            # build transform using arguments.
            kwargs["const_cfg"] = const_cfg  # feed const data such as label map.
            t.append(transform_f(**kwargs))

        for subset in subsets.split(","):
            transforms[subset] += t

    # 2.2. actually apply transformations.
    initial_transform = None
    if "initial_transform" in dataset_cfg:
        # find transform from name
        transform_f = find_transform_from_name(dataset_cfg["initial_transform"]["name"])
        # build transform using arguments.
        kwargs = dataset_cfg["initial_transform"].get("args", {})
        kwargs["const_cfg"] = const_cfg
        initial_transform = transform_f(**kwargs)
    transforms = {
        subset: ComposeTransforms(transforms[subset]) for subset in transforms.keys()
    }
    datasets = {
        subset: ApplyDataTransformations(
            base_dataset=datasets[subset],
            initial_transform=initial_transform,
            transforms=transforms[subset],
        )
        for subset in datasets.keys()
    }
    # 3. apply datasets.
    apply_dataset_cfg = dataset_cfg["transformations"]
    for subsets, d_configs in apply_dataset_cfg:
        d_operations = []
        # for each element of transforms,
        for d_config in d_configs:
            d_name, kwargs = d_config["name"], d_config["args"]
            dataset_f = find_dataset_from_name(d_name)
            # build dataset operation using arguments.
            d_operations.append(lambda base_dataset: dataset_f(base_dataset, **kwargs))

        for subset in subsets.split(","):
            for d_operation in d_operations:
                datasets[subset] = d_operation(datasets[subset])
    return datasets


def replace_non_json_serializable(cfg):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    if type(cfg) == dict:
        for key, value in cfg.items():
            if not is_jsonable(value):
                cfg[key] = replace_non_json_serializable(cfg[key])
        return cfg
    else:
        return (
            cfg
            if is_jsonable(cfg)
            else f"instance of {cfg.__class__.__name__}, pls check pkl."
        )


def initialize_environment(
    cfg=None, base_name="default-experiment", verbose="DEFAULT", debug_mode=False
):
    if cfg:
        base_name = cfg["name"]
        verbose = cfg.get("VERBOSE", "DEFAULT")
        debug_mode = "TRUE" if ("DEBUG_MODE" in cfg and cfg["DEBUG_MODE"]) else "FALSE"

    # set os.environ
    set_verbose(verbose)
    timestamp = get_timestamp()
    experiment_name = f"{base_name}-{timestamp}"
    os.environ["DEBUG_MODE"] = debug_mode

    if cfg:
        # print final config.
        pretty_cfg = replace_non_json_serializable(deepcopy(cfg))
        pretty_cfg = json.dumps(pretty_cfg, indent=2, sort_keys=True)
        print_to_end("=")

        print("modular-PyTorch-lightning")
        print("Env setup is completed, start_time:", timestamp)
        print("")
        print("Final config after merging:", pretty_cfg)

        filename = f"configs/logs/{experiment_name}"

        # pkl should be guaranteed to work.
        print(f"Saving config to: {filename}.pkl")
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(cfg, file)

        print(f"Saving config to: {filename}.yaml")
        with open(filename + ".yaml", "w") as file:
            yaml.dump(pretty_cfg, file, allow_unicode=True, default_flow_style=False)

        print(f"Saving config to: {filename}.json")
        with open(filename + ".json", "w") as file:
            json.dump(pretty_cfg, file)

    print_to_end("=")
    return experiment_name


def makedir(path):
    # check if `path` is file and remove last component.
    if Path(path).stem != path.split("/")[-1]:
        path = Path(path).parent
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def print_to_end(char="#"):
    rows, columns = os.popen("stty size", "r").read().split()
    columns = max(int(columns), 40)
    spaces = char * (columns // len(char))
    print(spaces)
