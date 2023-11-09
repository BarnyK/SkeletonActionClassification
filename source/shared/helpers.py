import re
from os import path
from typing import List, Tuple

import numpy as np
import yaml
from easydict import EasyDict as edict

from shared.dataset_info import DatasetInfo


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def create_ntu_outfile_name(output_folder: str, dataset_info: DatasetInfo, skeleton_type: str) -> str:
    filename = dataset_info.to_ntu_filename()
    file = filename + f".{skeleton_type}" + ".apskel.pkl"
    return path.join(output_folder, file)


def get_outfile_name(filepath: str, output_folder: str, skeleton_type: str) -> str:
    filename = path.split(filepath)[-1]
    no_ext_filename = path.splitext(filename)[0]
    filename = no_ext_filename + f".{skeleton_type}" + ".apskel.pkl"
    return path.join(output_folder, filename)


def sparse_to_adjacency_matrix(point_list: List[Tuple[int, int]]) -> np.ndarray:
    maxi = max([x for point in point_list for x in point])
    res = np.zeros((maxi, maxi), dtype=int)
    for x, y in point_list:
        res[x, y] = 1
        res[y, x] = 1
    return res


def calculate_interval(length, samples):
    i = 1
    while samples < length:
        samples *= 2
        i += 1
    return i


def parse_training_log(filename):
    with open(filename) as f:
        file = f.read()

    stats_regex = "\[(?P<epoch>[0-9]+)\] - (.*) - (\d+\.\d+),(\d+\.\d+),(\d+\.\d+)"
    stats_regex = re.compile(stats_regex)
    stats_match = stats_regex.findall(file)

    time_regex = "\[([0-9]+)\] - (.*) - (\d\:\d+\:\d+\.\d+)"
    time_regex = re.compile(time_regex)
    times = time_regex.findall(file)

    stats = [(int(x[0]), x[1], float(x[2]), float(x[3]), float(x[4])) for x in stats_match]
    # time_matches = [(int(x[0]), x[1], float(x[2]), float(x[3])) for x in stats_match]

    train_stats = [x for x in stats if "train" in x[1]]
    train_times = [x for x in times if "train" in x[1]]

    eval_stats = [x for x in stats if "eval" in x[1]]
    eval_times = [x for x in times if "eval" in x[1]]

    return train_stats, train_times, eval_stats, eval_times
