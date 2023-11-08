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
