import re
from os import path
from typing import List, Tuple

import numpy as np
import yaml
from easydict import EasyDict as edict

from shared.structs import NtuNameData, DatasetInfo


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


template = re.compile(
    "[A-Z](?P<set>[0-9]+)[A-Z](?P<camera>[0-9]+)[A-Z](?P<person>[0-9]+)[A-Z](?P<replication>[0-9]+)[A-Z](?P<action>[0-9]+)(?:_rgb)?\.(?:avi|.+\.apskel.pkl|skeleton)"
)


def name_to_data(filepath: str):
    filename = path.split(filepath)[-1]
    match = template.match(filename)
    if match:
        data = match.groupdict()
        data = {k: int(v) for k, v in data.items()}
        return NtuNameData(**data)
    return None


def name_to_ntu_data(filepath: str) -> DatasetInfo:
    filename = path.split(filepath)[-1]
    match = template.match(filename)
    if match:
        data = match.groupdict()
        data = {k: int(v) for k, v in data.items()}
        return DatasetInfo("ntu", data)
    return None


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
