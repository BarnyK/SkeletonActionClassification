from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Union

from tqdm import tqdm

import datasets
from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion
from shared.structs import SkeletonData


@dataclass
class PreprocessConfig:
    box_conf_threshold: float = 0.7
    max_pose_conf_threshold: float = 0.7
    pose_tracking_threshold: int = 90
    keypoint_fill_type: str = "interpolation"
    use_nms: bool = True


def _preprocess_data(data: SkeletonData, config: PreprocessConfig):
    skeleton_filters.remove_bodies_by_box_confidence(data, 0.7, 0.9, 0.9)
    skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.55)
    nms(data, True)
    pose_track(data.frames, threshold=90, width_ratio=1.8, height_ratio=0.55)
    select_tracks_by_motion(data, 2)
    keypoint_fill(data, "interpolation")
    return data


def preprocess_file(in_file: str, config: PreprocessConfig):
    data = SkeletonData.load(in_file)
    _preprocess_data(data, config)
    if data.no_bodies():
        tqdm.write(f"Empty bodies in: {in_file}")
        return in_file, None
    try:
        mat = data.to_matrix()
        confMat = data.to_matrix_confidences()
    except ValueError:
        tqdm.write(f"Empty bodies in: {in_file}")
        return in_file, None
    # Should be of shape, M, T, V, C
    return data.dataset_info.__dict__, mat, confMat


def preprocess_files(input_path: Union[str, list[str]], output_path: str, config: PreprocessConfig,
                     split_strategy: str = "ntu_xsub",
                     processes: int = 12, no_pool: bool = False):
    files = []
    if isinstance(input_path, str):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    elif isinstance(input_path, list):
        for path in input_path:
            files += [os.path.join(path, f) for f in os.listdir(path)]

    files = files[:100]

    if no_pool:
        results = []
        for file in files:
            results.append(preprocess_file(file, config))
    else:
        pool = multiprocessing.Pool(processes=processes)
        results = list(tqdm(pool.imap(partial(preprocess_file, config=config), files), total=len(files)))
        pool.close()
        pool.join()

    split_func = datasets.split_map[split_strategy]
    train_split, test_split = split_func(results)

    data = {
        "labels": [x[0] for x, y, *_ in results if y is not None],
        "points": [y for x, y, in results if y is not None]
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/ntu_coco.combined", PreprocessConfig(), "ntu_xsub",1, True)
