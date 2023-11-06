from __future__ import annotations

import multiprocessing
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Union, Iterable

import numpy as np
from dataclass_wizard import YAMLWizard
from tqdm import tqdm

import datasets
from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion, assign_tids_by_order, select_by_order, \
    select_by_confidence, select_by_size, body_bbox, ntu_track_selection
from shared import ntu_loader
from shared.dataset_info import name_to_ntu_data
from shared.skeletons import ntu_coco
from shared.structs import SkeletonData


@dataclass
class PreprocessConfig(YAMLWizard, key_transform='SNAKE'):
    use_box_conf: bool = True
    box_conf_threshold: float = 0.7
    box_conf_max_total: float = 0.9
    box_conf_max_frames: float = 0.9

    use_max_pose_conf: bool = True
    max_pose_conf_threshold: float = 0.55

    use_nms: bool = True

    use_tracking: bool = True
    pose_tracking_threshold: int = 90
    pose_tracking_width_ratio: float = 1.8
    pose_tracking_height_ratio: float = 0.55

    use_motion_selection: bool = True

    use_size_selection: bool = False
    use_confidence_selection: bool = False
    use_order_selection: bool = False
    max_body_count: int = 2
    keypoint_fill_type: str = "interpolation"

    transform_to_combined: bool = False
    alphapose_skeletons: bool = True
    remove_missing_from_file: bool = False


def ntu_preprocess_cfg():
    return PreprocessConfig(
        use_box_conf=False,
        use_max_pose_conf=False,
        use_nms=True,
        use_tracking=False,
        pose_tracking_threshold=90,
        pose_tracking_width_ratio=1.9,
        pose_tracking_height_ratio=0.55,
        use_motion_selection=True,
        use_size_selection=True,
        use_order_selection=False,
        max_body_count=2,
        keypoint_fill_type="interpolation",
        transform_to_combined=False,
        alphapose_skeletons=False,
        remove_missing_from_file=True,
    )


def attach_fake_bbox(data: SkeletonData):
    for frame in data.frames:
        for body in frame.bodies:
            bbox = body_bbox(body)
            body.box = np.array([bbox[0], bbox[2], bbox[1], bbox[3]])
            body.boxConf = np.array([0.8])


def _preprocess_data_ntu(data: SkeletonData, cfg: PreprocessConfig):
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)
    # box conf doesn't work because no info about boxes
    # max_pose conf doesn't work either
    # nms is only doing wrong things
    debug_copy = deepcopy(data)

    if data.no_bodies():
        return data

    ntu_track_selection(data, cfg.max_body_count)
    keypoint_fill(data, cfg.keypoint_fill_type)
    return data


def _preprocess_data_ap(data: SkeletonData, cfg: PreprocessConfig) -> SkeletonData:
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)
    if cfg.use_box_conf:
        skeleton_filters.remove_bodies_by_box_confidence(data, cfg.box_conf_threshold, cfg.box_conf_max_total,
                                                         cfg.box_conf_max_frames)
    if cfg.use_max_pose_conf:
        skeleton_filters.remove_by_max_possible_pose_confidence(data, cfg.max_pose_conf_threshold)
    if cfg.use_nms:
        nms(data, True)

    if cfg.use_size_selection:
        select_by_size(data, cfg.max_body_count)
    elif cfg.use_confidence_selection:
        select_by_confidence(data, cfg.max_body_count)
    elif cfg.use_order_selection:
        select_by_order(data, cfg.max_body_count)

    if cfg.use_tracking:
        pose_track(data.frames,
                   threshold=cfg.pose_tracking_threshold,
                   width_ratio=cfg.pose_tracking_width_ratio,
                   height_ratio=cfg.pose_tracking_height_ratio)
        if cfg.use_motion_selection:
            select_tracks_by_motion(data, cfg.max_body_count)
    else:
        assign_tids_by_order(data)
    keypoint_fill(data, cfg.keypoint_fill_type)
    return data


def preprocess_file(in_file: str, cfg: PreprocessConfig):
    if cfg.alphapose_skeletons:
        data = SkeletonData.load(in_file)
        _preprocess_data_ap(data, cfg)
    else:
        data = ntu_loader.read_file(in_file)
        data = _preprocess_data_ntu(data, cfg)

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
    return data.dataset_info, mat, confMat, data.type, data.original_image_shape


def preprocess_files(input_path: Union[str, list[str]], output_path: str, cfg: PreprocessConfig,
                     split_strategy: Union[str, Iterable[str]] = ("ntu_xsub",),
                     processes: int = 12, no_pool: bool = False, missing_file: str = None):
    files = []
    if isinstance(input_path, str):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    elif isinstance(input_path, list):
        for path in input_path:
            files += [os.path.join(path, f) for f in os.listdir(path)]

    if cfg.remove_missing_from_file and missing_file:
        with open(missing_file) as f:
            missing_files = f.read().split("\n")
        missing_files = [x.strip() for x in missing_files if x.strip()]
        missing_files = set([x for x in missing_files if len(x.split()) == 1])
        new_files = []
        for file in files:
            info = name_to_ntu_data(file)
            if info and info.to_filename() in missing_files:
                continue
            new_files.append(file)
        files = new_files

    if isinstance(split_strategy, str):
        split_strategy = [split_strategy]

    if no_pool:
        results = []
        for file in tqdm(files):
            results.append(preprocess_file(file, cfg))
    else:
        pool = multiprocessing.Pool(processes=processes)
        results = list(tqdm(pool.imap(partial(preprocess_file, cfg=cfg), files), total=len(files)))
        pool.close()
        pool.join()

    os.makedirs(output_path, exist_ok=False)
    results = [x for x in results if x[1] is not None]

    for strategy in split_strategy:
        split_func = datasets.split_map[strategy]
        train_split, test_split = split_func(results)
        tqdm.write(f"{strategy}: {len(train_split['action']) + len(test_split['action'])}")
        tqdm.write(f"\tTrain: {len(train_split['action'])}")
        tqdm.write(f"\tTest: {len(test_split['action'])}")

        train_filename = os.path.join(output_path, f"{strategy}.train.pkl")
        with open(train_filename, "wb") as f:
            pickle.dump(train_split, f)

        test_filename = os.path.join(output_path, f"{strategy}.test.pkl")
        with open(test_filename, "wb") as f:
            pickle.dump(test_split, f)

# if __name__ == '__main__':
#     preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
#                       "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
#                      "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1",
#                      PreprocessConfig(),
#                      datasets.all_splits,
#                      12,
#                      True)
#     cfg = PreprocessConfig()
#     cfg.keypoint_fill_type = "mice"
#     preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
#                       "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_120_coco"],
#                      "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_mice_fill_bad",
#                      cfg,
#                      datasets.all_splits,
#                      3,
#                      True)
#
#     cfg = ntu_preprocess_cfg()
#     preprocess_files(["/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons",
#                       "/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons_120"],
#                      "/media/barny/SSD4/MasterThesis/Data/prepped_data/ntu_test1",
#                      cfg,
#                      datasets.all_splits,
#                      3,
#                      True
#                      )
