from __future__ import annotations

import multiprocessing
import os
import pickle
from argparse import Namespace
from functools import partial
from typing import Union

from tqdm import tqdm

import shared.datasets
from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion, assign_tids_by_order, select_by_order, \
    select_by_confidence, select_by_size, ntu_track_selection
from procedures.config import PreprocessConfig, GeneralConfig
from shared import ntu_loader
from shared.dataset_info import name_to_ntu_data
from shared.helpers import folder_check
from shared.skeletons import ntu_coco
from shared.structs import SkeletonData


def _preprocess_data_ntu(data: SkeletonData, cfg: PreprocessConfig):
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)

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


def preprocess_file(in_file: str, cfg: PreprocessConfig, return_raw_data: bool = False):
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
        if mat is None or confMat is None:
            tqdm.write(f"Empty bodies in: {in_file}")
            return in_file, None
    except ValueError:
        tqdm.write(f"Empty bodies in: {in_file}")
        return in_file, None

    # Should be of shape, M, T, V, C
    if return_raw_data:
        return data
    return data.dataset_info, mat, confMat, data.type, data.original_image_shape


def preprocess_files_to_file(files: list[str], output_path: str, cfg: PreprocessConfig):
    assert os.path.isdir(output_path)
    for file in files:
        out_file = os.path.join(output_path, os.path.split(file)[-1])
        data = preprocess_file(file, cfg, True)
        if not isinstance(data, SkeletonData):
            print("Something went wrong during preprocessing")
            continue
        data.save(out_file)


def preprocess_files(input_path: Union[str, list[str]], output_path: str, cfg: PreprocessConfig):
    assert os.path.isdir(output_path)
    files = []
    if isinstance(input_path, str):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    elif isinstance(input_path, list):
        for path in input_path:
            files += [os.path.join(path, f) for f in os.listdir(path)]

    if cfg.remove_missing_from_file and cfg.missing_file:
        with open(cfg.missing_file) as f:
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

    if isinstance(cfg.split_strategy, str):
        cfg.split_strategy = [cfg.split_strategy]

    if cfg.processes == 0:
        results = []
        for file in tqdm(files):
            results.append(preprocess_file(file, cfg))
    else:
        pool = multiprocessing.Pool(processes=cfg.processes)
        results = list(tqdm(pool.imap(partial(preprocess_file, cfg=cfg), files), total=len(files)))
        pool.close()
        pool.join()

    results = [x for x in results if x[1] is not None]

    for strategy in cfg.split_strategy:
        split_func = shared.datasets.split_map[strategy]
        train_split, test_split = split_func(results)
        tqdm.write(f"{strategy}: {len(train_split['action']) + len(test_split['action'])}")
        tqdm.write(f"\tTrain: {len(train_split['action'])}")
        tqdm.write(f"\tTest: {len(test_split['action'])}")

        if len(train_split['action']) > 0:
            train_filename = os.path.join(output_path, f"{strategy}.train.pkl")
            with open(train_filename, "wb") as f:
                pickle.dump(train_split, f)

        if len(test_split['action']) > 0:
            test_filename = os.path.join(output_path, f"{strategy}.test.pkl")
            with open(test_filename, "wb") as f:
                pickle.dump(test_split, f)


def handle_preprocess(args: Namespace):
    cfg = GeneralConfig.from_yaml_file(args.config)
    if args.processes != -1:
        cfg.prep_config.processes = args.processes
    dir_check = [os.path.isdir(x) for x in args.inputs]
    file_check = [os.path.isfile(x) for x in args.inputs]

    if not folder_check(args.save_path):
        return False
    if all(dir_check):
        preprocess_files(args.inputs, args.save_path, cfg.prep_config)
        return True
    elif all(file_check):
        preprocess_files_to_file(args.inputs, args.save_path, cfg.prep_config)
        return True
    else:
        print("Inputs are invalid.")
        if any(file_check) or any(dir_check):
            print("Inputs contain both files and directories")
        else:
            print("Some of the paths are invalid")
        return False


if __name__ == "__main__":
    cfg = GeneralConfig.from_yaml_file("./configs/general/ut_test_conf.yaml")
    cfg.prep_config.processes = 0
    preprocess_files(["/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set1_coco",
                      "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ut_set2_coco"],
                     "/media/barny/SSD4/MasterThesis/Data/prepped_data/ap_ut_test1",
                     cfg.prep_config)
