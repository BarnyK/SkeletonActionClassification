from __future__ import annotations

import multiprocessing
import os
import pickle
from argparse import Namespace
from functools import partial
from typing import Union

from tqdm import tqdm

import shared.dataset_statics
from pose_estimation import ntu_loader
from procedures.config import PreprocessConfig, GeneralConfig
from procedures.utils.prep import preprocess_data_ap, preprocess_data_ntu
from shared.dataset_info import name_to_ntu_data
from shared.helpers import folder_check
from shared.structs import SkeletonData


def preprocess_file(in_file: str, cfg: PreprocessConfig, return_raw_data: bool = False):
    if cfg.alphapose_skeletons:
        data = SkeletonData.load(in_file)
        preprocess_data_ap(data, cfg)
    else:
        data = ntu_loader.read_file(in_file)
        data = preprocess_data_ntu(data, cfg)

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

    # random.seed(0)
    # files = random.sample(files, k=5000)

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

    # results = [x for x in results if x[1] is not None]
    # times_mean = np.array([x[-1] for x in results]).mean(0)
    # times_var = np.array([x[-1] for x in results]).var(0)
    # print(", ".join([f"{x:.4}" for x in times_mean]))
    # print(", ".join([f"{x:.4}" for x in times_var]))
    # return

    for strategy in cfg.split_strategy:
        split_func = shared.dataset_statics.split_map[strategy]
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
