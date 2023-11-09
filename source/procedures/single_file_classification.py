from __future__ import annotations
import os
import time
from collections import defaultdict
from copy import deepcopy
from queue import Queue
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
from typing import List

from datasets.pose_dataset import solve_feature_transform_requirements, transform_to_2pgcn_input, \
    transform_to_stgcn_input
from datasets.transform_wrappers import calculate_channels, TransformsDict
from models import create_stgcnpp
from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from preprocessing.normalizations import create_norm_func, setup_norm_func
from procedures.config import GeneralConfig
from procedures.preprocess_files import _preprocess_data_ap
from procedures.training import load_model
from shared.helpers import calculate_interval
from shared.statics import all_actions
from shared.structs import SkeletonData, FrameData


def window_worker(
        q: Queue, datalen: int, pose_data_queue: Queue, length: int, interlace: int
):
    window = []
    for i in tqdm(range(datalen), disable=False):
        data = pose_data_queue.get()
        if data is None:
            break
        x = FrameData(i, len(data), data)
        window.append(x)
        if len(window) == length:
            q.put(window)
            window = window[-interlace:]
    if len(window) > interlace:
        q.put(window)
    q.put(None)


def run_window_worker(
        datalen: int, pose_data_queue: Queue, length: int, interlace: int
):
    q = Queue(5)
    window_worker_thread = Thread(
        target=window_worker, args=(q, datalen, pose_data_queue, length, interlace)
    )
    window_worker_thread.start()
    return q


def fill_frames(data: SkeletonData, size: int):
    seq = data.frames[-1].seqId
    for i in range(size - len(data.frames)):
        data.frames.append(FrameData(seq + i + 1, 0, []))
    data.length = data.lengthB = size


def aggregate_results(window_results: tuple[int, int, torch.Tensor]) -> list[int, ...]:
    per_frame = defaultdict(list)
    for start_frame, end_frame, data in window_results:
        for fi in range(start_frame, end_frame + 1):
            per_frame[fi].append(data)
    out = []
    for i, data in per_frame.items():
        if len(data) == 1:
            out.append(data[0].max(0)[1].item())
            continue
        tmp = torch.concat([x.unsqueeze(1) for x in data], dim=1)
        out.append(tmp.mean(1).max(0)[1].item())
    return out


def single_file_classification(filename, cfg: GeneralConfig):
    assert os.path.isfile(filename)

    # Setup
    pose_cfg = cfg.pose_config
    device = torch.device(cfg.device)
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device)

    # Detection
    detector = init_detector(opts, det_cfg)
    detector.load_model()

    frame_interval = calculate_interval(cfg.window_length, cfg.samples_per_window)

    det_loader = DetectionLoader(
        filename, detector, ap_cfg, opts, "video",
        pose_cfg.detector_batch_size, pose_cfg.detector_queue_size, frame_interval
    )
    det_loader.start()

    # Pose estimation
    pose_model = init_pose_model(device, ap_cfg, ap_cfg.weights_file)
    pose_data_queue = run_pose_worker(
        pose_model, det_loader, opts, pose_cfg.estimation_batch_size, pose_cfg.estimation_queue_size
    )

    window_queue = run_window_worker(det_loader.length, pose_data_queue, cfg.samples_per_window, cfg.interlace)

    # Classification
    norm_func = create_norm_func(cfg.normalization_type)

    state_dict = load_model(cfg.best_model_path, None, None, None, device)
    num_classes = state_dict['net'][[x for x in state_dict['net'].keys()][-1]].shape[0]

    channels = calculate_channels(cfg.features, 2)
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(num_classes, channels, cfg.skeleton_type)
        model.to(device)
    else:
        raise ValueError("2p-gcn not supported yet")

    # Load state dict
    model.load_state_dict(state_dict['net'])
    model.eval()
    if norm_state_dict := state_dict.get("normalization"):
        setup_norm_func(norm_func, state_dict=norm_state_dict)
    else:
        setup_norm_func(norm_func, train_file=cfg.train_config.train_file)

    # Set up feature
    required_transforms = solve_feature_transform_requirements(cfg.features)
    transforms = {key: TransformsDict[key](cfg.skeleton_type) for key in required_transforms}

    st = time.time()
    wc = 0
    results = []
    window_results = []
    while True:
        frames: list[FrameData] = window_queue.get()
        if frames is None:
            break
        start_frame, end_frame = frames[0].seqId, frames[-1].seqId
        wc += 1
        data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
                            len(frames), deepcopy(frames), len(frames), det_loader.frameSize, frame_interval)

        if data.length != cfg.samples_per_window:
            fill_frames(data, cfg.samples_per_window)
        _preprocess_data_ap(data, cfg.prep_config)
        points = points = data.to_matrix()
        if points is None:
            continue
        points = norm_func(points)
        # Calc features
        feature_dictionary = {"joints": points}
        for feature in required_transforms:
            transforms[feature](feature_dictionary)

        # Transforms to correct input size
        if isinstance(cfg.features[0], str):
            features = transform_to_stgcn_input(feature_dictionary, cfg.features)
        elif isinstance(cfg.features[0], list):
            features = transform_to_2pgcn_input(feature_dictionary, cfg.features, cfg.symmetry_processing)
        else:
            raise KeyError("features are wrong")
        features = torch.from_numpy(features).float().unsqueeze(0)
        features = features.to(device, non_blocking=True)
        # Run through model
        with torch.no_grad():
            out = model(features)
            top5 = torch.topk(out, 5)[1]
        # Save result
        results.append(tuple(all_actions[x + 1] for x in top5[0].tolist()))
        window_results.append((start_frame, end_frame, out[0].cpu()))

    et = time.time()
    print(wc / (et - st))
    for res in results: print(res)
    out = aggregate_results(window_results)
    print(out)
    # tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=False)
    # frames = []
    # for i in tq:
    #     data = pose_data_queue.get()
    #     if data is None:
    #         break
    #     frame_data = FrameData(i, len(data), data)
    #     frames.append(frame_data)
    #
    # data = SkeletonData(
    #     "estimated",
    #     cfg.skeleton_type,
    #     DatasetInfo(),
    #     filename,
    #     det_loader.datalen,
    #     frames,
    #     len(frames),
    #     det_loader.frameSize,
    #     cfg.frame_interval
    # )
    # visualize(data, data.video_file, wait_key=1000//30, draw_bbox=True, draw_confidences=True, draw_frame_number=True)


if __name__ == "__main__":
    config = GeneralConfig.from_yaml_file("/media/barny/SSD4/MasterThesis/Data/logs/default_64_32_0/config.yaml")
    config.interlace = 16
    single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.1.avi", config)
    single_file_classification("/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S001C001P001R001A006_rgb.avi", config)
    # single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.2.avi", config)
    # single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.2.avi", config)
