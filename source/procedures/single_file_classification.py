from __future__ import annotations

import os
import time
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from datasets.pose_dataset import solve_feature_transform_requirements, transform_to_tpgcn_input, \
    transform_to_stgcn_input
from datasets.transform_wrappers import calculate_channels, TransformsDict
from models import create_stgcnpp
from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from preprocessing.normalizations import create_norm_func, setup_norm_func
from procedures.config import GeneralConfig
from procedures.preprocess_files import _preprocess_data_ap
from procedures.training import load_model
from shared.datasets import adjusted_actions_maps
from shared.helpers import calculate_interval
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import visualize


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


def aggregate_results(window_results: list[tuple[int, int, torch.Tensor], ...]) -> list[int, ...]:
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


def single_file_classification(filename, cfg: GeneralConfig, model_path: Union[str, None] = None):
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
    if not model_path:
        model_path = cfg.best_model_path
    state_dict = load_model(model_path, None, None, None, device)
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
    unique_frames = []
    pp_times = []
    while True:
        frames: list[FrameData] = window_queue.get()
        if frames is None:
            break
        for frame in frames:
            if frame.seqId not in [x.seqId for x in unique_frames]:
                unique_frames.append(deepcopy(frame))
        start_frame, end_frame = frames[0].seqId, frames[-1].seqId
        wc += 1
        data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
                            len(frames), deepcopy(frames), len(frames), det_loader.frameSize, frame_interval)

        if data.length != cfg.samples_per_window:
            fill_frames(data, cfg.samples_per_window)
        pp_time = time.time()
        _preprocess_data_ap(data, cfg.prep_config)
        pp_times.append(time.time() - pp_time)
        points = data.to_matrix()
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
            features = transform_to_tpgcn_input(feature_dictionary, cfg.features, cfg.symmetry_processing)
        else:
            raise KeyError("features are wrong")
        features = torch.from_numpy(features).float().unsqueeze(0)
        features = features.to(device, non_blocking=True)
        # Run through model
        with torch.no_grad():
            out = model(features)
            top5 = torch.topk(out, 5)[1]
        # Save result
        results.append(tuple(x for x in top5[0].tolist()))
        window_results.append((start_frame, end_frame, out[0].cpu()))
    print("pptimes ",pp_times)
    print("pptimes sum ",np.sum(pp_times))
    action_map = adjusted_actions_maps[cfg.dataset]
    et = time.time()
    print(wc / (et - st))
    for res in results:
        print([action_map[x] for x in res])
    out = aggregate_results(window_results)
    print(out)

    total_data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
                              len(unique_frames), unique_frames, len(unique_frames), det_loader.frameSize,
                              frame_interval)
    for frame, action_id in zip(total_data.frames, out):
        frame.text = action_map[action_id]

    fps = 30 * cfg.samples_per_window / cfg.window_length
    #visualize(total_data, total_data.video_file, int(1000 / 30), print_frame_text=True, skip_frames=True)


def handle_classify(args: Namespace):
    cfg = GeneralConfig.from_yaml_file(args.config)
    if not os.path.isfile(args.video_file):
        print(f"{args.video_file} does not exist")
        return False
    if args.model and not os.path.isfile(args.model):
        print(f"{args.model} does not exist")
        return False

    ## TODO method mean/window
    single_file_classification(args.video_file, cfg)


if __name__ == "__main__":
    config = GeneralConfig.from_yaml_file(
        "/media/barny/SSD4/MasterThesis/Data/logs/window_tests/default_64_32_2/config.yaml")
    config.interlace = 16
    single_file_classification("/media/godchama/ssd/hoshimatic.cut2.mp4", config)
    # single_file_classification("/media/barny/SSD4/MasterThesis/Data/ut-interaction/ut-interaction_set1/seq1.avi",
    #                            config)
