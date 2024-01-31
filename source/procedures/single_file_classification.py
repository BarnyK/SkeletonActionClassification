from __future__ import annotations

import os
import time
import warnings
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
from models import create_stgcnpp, create_tpgcn
from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from preprocessing.normalizations import create_norm_func, setup_norm_func
from procedures.config import GeneralConfig
from procedures.training import load_model
from procedures.utils.prep import preprocess_per_frame, preprocess_data_rest
from shared.dataset_statics import adjusted_actions_maps
from shared.helpers import calculate_interval, run_qsp, fill_frames, swap_extension
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import Visualizer

# Filter out warnings from AlphaPose
warnings.filterwarnings("ignore")
DEBUG = False


def window_worker(
        q: Queue, datalen: int, pose_data_queue: Queue, cfg: GeneralConfig
):
    """Read results from pose detection and pass them through per-frame preprocessing"""
    window = []
    length = cfg.samples_per_window
    interlace = cfg.interlace
    for i in tqdm(range(datalen), disable=False):
        data = pose_data_queue.get()
        if data is None:
            break
        fd = FrameData(i, len(data), data)
        preprocess_per_frame(fd, cfg)
        window.append(fd)
        if len(window) == length:
            q.put(window)
            if interlace > 0:
                window = window[-interlace:]
            else:
                window = []
    if len(window) > interlace:
        q.put(window)
    q.put(None)


def run_window_worker(
        datalen: int, pose_data_queue: Queue, cfg: GeneralConfig
):
    cfg = deepcopy(cfg)
    q = Queue(16)
    window_worker_thread = Thread(
        target=window_worker, args=(q, datalen, pose_data_queue, cfg)
    )
    window_worker_thread.start()
    return q, window_worker_thread


def calculate_frame_results(frame_results, to_index, frame_windows, window_results):
    """Calculates class id results from window"""
    cache = {}
    start_index = len(frame_results)
    for index in range(start_index, to_index):
        windows = tuple(frame_windows[index])
        if windows in cache:
            frame_results.append(cache[windows])
            continue
        windows_data = [window_results[window_id] for window_id in windows]
        windows_data = [x[2] for x in windows_data if x[2] is not None]
        if len(windows_data) == 0:
            frame_results.append(None)
            cache[windows] = None
            continue
        stacked_data = torch.stack([windows_data[0], windows_data[0]])
        stacked_data = torch.nn.functional.softmax(stacked_data, 1)
        summed_data = stacked_data.sum(0)
        result = summed_data.max(0)[1]  # index
        result = result.item()
        frame_results.append(result)
        cache[windows] = result

    return frame_results, start_index


def get_per_frame_results(window_results: dict, frame_windows: dict):
    """Returns a list of tensors where each index has information about data returned on the corresponding frame"""
    cache = {}
    per_frame_results = []
    for index in sorted(frame_windows.keys()):
        windows = tuple(frame_windows[index])
        if windows in cache:
            per_frame_results.append(cache[windows])
            continue

        windows_data = [window_results[window_id] for window_id in windows]
        windows_data = [x[2] for x in windows_data if x[2] is not None]
        if len(windows_data) == 0:
            per_frame_results.append(None)
            cache[windows] = None
            continue

        stacked_data = torch.stack([windows_data[0], windows_data[0]])
        stacked_data = torch.nn.functional.softmax(stacked_data, 1)
        summed_data = stacked_data.sum(0)
        per_frame_results.append(summed_data)
        cache[windows] = summed_data

    # Get any result to know the shape
    any_res = [x for x in per_frame_results if x is not None][0]
    for i in range(len(per_frame_results)):
        if per_frame_results[i] is None:
            per_frame_results[i] = torch.zeros_like(any_res)
    return per_frame_results


def calculate_total_result(window_results: dict) -> int:
    """Calculates mean class for all windows"""
    whole_data = torch.stack([res for _, _, res, _ in window_results.values() if res is not None])
    whole_data = torch.nn.functional.softmax(whole_data, 1)
    whole_results = torch.sum(whole_data, 0)
    class_id = torch.argmax(whole_results).item()
    return class_id


def single_file_classification(video_file: str, cfg: GeneralConfig, model_path: Union[str, None] = None,
                               save_file: str = None, result_method: str = "window", window_save_file: str = None):
    assert os.path.isfile(video_file)

    # Setup for pose estimation and windowing
    device = torch.device(cfg.device)
    frame_interval = calculate_interval(cfg.window_length, cfg.samples_per_window)
    det_loader, pose_data_queue, ap_threads = create_alphapose_workers(cfg, device, video_file, frame_interval)
    window_queue, window_thread = run_window_worker(det_loader.length, pose_data_queue, cfg)

    # Queue size printer
    if DEBUG:
        qsp_thread = run_qsp(
            [det_loader.image_queue, det_loader.det_queue, det_loader.pose_queue, pose_data_queue, window_queue],
            ["det", "post", "pose", "pose2", "window"])
    else:
        qsp_thread = None

    # Classification
    model, norm_func = create_model_norm(cfg, device, model_path)

    # Set up features
    required_transforms = solve_feature_transform_requirements(cfg.features)
    transforms = {key: TransformsDict[key](cfg.skeleton_type) for key in required_transforms}

    # Set up variables required for saving results
    start_time = time.time()
    window_count = 0
    unique_frames = []
    frame_windows_mapping = defaultdict(list)
    window_results = {}
    frame_results = []

    # Set up visualization
    visualizer = Visualizer(video_file, cfg.skeleton_type, frame_interval, True, save_file)
    vis_thread = visualizer.run_visualize()

    all_threads = {"prep": ap_threads[0], "det": ap_threads[1], "post": ap_threads[2], "pose": ap_threads[3],
                   "window": window_thread, "vis": vis_thread}
    if qsp_thread:
        all_threads['qsp'] = qsp_thread
    all_queues = {"prep": det_loader.image_queue, "det": det_loader.det_queue, "post": det_loader.pose_queue,
                  "pose": pose_data_queue, "window": window_queue, "vis": visualizer.queue}
    try:
        while True:
            frames: list[FrameData] = window_queue.get()
            if frames is None:
                break
            for frame in frames:
                frame_windows_mapping[frame.seqId].append(window_count)
                if frame.seqId not in [x.seqId for x in unique_frames]:
                    unique_frames.append(deepcopy(frame))
            start_frame, end_frame = frames[0].seqId, frames[-1].seqId

            data = SkeletonData("estimated", cfg.skeleton_type, None, video_file,
                                len(frames), deepcopy(frames), len(frames), det_loader.frameSize, frame_interval)

            if data.no_bodies():
                window_results[window_count] = (start_frame, end_frame, None, None)
                window_count += 1
                frame_results, start_index = calculate_frame_results(frame_results, frames[0].seqId,
                                                                     frame_windows_mapping,
                                                                     window_results)
                add_action_data(cfg.dataset, frame_results, start_index, frames[0].seqId, unique_frames, visualizer)
                continue

            # Fill frames if unfinished window
            if data.length != cfg.samples_per_window:
                fill_frames(data, cfg.samples_per_window)

            # Preprocess
            preprocess_data_rest(data, cfg.prep_config)
            points = data.to_matrix()
            if points is None:
                window_results[window_count] = (start_frame, end_frame, None, None)
                window_count += 1
                frame_results, start_index = calculate_frame_results(frame_results, frames[0].seqId,
                                                                     frame_windows_mapping,
                                                                     window_results)
                add_action_data(cfg.dataset, frame_results, start_index, frames[0].seqId, unique_frames, visualizer)
                continue

            features = prepare_features(cfg, norm_func, points, required_transforms, transforms)
            features = features.to(device, non_blocking=True)
            # Run through model
            with torch.no_grad():
                out = model(features)
                top5 = torch.topk(out, 5)[1]

            # Save results
            top5_tuple = tuple(x for x in top5[0].tolist())
            window_results[window_count] = (start_frame, end_frame, out[0].cpu(), top5_tuple)
            window_count += 1

            # calculate results to frames[0].seqId
            frame_results, start_index = calculate_frame_results(frame_results, frames[0].seqId, frame_windows_mapping,
                                                                 window_results)
            add_action_data(cfg.dataset, frame_results, start_index, frames[0].seqId, unique_frames, visualizer)
    except KeyboardInterrupt:
        # Stop visualization and close all threads by consuming from all_queues
        visualizer.stop()
        close_(all_queues)
        time.sleep(3)
        for name, thread in all_threads.items():
            if thread:
                print(name, thread.is_alive())
        return

    # Add all the data together and produce final mean result
    frame_results, start_index = calculate_frame_results(frame_results, len(frame_windows_mapping),
                                                         frame_windows_mapping, window_results)
    add_action_data(cfg.dataset, frame_results, start_index, len(frame_windows_mapping), unique_frames, visualizer)
    mean_class_id = calculate_total_result(window_results)

    # Calculate final time and fps here
    end_time = time.time()
    fps = len(unique_frames) / (end_time - start_time)

    print("FPS: ", fps)
    print("Average action in the video: ", adjusted_actions_maps[cfg.dataset][mean_class_id])

    # Save results using visualizer class
    if visualizer.save_file:
        total_data = SkeletonData("estimated", cfg.skeleton_type, None, video_file,
                                  len(unique_frames), unique_frames, len(unique_frames), det_loader.frameSize,
                                  frame_interval)
        for frame in total_data.frames:
            if result_method == "mean":
                frame.text = adjusted_actions_maps[cfg.dataset][mean_class_id]
            visualizer.put(frame)

        skeleton_filename = swap_extension(save_file, "apskel.pkl")
        total_data.save(skeleton_filename)
    visualizer.put(None)

    # Save per window results
    if window_save_file:
        whole_data = get_per_frame_results(window_results, frame_windows_mapping)
        whole_data = np.stack(whole_data)
        with open(window_save_file, "wb") as f:
            np.save(f, whole_data)

    # Ensure queues empty
    for name, q in all_queues.items():
        if name == "vis":
            continue
        while not q.empty():
            q.get()

    # Wait for visualizer
    vis_thread.join()
    return fps


def close_(queues):
    def empty_queue(q):
        while not q.empty():
            q.get()

    for name, q in queues.items():
        empty_queue(q)
        if name == "prep":
            q.put([None] * 4)
        if name == "det":
            q.put([None] * 5)
        if name == "post":
            q.put([None] * 5)
        else:
            q.put(None)


def add_action_data(dataset, frame_results, start_index, end_index, unique_frames, visualizer):
    for i in range(start_index, end_index):
        action_name = adjusted_actions_maps[dataset].get(frame_results[i], "None")
        unique_frames[i].text = action_name
        if not visualizer.save_file:
            visualizer.put(unique_frames[i])


def prepare_features(cfg, norm_func, points, required_transforms, transforms):
    points = norm_func(points)
    feature_dictionary = {"joints": points}
    for feature in required_transforms:
        transforms[feature](feature_dictionary)
    # Transforms to correct input size
    if isinstance(cfg.features[0], str):
        features = transform_to_stgcn_input(feature_dictionary, cfg.features)
    elif isinstance(cfg.features[0], list):
        features = transform_to_tpgcn_input(feature_dictionary, cfg.features, cfg.symmetry_processing, cfg.copy_pad)
    else:
        raise KeyError("features are wrong")
    features = torch.from_numpy(features).float().unsqueeze(0)
    return features


def create_model_norm(cfg: GeneralConfig, device: torch.device, model_path: str = None):
    norm_func = create_norm_func(cfg.normalization_type)
    if not model_path:
        model_path = cfg.best_model_path
    state_dict = load_model(model_path, None, None, None, device)
    num_classes = state_dict['net'][[x for x in state_dict['net'].keys()][-1]].shape[0]
    channels = calculate_channels(cfg.features, 2)
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(num_classes, channels, cfg.skeleton_type)
    elif cfg.model_type == "2pgcn":
        model = create_tpgcn(num_classes, len(cfg.features), channels,
                             cfg.skeleton_type, cfg.labeling, cfg.graph_type)
    else:
        raise KeyError(f"model type {cfg.model_type} not supported")
    model.to(device)
    # Load state dict
    model.load_state_dict(state_dict['net'])
    model.eval()
    if norm_state_dict := state_dict.get("normalization"):
        setup_norm_func(norm_func, state_dict=norm_state_dict)
    else:
        setup_norm_func(norm_func, train_file=cfg.train_config.train_file)
    return model, norm_func


def create_alphapose_workers(cfg: GeneralConfig, device: torch.device, filename: str,
                             frame_interval: int) -> tuple[DetectionLoader, Queue, list[Thread, ...]]:
    pose_cfg = cfg.pose_config
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device)
    # Detection
    detector = init_detector(opts, det_cfg)
    detector.load_model()
    det_loader = DetectionLoader(
        filename, detector, ap_cfg, opts, "video",
        pose_cfg.detector_batch_size, pose_cfg.detector_queue_size, frame_interval
    )
    det_threads = det_loader.start()
    # Pose estimation
    pose_model = init_pose_model(device, ap_cfg, ap_cfg.weights_file)
    pose_data_queue, pose_thread = run_pose_worker(
        pose_model, det_loader, opts, pose_cfg.estimation_batch_size, pose_cfg.estimation_queue_size
    )
    all_threads = det_threads + [pose_thread]
    return det_loader, pose_data_queue, all_threads


def handle_classify(args: Namespace):
    cfg = GeneralConfig.from_yaml_file(args.config)
    if not os.path.isfile(args.video_file):
        print(f"{args.video_file} does not exist")
        return False
    if args.model and not os.path.isfile(args.model):
        print(f"{args.model} does not exist")
        return False
    if args.save_file and not os.path.isdir(os.path.split(args.save_file)[0]):
        print(f"Folder for {args.save_file} does not exist")
        return False
    if args.window_save_file and not os.path.isdir(os.path.split(args.window_save_file)[0]):
        print(f"Folder for {args.window_save_file} does not exist")
        return False
    if args.window_length is not None:
        if args.window_length < 0:
            print(
                f"Incorrect window length value.")
            return False
        cfg.window_length = args.window_length
    if args.samples is not None:
        if args.samples < 0:
            print(
                f"Incorrect samples value.")
            return False
        cfg.samples_per_window = args.samples
    if args.interlace is not None:
        if args.interlace < 0 or args.interlace >= cfg.samples_per_window:
            print(
                f"Incorrect interlace value. Interlace should be an integer above 0 and below samples per window count")
            return False
        cfg.interlace = args.interlace

    single_file_classification(args.video_file, cfg, args.model, args.save_file, args.method, args.window_save_file)
