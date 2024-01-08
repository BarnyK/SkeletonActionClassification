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
from models import create_stgcnpp, create_tpgcn
from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.normalizations import create_norm_func, setup_norm_func
from preprocessing.tracking import pose_track
from procedures.config import GeneralConfig
from procedures.training import load_model
from procedures.utils.prep import preprocess_per_frame, preprocess_data_rest
from shared.dataset_statics import adjusted_actions_maps
from shared.helpers import calculate_interval, run_qsp, fill_frames
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import Visualizer


def window_worker(
        q: Queue, datalen: int, pose_data_queue: Queue, cfg: GeneralConfig
):
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


def calculate_frame_results(frame_results, to_index, frame_windows, window_results):
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


def single_file_classification(filename, cfg: GeneralConfig, model_path: Union[str, None] = None,
                               save_file: str = None):
    assert os.path.isfile(filename)

    # Setup
    device = torch.device(cfg.device)
    frame_interval = calculate_interval(cfg.window_length, cfg.samples_per_window)

    det_loader, pose_data_queue, ap_threads = create_alphapose_workers(cfg, device, filename, frame_interval)

    window_queue, window_thread = run_window_worker(det_loader.length, pose_data_queue, cfg)

    qsp_thread = run_qsp([det_loader.image_queue, det_loader.det_queue, det_loader.pose_queue, window_queue],
                         ["det", "post", "pose", "window"])

    # Classification
    model, norm_func = create_model_norm(cfg, device, model_path)

    # Set up features
    required_transforms = solve_feature_transform_requirements(cfg.features)
    transforms = {key: TransformsDict[key](cfg.skeleton_type) for key in required_transforms}

    start_time = time.time()
    window_count = 0
    unique_frames = []
    frame_windows_mapping = defaultdict(list)
    window_results = {}
    frame_results = []

    visualizer = Visualizer(filename, cfg.skeleton_type, frame_interval, True, save_file)
    vis_thread = visualizer.run_visualize()

    all_threads = {"prep": ap_threads[0], "det": ap_threads[1], "post": ap_threads[2], "window": window_thread,
                   "qsp": qsp_thread, "vis": vis_thread}
    all_queues = {"prep": det_loader.image_queue, "det": det_loader.det_queue, "post": det_loader.pose_queue,
                  "window": window_queue, "vis": visualizer.queue}
    try:
        print("Interruptible")
        while True:
            frames: list[FrameData] = window_queue.get()
            if frames is None:
                break
            for frame in frames:
                frame_windows_mapping[frame.seqId].append(window_count)
                if frame.seqId not in [x.seqId for x in unique_frames]:
                    unique_frames.append(deepcopy(frame))
            start_frame, end_frame = frames[0].seqId, frames[-1].seqId

            data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
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
        visualizer.stop()
        close_(all_queues)
        time.sleep(3)
        for name, thread in all_threads.items():
            if thread:
                print(name, thread.is_alive())
        return

    frame_results, start_index = calculate_frame_results(frame_results, len(frame_windows_mapping),
                                                         frame_windows_mapping, window_results)
    add_action_data(cfg.dataset, frame_results, start_index, len(frame_windows_mapping), unique_frames, visualizer)

    end_time = time.time()
    fps = len(unique_frames) / (end_time - start_time)
    print(f"fps:        {fps:.4}")
    print(f"Total time: {end_time - start_time:.4}")

    whole_data = np.stack([res for _, _, res, _ in window_results.values() if res is not None])
    whole_results = np.sum(whole_data, 0)
    res = np.argmax(whole_results)
    print(adjusted_actions_maps[cfg.dataset][res])
    # for w in window_results:
    #     print(window_results[w])

    # #
    if visualizer.save_file:
        total_data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
                                  len(unique_frames), unique_frames, len(unique_frames), det_loader.frameSize,
                                  frame_interval)
        pose_track(total_data.frames)
        keypoint_fill(total_data, cfg.prep_config.keypoint_fill_type)
        for frame in total_data.frames:
            visualizer.put(frame)
    visualizer.put(None)
    # for frame, action_id in zip(total_data.frames, out):
    #     frame.text = action_map[frame_results[frame.seqId]]
    # fps = 30 * cfg.samples_per_window / cfg.window_length
    # # visualize(total_data, total_data.video_file, int(1000 / fps), print_frame_text=False, skip_frames=True,
    # #           save_file="/home/barny/naaaaaaah.mp4", draw_bbox=True)

    for name, q in all_queues.items():
        if name == "vis":
            continue
        while not q.empty():
            q.get()
    print(len(visualizer.uniques))
    print(vis_thread.join())
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

    single_file_classification(args.video_file, cfg)


def prepend_extension(filename, prefix):
    root, filename = os.path.split(filename)
    filename, extension = os.path.splitext(filename)
    return os.path.join(root, f"{filename}.{prefix}.{extension}", )


if __name__ == "__main__":
    config_files = ["/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml",
                    "/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml"]
    params = [
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 60, 0),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 30, 0),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 30, 10),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 30, 15),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 30, 20),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 60, 15, 0),
        ("/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml", 30, 30, 0)
    ]

    params = [
        ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_bo_ntu120_xset_0/config.yaml",64,32,30),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 60, 0),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 60, 0),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 30, 0),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 30, 10),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 30, 15),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 30, 20),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 60, 15, 0),
        # ("/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo_ntu_xsub_0/config.yaml", 30, 30, 0)
    ]
    res = []
    for i, (filename, window_length, samples, interlace) in enumerate(params):
        fpses = []
        times = []
        config = GeneralConfig.from_yaml_file(filename)
        config.interlace = interlace
        config.window_length = window_length
        config.samples_per_window = samples
        #config.pose_config.detector_queue_size = 20
        for i in range(1):
            print(config.name, config.window_length, config.samples_per_window, config.interlace, i)
            st = time.time()
            input_file = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/ut-interaction_set1/seq3.avi"
            input_file = "/media/barny/SSD4/MasterThesis/Data/ut-interaction/seq1.cut.avi"
            input_file = "/home/barny/dance_practice.cut.avi"
            input_file = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb_120/S021C003P055R001A101_rgb.avi"
            out_filename = prepend_extension(prepend_extension(input_file, config.name),
                                             f"{config.window_length}.{config.samples_per_window}.{config.interlace}")
            out = os.path.join("/media/barny/SSD4/MasterThesis/result_videos/", os.path.split(out_filename)[-1])
            x = single_file_classification(
                input_file,
                config, None, out)
            et = time.time()
            times.append(et - st)
            fpses.append(x)

        print(f"Mean exec time: {np.mean(times):.5}")  # 38.792
        print(f"Mean fps: {np.mean(fpses):.5}")  # 2.1821 # 2.1684
        res.append(
            f"{config.name} {config.window_length} {config.samples_per_window} {config.interlace} {np.mean(fpses):.5}")
    for r in res:
        print(r)

    """
    stgcn_jo_ntu_xsub_0 60 60 0 62.385
stgcn_jo_ntu_xsub_0 60 30 0 93.336
stgcn_jo_ntu_xsub_0 60 30 10 90.429
stgcn_jo_ntu_xsub_0 60 30 15 88.298
stgcn_jo_ntu_xsub_0 60 30 20 88.824
stgcn_jo_ntu_xsub_0 60 15 0 130.79
    """

    """
    stgcn_jo_ntu_xsub_0 60 60 0 58.231
    stgcn_jo_ntu_xsub_0 60 30 0 70.839
    stgcn_jo_ntu_xsub_0 60 30 10 73.537
    stgcn_jo_ntu_xsub_0 60 30 15 70.035
    stgcn_jo_ntu_xsub_0 60 30 20 67.809
    stgcn_jo_ntu_xsub_0 60 15 0 98.51
    stgcn_jo_ntu_xsub_0 30 30 0 59.565
    """
"""1080p, 20
stgcn_jo_ntu_xsub_0 60 60 0 45.397
stgcn_jo_ntu_xsub_0 60 30 0 56.51
stgcn_jo_ntu_xsub_0 60 30 10 54.772
stgcn_jo_ntu_xsub_0 60 30 15 52.366
stgcn_jo_ntu_xsub_0 60 30 20 53.571
stgcn_jo_ntu_xsub_0 60 15 0 55.516
stgcn_jo_ntu_xsub_0 30 30 0 44.417
"""