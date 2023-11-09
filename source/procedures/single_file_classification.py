import os
import time
from copy import deepcopy
from queue import Queue
from threading import Thread

import torch
from tqdm import tqdm

from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from procedures.config import GeneralConfig
from procedures.preprocess_files import _preprocess_data_ap
from shared.helpers import calculate_interval
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


def single_file_classification(filename, cfg: GeneralConfig):
    assert os.path.isfile(filename)

    pose_cfg = cfg.pose_config

    device = torch.device(cfg.device)
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device)

    detector = init_detector(opts, det_cfg)
    detector.load_model()

    pose_model = init_pose_model(device, ap_cfg, ap_cfg.weights_file)

    frame_interval = calculate_interval(cfg.window_length, cfg.samples_per_window)

    det_loader = DetectionLoader(
        filename, detector, ap_cfg, opts, "video",
        pose_cfg.detector_batch_size, pose_cfg.detector_queue_size, frame_interval
    )
    det_loader.start()

    pose_data_queue = run_pose_worker(
        pose_model, det_loader, opts, pose_cfg.estimation_batch_size, pose_cfg.estimation_queue_size
    )
    time.sleep(1)

    window_queue = run_window_worker(det_loader.length, pose_data_queue, cfg.samples_per_window, cfg.interlace)
    st = time.time()
    wc = 0
    while True:
        frames = window_queue.get()
        if frames is None:
            break
        wc += 1
        data = SkeletonData("estimated", cfg.skeleton_type, None, filename,
                            len(frames), deepcopy(frames), len(frames), det_loader.frameSize, frame_interval)

        if data.length != cfg.samples_per_window:
            fill_frames(data, cfg.samples_per_window)
        _preprocess_data_ap(data, cfg.prep_config)
        # Frame preproc
        # preproc part which is per frame
        # deepcopy
        # tracking
        # selection
        # fill
        # mat form
        # features
    et = time.time()
    print(wc / (et-st))

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
    config = GeneralConfig()

    single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.2.avi", config)
    single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.2.avi", config)
    single_file_classification("/media/barny/SSD4/MasterThesis/Data/concatenated.2.avi", config)
