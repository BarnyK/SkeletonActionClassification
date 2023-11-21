import os

import torch
from tqdm import tqdm

from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from procedures.config import GeneralConfig
from shared.dataset_info import DatasetInfo
from shared.helpers import calculate_interval
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import visualize


def single_file_pose(filename, cfg: GeneralConfig):
    assert os.path.isfile(filename)
    pose_cfg = cfg.pose_config

    device = torch.device(cfg.device)
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device, pose_cfg.detector_cfg, pose_cfg.detector_weights,
                                            pose_cfg.estimation_cfg, pose_cfg.estimation_weights)

    detector = init_detector(opts, det_cfg)
    detector.load_model()

    pose_model = init_pose_model(device, ap_cfg, ap_cfg.weights_file)

    frame_interval = calculate_interval(cfg.window_length, cfg.samples_per_window)

    det_loader = DetectionLoader(
        filename, detector, ap_cfg, opts, "video",
        pose_cfg.detector_batch_size, pose_cfg.detector_queue_size, frame_interval
    )
    det_loader.start()

    pose_data_queue, _ = run_pose_worker(
        pose_model, det_loader, opts, pose_cfg.estimation_batch_size, pose_cfg.estimation_queue_size
    )

    tq = tqdm(range(det_loader.datalen), dynamic_ncols=True, disable=False)
    frames = []
    for i in tq:
        data = pose_data_queue.get()
        if data is None:
            break
        frame_data = FrameData(i, len(data), data)
        frames.append(frame_data)

    data = SkeletonData(
        "estimated",
        cfg.skeleton_type,
        DatasetInfo(),
        filename,
        det_loader.datalen,
        frames,
        len(frames),
        det_loader.frameSize,
        frame_interval
    )
    visualize(data, data.video_file, wait_key=1000 // 30,
              draw_bbox=True, draw_confidences=True, draw_frame_number=True)


if __name__ == "__main__":
    config = GeneralConfig()
    single_file_pose("/media/barny/SSD4/MasterThesis/Data/concatenated.5.avi", config)
