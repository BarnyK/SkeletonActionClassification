import os

import torch
from tqdm import tqdm

from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from procedures.config import PoseEstimationConfig
from shared.dataset_info import DatasetInfo
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import visualize


def single_file_pose(filename, cfg: PoseEstimationConfig):
    assert os.path.isfile(filename)

    device = torch.device(cfg.device)
    ap_cfg, det_cfg, opts = read_ap_configs(cfg.skeleton_type, device)

    detector = init_detector(opts, det_cfg)
    detector.load_model()

    pose_model = init_pose_model(device, ap_cfg, ap_cfg.weights_file)

    det_loader = DetectionLoader(
        filename, detector, ap_cfg, opts, "video",
        cfg.detector_batch_size, cfg.detector_queue_size, cfg.frame_interval
    )
    det_loader.start()

    pose_data_queue = run_pose_worker(
        pose_model, det_loader, opts, cfg.estimation_batch_size, cfg.estimation_queue_size
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
        cfg.frame_interval
    )
    visualize(data, data.video_file, wait_key=1000//30, draw_bbox=True, draw_confidences=True, draw_frame_number=True)


if __name__ == "__main__":
    config = PoseEstimationConfig()
    config.frame_interval = 2
    single_file_pose("/media/barny/SSD4/MasterThesis/Data/concatenated.5.avi", config)
