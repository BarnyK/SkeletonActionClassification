import os
from argparse import Namespace

import torch
from tqdm import tqdm

from pose_estimation import DetectionLoader, init_detector, init_pose_model, read_ap_configs, run_pose_worker
from procedures.config import GeneralConfig
from shared.dataset_info import DatasetInfo
from shared.helpers import calculate_interval, swap_extension
from shared.structs import SkeletonData, FrameData
from shared.visualize_skeleton_file import visualize


def single_file_pose(filename, cfg: GeneralConfig, save_file: str = None):
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
              draw_bbox=True, draw_confidences=False, draw_frame_number=False, skip_frames=True, save_file=save_file)
    if save_file is not None:
        skeleton_filename = swap_extension(save_file, "apskel.pkl")
        data.save(skeleton_filename)
    # DEBUG
    # skeleton_filters.remove_bodies_by_box_confidence(data, cfg.prep_config.box_conf_threshold)
    # skeleton_filters.remove_by_max_possible_pose_confidence(data, cfg.prep_config.max_pose_conf_threshold)
    # visualize(data, data.video_file, wait_key=1000 // 30,
    #           draw_bbox=True, draw_confidences=False, draw_frame_number=False,
    #           save_file=prepend_filename(save_file, "filtered_"))
    # pose_track(data.frames,
    #            threshold=cfg.prep_config.pose_tracking_threshold,
    #            width_ratio=cfg.prep_config.pose_tracking_width_ratio,
    #            height_ratio=cfg.prep_config.pose_tracking_height_ratio)
    # visualize(data, data.video_file, wait_key=1000 // 30,
    #           draw_bbox=False, draw_confidences=True, draw_frame_number=True,
    #           save_file=prepend_filename(save_file, "tracked_"))
    # keypoint_fill(data, "knn")
    # visualize(data, data.video_file, wait_key=1000 // 30,
    #           draw_bbox=False, draw_confidences=True, draw_frame_number=True,
    #           save_file=prepend_filename(save_file, "filled_"))


def handle_pose_estimation(args: Namespace):
    cfg = GeneralConfig.from_yaml_file(args.config)
    if not os.path.isfile(args.video_file):
        print(f"{args.video_file} does not exist")
        return False
    if args.save_file and not os.path.isdir(os.path.split(args.save_file)[0]):
        print(f"Folder for {args.save_file} does not exist")
        return False
    cfg = GeneralConfig.from_yaml_file(args.config)
    single_file_pose(args.video_file, cfg, args.save_file)


if __name__ == "__main__":
    config = GeneralConfig.from_yaml_file(
        "/media/barny/SSD4/MasterThesis/Data/logs/random/2pgcn_mutual120xset_base/config.yaml"
    )
    config.samples_per_window = config.window_length
    input_file = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S009C003P008R002A060_rgb.avi"
    input_file = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S003C002P007R002A060_rgb.avi"
    input_file = "/media/barny/SSD4/MasterThesis/Data/nturgb+d_rgb/S009C003P008R002A060_rgb.avi"
    out = os.path.join("/media/barny/SSD4/MasterThesis/result_videos/boxes/", os.path.split(input_file)[-1])
    single_file_pose(input_file, config, out)
