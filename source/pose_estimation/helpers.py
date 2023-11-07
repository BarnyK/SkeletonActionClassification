import torch
from alphapose.models import builder
from detector.yolo_api import YOLODetector
from easydict import EasyDict

from shared.helpers import update_config


def init_detector(opts: EasyDict, detector_cfg: EasyDict):
    detector = YOLODetector(detector_cfg, opts)
    return detector


def init_pose_model(device: torch.device, general_config: EasyDict, weights_file: str):
    pose_model = builder.build_sppe(
        general_config.MODEL, preset_cfg=general_config.DATA_PRESET
    )
    pose_model.load_state_dict(torch.load(weights_file, map_location=device))
    pose_model.to(device, non_blocking=True)
    pose_model.eval()
    return pose_model


def read_ap_configs(
        skeleton_type: str = "coco17",
        device: torch.device = torch.device("cuda:0"),
        det_config: str = "./configs/detector/yolov3-spp.cfg",
        det_weight: str = "./weights/detector/yolov3-spp.weights",
        est_config: str = "./configs/alphapose/256x192_res50_lr1e-3_1x.yaml",
        est_weight: str = "./weights/alphapose/fast_res50_256x192.pth",
):
    opts = EasyDict()
    opts.device = device
    opts.sp = True  # single process
    opts.tracking = False
    opts.gpus = "0"
    opts.gpus = (
        [int(i) for i in opts.gpus.split(",")]
        if torch.cuda.device_count() >= 1
        else [-1]
    )

    detector_cfg = EasyDict()
    detector_cfg.CONFIG = det_config
    detector_cfg.WEIGHTS = det_weight
    detector_cfg.INP_DIM = 608
    detector_cfg.NMS_THRES = 0.6
    detector_cfg.CONFIDENCE = 0.1
    detector_cfg.NUM_CLASSES = 80

    if skeleton_type in ["coco17", "halpe"]:
        general_config = update_config(est_config)
        general_config.weights_file = est_weight
    else:
        raise ValueError(f"invalid skeleton type {skeleton_type}")

    return general_config, detector_cfg, opts
