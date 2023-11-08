from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Union

from dataclass_wizard import YAMLWizard


@dataclass
class EvalConfig(YAMLWizard, key_transform='SNAKE'):
    output_path: str = ""

    test_file: str = ""
    test_batch_size: int = 128
    test_clips_count: int = 8

    eval_interval: int = 1
    eval_last_n: int = 10


@dataclass
class TrainingConfig(YAMLWizard, key_transform='SNAKE'):
    epochs: int = 80

    train_file: str = ""
    train_batch_size: int = 64

    sgd_lr: float = 0.1
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 0.0002
    sgd_nesterov: bool = True

    cosine_shed_eta_min: float = 0.0001

    use_scale_augment: bool = False
    scale_value: float = 0.2


@dataclass
class PreprocessConfig(YAMLWizard, key_transform='SNAKE'):
    processes: int = 0
    missing_file: str = ""
    split_strategy: Union[str, Iterable[str]] = ("ntu_xsub",)

    use_box_conf: bool = True
    box_conf_threshold: float = 0.7
    box_conf_max_total: float = 0.9
    box_conf_max_frames: float = 0.9

    use_max_pose_conf: bool = True
    max_pose_conf_threshold: float = 0.55

    use_nms: bool = True

    use_tracking: bool = True
    pose_tracking_threshold: int = 90
    pose_tracking_width_ratio: float = 1.8
    pose_tracking_height_ratio: float = 0.55

    use_motion_selection: bool = True

    use_size_selection: bool = False
    use_confidence_selection: bool = False
    use_order_selection: bool = False
    max_body_count: int = 2
    keypoint_fill_type: str = "interpolation"

    transform_to_combined: bool = False
    alphapose_skeletons: bool = True
    remove_missing_from_file: bool = False


def ntu_preprocess_cfg():
    return PreprocessConfig(
        use_box_conf=False,
        use_max_pose_conf=False,
        use_nms=True,
        use_tracking=False,
        pose_tracking_threshold=90,
        pose_tracking_width_ratio=1.9,
        pose_tracking_height_ratio=0.55,
        use_motion_selection=True,
        use_size_selection=True,
        use_order_selection=False,
        max_body_count=2,
        keypoint_fill_type="interpolation",
        transform_to_combined=False,
        alphapose_skeletons=False,
        remove_missing_from_file=True,
    )


@dataclass
class PoseEstimationConfig(YAMLWizard, key_transform='SNAKE'):
    dataset_name = "ntu"
    detector_cfg: str = "./configs/detector/yolov3-spp.cfg"
    detector_weights: str = "./weights/detector/yolov3-spp.weights"

    estimation_cfg = "./configs/alphapose/256x192_res50_lr1e-3_1x.yaml"
    estimation_weights: str = "./weights/alphapose/fast_res50_256x192.pth"

    detector_batch_size: int = 5
    detector_queue_size: int = 256

    estimation_batch_size: int = 8
    estimation_queue_size: int = 64

    input_folder: str = "/media/barny/SSD4/MasterThesis/Data/ntu_sample/"
    output_folder: str = "/media/barny/SSD4/MasterThesis/Data/ntu_sample/"


@dataclass
class GeneralConfig(YAMLWizard, key_transform='SNAKE'):
    name: str = "default"
    skeleton_type: str = "coco17"
    model_type: str = "stgcnpp"  #: TODO: 2P-GCN
    device: str = "cuda:0"  #: could be "cuda:0" or "cpu"
    features: list[str] = field(default_factory=lambda: ['joints'])
    window_length: int = 64
    samples_per_window: int = 32
    interlace: int = 16
    symmetry_processing: bool = False  #: Only works with 2p-GCN
    normalization_type: str = "spine_align"

    train_config: TrainingConfig = TrainingConfig()
    eval_config: EvalConfig = EvalConfig()
    pose_config: PoseEstimationConfig = PoseEstimationConfig()
    prep_config: PreprocessConfig = PreprocessConfig()
    log_folder: str = "logs"

    def best_model_path(self):
        return os.path.join(self.log_folder, self.name, "best.pth")


if __name__ == "__main__":
    x = GeneralConfig("test")
    x = x.from_yaml_file("../configs/general/default.yaml")
    x.name = "XDDD"
    print(x.to_yaml())
