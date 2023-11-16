from __future__ import annotations
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Union, Type, Optional, List, AnyStr, TextIO, BinaryIO

from dataclass_wizard import YAMLWizard
from dataclass_wizard.type_def import T, Encoder, Decoder

import shared.datasets


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
    keep_best_n: int = 5


@dataclass
class PreprocessConfig(YAMLWizard, key_transform='SNAKE'):
    processes: int = 12
    missing_file: str = ""
    split_strategy: list[str] = field(default_factory=lambda: shared.datasets.all_splits)

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
    use_3d_points: bool = False


def ntu_preprocess_cfg() -> PreprocessConfig:
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


@dataclass
class GeneralConfig(YAMLWizard, key_transform='SNAKE'):
    @classmethod
    def from_yaml(cls: Type[T], string_or_stream: Union[AnyStr, TextIO, BinaryIO], *, decoder: Optional[Decoder] = None,
                  **decoder_kwargs) -> Union[T, List[T]]:
        tmp = super().from_yaml(string_or_stream, decoder=decoder, **decoder_kwargs)
        if not isinstance(tmp, list):
            for i, feat in enumerate(tmp.features):
                x = [x.strip() for x in feat.split(",")]
                tmp.features[i] = x
        return tmp

    def to_yaml(self: T, *, encoder: Optional[Encoder] = None, **encoder_kwargs) -> AnyStr:
        tmp = deepcopy(self)
        if tmp.features:
            for i, _ in enumerate(tmp.features):
                if isinstance(tmp.features[i], list):
                    tmp.features[i] = ",".join(tmp.features[i])

        return tmp.to_yaml_(encoder=encoder, **encoder_kwargs)

    def to_yaml_(self: T, *, encoder: Optional[Encoder] = None, **encoder_kwargs) -> AnyStr:
        return super().to_yaml(encoder=encoder, **encoder_kwargs)

    name: str = "default"
    skeleton_type: str = "coco17"
    model_type: str = "stgcnpp"
    device: str = "cuda:0"  #: could be "cuda:0" or "cpu"
    features: Union[list[str], list[list[str]]] = field(default_factory=lambda: ['joints'])
    window_length: int = 64
    samples_per_window: int = 32
    interlace: int = 16
    symmetry_processing: bool = False  #: Only works with 2p-GCN
    labeling: str = "distance"  #: Only works with 2P-GCN
    graph_type: str = "mutual"  #: Only works with 2P-GCN
    normalization_type: str = "spine_align"

    train_config: TrainingConfig = TrainingConfig()
    eval_config: EvalConfig = EvalConfig()
    pose_config: PoseEstimationConfig = PoseEstimationConfig()
    prep_config: PreprocessConfig = PreprocessConfig()
    log_folder: str = "logs"
    dataset: str = "ntu"

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.log_folder, self.name, "best.pth")

    @staticmethod
    def compare(instance1: GeneralConfig, instance2: GeneralConfig) -> list[str]:
        diffs = []
        class_fields = fields(instance1)
        for field_ in class_fields:
            field_name = field_.name
            value1 = getattr(instance1, field_name)
            value2 = getattr(instance2, field_name)

            if value1 != value2:
                if is_dataclass(value1):
                    diffs_nested = GeneralConfig.compare(value1, value2)
                    diffs_nested = [f"{field_name}.{diff}" for diff in diffs_nested]
                    diffs.extend(diffs_nested)
                else:
                    diffs.append(f"{field_name}: {value1} != {value2}")

        return diffs


if __name__ == "__main__":
    x = GeneralConfig("test")
    x = x.from_yaml_file("../configs/general/default.yaml")
    # x.name = "XDDD"
    print(x.to_yaml())
    x.to_yaml_file("../configs/general/default_ap_xview.yaml")
