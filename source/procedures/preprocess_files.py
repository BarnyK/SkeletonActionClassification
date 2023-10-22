import multiprocessing
import os
import pickle
from dataclasses import dataclass
from functools import partial

from tqdm import tqdm

from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion
from shared.structs import SkeletonData


@dataclass
class PreprocessConfig:
    box_conf_threshold: float = 0.7
    max_pose_conf_threshold: float = 0.7
    pose_tracking_threshold: int = 90
    keypoint_fill_type: str = "interpolation"
    use_nms: bool = True


def _preprocess_data(data: SkeletonData, config: PreprocessConfig):
    skeleton_filters.remove_bodies_by_box_confidence(data, 0.7, 0.9, 0.9)
    skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.55)
    nms(data, True)
    pose_track(data.frames, threshold=90,width_ratio=1.8, height_ratio=0.55)
    select_tracks_by_motion(data, 2)
    keypoint_fill(data, "interpolation")
    return data


def preprocess_file(in_file: str, config: PreprocessConfig):
    data = SkeletonData.load(in_file)
    _preprocess_data(data, config)
    try:
        mat = data.to_matrix()
    except ValueError:
        tqdm.write(in_file)
        return in_file, None
    # Should be of shape, M, T, V, C
    return data.dataset_info.__dict__, mat


def preprocess_files(input_path: str, output_path: str, config: PreprocessConfig):
    # input_path should be a folder containing files created using pose_estimation
    # The files will be read and transformed into a single data file
    # Split can be done if a strategy is provided
    files = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    pool = multiprocessing.Pool(processes=12)
    results = list(tqdm(pool.imap(partial(preprocess_file, config=config), files), total=len(files)))
    pool.close()
    pool.join()

    data = {
        "labels": [x for x, y in results if y is not None],
        "points": [y for x, y in results if y is not None]
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                     "/media/barny/SSD4/MasterThesis/Data/ntu_coco.combined", PreprocessConfig())
