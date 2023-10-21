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


def preprocess_file2(in_file: str, config: PreprocessConfig, ):
    data = SkeletonData.load(in_file)
    skeleton_filters.remove_bodies_by_box_confidence(data, 0.7, 0.3, 0.2)
    if data.no_bodies():
        return "box_conf", in_file
    skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
    if data.no_bodies():
        return "pose_conf", in_file
    nms(data, True)
    pose_track(data.frames, threshold=90)
    select_tracks_by_motion(data, 2)
    if data.no_bodies():
        return "motion", in_file
    # visualize(data, data.video_file, 1000 // 30)
    # keypoint_fill(data, "interpolation")
    return None


def preprocess_files2(inpath: str, outpath: str, config: PreprocessConfig):
    # inpath should be a folder containing files created using pose_estimation
    # The files will be read and transformed into a single data file
    # Split can be done if a strategy is provided
    files = [os.path.join(inpath, f) for f in os.listdir(inpath)]
    # files = files[2200:2300]
    # files = files[2245:2246]
    ###
    # files = files[:100]
    # for file in tqdm(files):
    #     preprocess_file(file, config)

    # with open("/media/barny/SSD4/MasterThesis/Data/bad.files","r") as f:
    #     data = f.read().split("\n")
    #     data = [x.split("\t") for x in data]
    #
    # data = [x for x in data if x[0] != "motion"]
    # files = [y for x,y in data]
    #
    # results = []
    # for file in tqdm(files):
    #     results.append(preprocess_file2(file, config))

    pool = multiprocessing.Pool(processes=12)
    results = list(tqdm(pool.imap(partial(preprocess_file, config=config), files), total=len(files)))
    pool.close()
    pool.join()

    data = {
        "labels": [x for x, y in results if y is not None],
        "points": [y for x, y in results if y is not None]
    }

    with open(outpath, "wb") as f:
        pickle.dump(data, f)


def _preprocess_data(data: SkeletonData, config: PreprocessConfig ):
    skeleton_filters.remove_bodies_by_box_confidence(data, 0.7, 0.9, 0.9)
    skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
    nms(data, True)
    pose_track(data.frames, threshold=90)
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


def preprocess_files(inpath: str, outpath: str, config: PreprocessConfig):
    # inpath should be a folder containing files created using pose_estimation
    # The files will be read and transformed into a single data file
    # Split can be done if a strategy is provided
    files = [os.path.join(inpath, f) for f in os.listdir(inpath)]

    pool = multiprocessing.Pool(processes=12)
    results = list(tqdm(pool.imap(partial(preprocess_file, config=config), files), total=len(files)))
    pool.close()
    pool.join()

    data = {
        "labels": [x for x, y in results if y is not None],
        "points": [y for x, y in results if y is not None]
    }

    with open(outpath, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_files("/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_coco",
                     "/media/barny/SSD4/MasterThesis/Data/ntu_coco.combined", PreprocessConfig())
