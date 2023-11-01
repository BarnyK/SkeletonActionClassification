from __future__ import annotations

import pickle

import numpy as np


def screen_normalization(mat: np.ndarray, screen_size: tuple[int, int], **kwargs) -> np.ndarray:
    # Normalize points by shifting points by half the size of screen and diving by it
    w, h = screen_size
    result = np.zeros_like(mat)
    result[..., 0] = (mat[..., 0] - w / 2) / (w / 2)
    result[..., 1] = (mat[..., 1] - h / 2) / (h / 2)
    return result


def relative_normalization(mat: np.ndarray, **kwargs) -> np.ndarray:
    # Normalizes points using minimal and maximal values of points
    result = mat.copy()
    x_max = np.max(mat[..., 0])
    x_min = np.min(mat[..., 0])
    y_max = np.max(mat[..., 1])
    y_min = np.min(mat[..., 1])

    w = x_max - x_min
    h = y_max - y_min

    result[..., 0] = (mat[..., 0] - w / 2) / w
    result[..., 1] = (mat[..., 1] - h / 2) / h
    return result


def spine_size_coco(mat: np.ndarray) -> np.ndarray:
    # Calculates the size of spine given a matrix with skeleton
    # Input should be at least 2-dimensional
    x = (mat[..., 5, :] + mat[..., 6, :]) / 2
    y = (mat[..., 11, :] + mat[..., 12, :]) / 2

    spine_sizes = np.linalg.norm(x - y, axis=-1)
    return spine_sizes


spine_func_map = {
    "coco17": spine_size_coco
}


def spine_normalization(mat: np.ndarray, skeleton_type: str, **kwargs) -> np.ndarray:
    # Normalize skeletons with the size of a spine of a first skeleton
    spine_func = spine_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    result = mat / spine_sizes[:, np.newaxis, np.newaxis]
    return result


def mean_spine_normalization(mat: np.ndarray, skeleton_type: str, **kwargs) -> np.ndarray:
    # Normalize skeletons with the mean spine size across frames
    spine_func = spine_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    spine_size = np.mean(spine_sizes)
    result = mat / spine_size
    return result


class ScreenNormalization:
    def __init__(self, dataset_file: str):
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
        self.image_shape = data.get("im_shape", (1920, 1080))

    def __call__(self, mat: np.ndarray, **kwargs):
        return screen_normalization(mat, self.image_shape)


class SpineNormalization:
    def __init__(self, dataset_file: str, use_mean: bool = False):
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
        points_list = data['poseXY']
        skeleton_type = data.get("skeleton_type", "coco17")

        spine_func = spine_func_map[skeleton_type]
        all_spines = []
        for points in points_list:
            spines = spine_func(points)
            all_spines.append(spines[0])
        all_spines = np.concatenate(all_spines)
        if use_mean:
            self.scale = np.mean(all_spines)
        else:
            self.scale = np.max(all_spines)

    def __call__(self, mat: np.ndarray, ):
        result = mat / self.scale
        return result


def no_norm(mat):
    return mat


def create_norm_func(norm_name: str, dataset_file: str, **kwargs):
    if norm_name == "screen":
        # Requires image_shape
        return ScreenNormalization(dataset_file)
    elif norm_name == "relative":
        return relative_normalization
    elif norm_name == "spine":
        return SpineNormalization(dataset_file, use_mean=False)
    elif norm_name == "mean_spine":
        return SpineNormalization(dataset_file, use_mean=True)
    else:
        return no_norm


if __name__ == "__main__":
    dataset_file_ = "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl"
    x1 = create_norm_func("screen", dataset_file=dataset_file_)
    x2 = create_norm_func("relative", dataset_file=dataset_file_)
    x3 = create_norm_func("spine", dataset_file=dataset_file_)
    x4 = create_norm_func("mean_spine", dataset_file=dataset_file_)
