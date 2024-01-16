from __future__ import annotations

import pickle

import numpy as np

from shared.skeletons import spine_size_func_map, align_func_map


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


def spine_normalization(mat: np.ndarray, skeleton_type: str, **kwargs) -> np.ndarray:
    # Normalize skeletons with the size of a spine of a first skeleton
    spine_func = spine_size_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    result = mat / spine_sizes[:, np.newaxis, np.newaxis]
    return result


def mean_spine_normalization(mat: np.ndarray, skeleton_type: str, **kwargs) -> np.ndarray:
    # Normalize skeletons with the mean spine size across frames
    spine_func = spine_size_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    spine_size = np.mean(spine_sizes)
    result = mat / spine_size
    return result


class ScreenNormalization:
    def __init__(self):
        self.image_shape = (1920, 1080)

    def __call__(self, mat: np.ndarray, **kwargs):
        return screen_normalization(mat, self.image_shape)

    def load_from_train_file(self, train_file):
        with open(train_file, "rb") as f:
            data = pickle.load(f)
        self.image_shape = data.get("im_shape", (1920, 1080))

    def load_from_state_dict(self, state_dict: dict):
        self.image_shape = state_dict['im_shape']

    def state_dict(self):
        return {"im_shape": self.image_shape}


class SpineNormalization:
    def __init__(self, use_mean: bool = False, align: bool = False):
        self.use_mean = use_mean
        self.align = align
        self.skeleton_type = None
        self.align_func = None
        self.scale = None

    def load_from_train_file(self, train_file):
        with open(train_file, "rb") as f:
            data = pickle.load(f)
        points_list = data['poseXY']
        self.skeleton_type = data.get("skeleton_type", "coco17")
        self.align_func = align_func_map.get(self.skeleton_type)
        spine_func = spine_size_func_map[self.skeleton_type]
        all_spines = []
        for points in points_list:
            spines = spine_func(points)
            all_spines.append(spines[0])
        all_spines = np.concatenate(all_spines)
        if self.use_mean:
            self.scale = np.mean(all_spines[all_spines < np.inf])
        else:
            self.scale = np.max(all_spines[all_spines < np.inf])

    def load_from_state_dict(self, state_dict: dict):
        self.align = state_dict['align']
        self.skeleton_type = state_dict['skeleton_type']
        self.scale = state_dict['scale']
        self.align_func = align_func_map.get(self.skeleton_type)
        self.use_mean = state_dict['use_mean']

    def __call__(self, mat: np.ndarray):
        result = mat / self.scale
        if self.align:
            align_value = self.align_func(result[..., 0, 0, :, :])
            result = result - align_value
        return result

    def state_dict(self):
        return {"align": self.align, "skeleton_type": self.skeleton_type, "scale": self.scale,
                "use_mean": self.use_mean}


def no_norm(mat):
    return mat


def create_norm_func(norm_name: str):
    if norm_name == "screen":
        # Requires image_shape
        norm_func = ScreenNormalization()
    elif norm_name == "relative":
        norm_func = relative_normalization
    elif norm_name == "spine":
        norm_func = SpineNormalization(use_mean=False, align=False)
    elif norm_name == "mean_spine":
        norm_func = SpineNormalization(use_mean=True, align=False)
    elif norm_name == "spine_align":
        norm_func = SpineNormalization(use_mean=False, align=True)
    elif norm_name == "mean_spine_align":
        norm_func = SpineNormalization(use_mean=True, align=True)
    elif norm_name == "none":
        norm_func = no_norm
    else:
        raise KeyError("")
    return norm_func


def setup_norm_func(norm_func, state_dict: dict = None, train_file: str = None):
    if isinstance(norm_func, (SpineNormalization, ScreenNormalization)):
        if state_dict:
            norm_func.load_from_state_dict(state_dict)
        elif train_file:
            norm_func.load_from_train_file(train_file)
    return


norm_types = ["screen", "relative", "spine", "mean_spine", "spine_align", "mean_spine_align", "none"]
