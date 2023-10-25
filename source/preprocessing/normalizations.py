from __future__ import annotations

import numpy as np


def screen_normalization(mat: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    # Normalize points by shifting points by half the size of screen and diving by it
    w, h = screen_size
    result = np.zeros_like(mat)
    result[..., 0] = (mat[..., 0] - w / 2) / (w / 2)
    result[..., 1] = (mat[..., 1] - h / 2) / (h / 2)
    return result


def relative_normalization(mat: np.ndarray) -> np.ndarray:
    # Normalizes points using minimal and maximal values of points
    result = mat.copy()
    x_max = np.max(mat[..., 0])
    x_min = np.min(mat[..., 0])
    y_max = np.max(mat[..., 1])
    y_min = np.min(mat[..., 1])

    w = x_max - x_min
    h = y_max - y_min

    result[..., 0] = (mat[..., 0] - w / 2) / (w * 2)
    result[..., 1] = (mat[..., 1] - h / 2) / (h * 2)
    return result


def spine_size_coco(mat: np.ndarray) -> np.ndarray:
    # Calculates the size of spine given a matrix with skeleton
    # Input should be at least 2-dimensional
    x = (mat[..., 5, :] + mat[..., 6, :]) / 2
    y = (mat[..., 11, :] + mat[..., 12, :]) / 2

    spine_sizes = np.linalg.norm(x - y, axis=-1)
    return spine_sizes


def spine_normalization(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Normalize skeletons with the size of a spine of a first skeleton
    spine_func_map = {
        "coco17": spine_size_coco
    }
    spine_func = spine_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    result = mat / spine_sizes[:, np.newaxis, np.newaxis]
    return result


def mean_spine_normalization(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Normalize skeletons with the mean spine size across frames
    spine_func_map = {
        "coco17": spine_size_coco
    }
    spine_func = spine_func_map[skeleton_type]

    spine_sizes = spine_func(mat[0, ...])
    spine_size = np.mean(spine_sizes)
    result = mat / spine_size
    return result


