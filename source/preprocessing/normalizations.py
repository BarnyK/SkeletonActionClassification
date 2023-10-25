from __future__ import annotations

from typing import Iterable

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


def joints_to_bones(points: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Calculate bone vectors from joints
    bones = {"coco17": ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                        (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
             }
    assert points.ndim == 4

    bone_pairs = bones.get(skeleton_type)
    if bone_pairs is None:
        raise ValueError()

    indices_1 = [x for x, _ in bone_pairs]
    indices_2 = [y for _, y in bone_pairs]
    bone_array = points[:, :, indices_1, :] - points[:, :, indices_2, :]

    return bone_array


def to_motion(mat: np.ndarray) -> np.ndarray:
    result = mat.copy()
    result[..., 1:, :, :] = np.diff(result, 1, axis=1)
    return result


def to_accel(mat: np.ndarray) -> np.ndarray:
    result = mat.copy()
    result[..., 1:, :, :] = np.diff(result, 1, axis=1)
    result[..., 1:, :, :] = np.diff(result, 1, axis=1)
    return result


def bone_angles(mat: np.ndarray) -> np.ndarray:
    # Calculate angles of bones to axes
    np.seterr(divide='ignore', invalid='ignore')
    magnitudes = np.linalg.norm(mat, axis=-1)
    angles = np.arccos(mat / magnitudes[..., np.newaxis] % 1)
    return angles


def to_angles(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Calculate angles in the body
    # Coco angles have shoulder points set to 0
    angles_map = {
        "coco17": (
            (6, 0, 5), (1, 0, 5), (2, 0, 6), (0, 1, 3), (0, 2, 4), (5, 14, 5), (5, 14, 5), (7, 5, 11), (8, 6, 12),
            (5, 7, 9), (6, 8, 10), (0, 5, 11), (0, 6, 12), (5, 11, 13), (6, 12, 14), (11, 13, 15),
            (12, 14, 16)),
    }

    angles = angles_map.get(skeleton_type)
    if angles is None:
        raise ValueError

    angle_array = __calculate_angles(mat, angles)
    angle_array[..., (5, 6)] = 0
    angle_array /= (2 * np.pi)

    return angle_array


def __calculate_angles(points: np.ndarray, angle_definitions: Iterable[tuple[int, int, int]]) -> np.ndarray:
    # Extract the three points for each angle definition
    p1_indices, p2_indices, p3_indices = np.array(angle_definitions).T
    p1 = points[..., p1_indices, :]
    p2 = points[..., p2_indices, :]
    p3 = points[..., p3_indices, :]

    # Calculate vectors from p2 to p1 and p3 to p2
    vector1 = p1 - p2
    vector2 = p3 - p2

    # Calculate dot products, magnitudes, and angles for all angles at once
    np.seterr(divide='ignore', invalid='ignore')
    dot_products = np.sum(vector1 * vector2, axis=-1)
    magnitudes1 = np.linalg.norm(vector1, axis=-1)
    magnitudes2 = np.linalg.norm(vector2, axis=-1)
    cosines: np.ndarray = dot_products / (magnitudes1 * magnitudes2)
    angles_in_radians = np.arccos(cosines)
    angles_in_radians[np.isnan(angles_in_radians)] = 0
    # angles_in_degrees = np.degrees(angles_in_radians)

    return angles_in_radians


def relative_joints(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    center_pos_map = {
        "coco17": lambda x: (x[..., 5, :] + x[..., 6, :]) / 2
    }
    center_func = center_pos_map.get(skeleton_type)
    centers = center_func(mat)
    result = mat - centers[:, :, np.newaxis, :]
    return result
