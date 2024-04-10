from __future__ import annotations

from typing import Iterable

import numpy as np

from shared.skeletons import bones_map, angles_map, center_pos_map


def joints_to_bones(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Calculate bone vectors from joints
    bone_pairs = bones_map.get(skeleton_type)
    if bone_pairs is None:
        raise ValueError()

    indices_1 = [x for x, _ in bone_pairs]
    indices_2 = [y for _, y in bone_pairs]
    result = mat[..., indices_1, :] - mat[..., indices_2, :]

    return result


def to_motion(mat: np.ndarray) -> np.ndarray:
    result = mat.copy()
    result[..., 1:, :, :] = np.diff(result, 1, axis=-3)
    return result


def to_accel(mat: np.ndarray) -> np.ndarray:
    result = mat.copy()
    result[..., 1:, :, :] = np.diff(result, 1, axis=-3)
    result[..., 1:, :, :] = np.diff(result, 1, axis=-3)
    return result


def bone_angles(mat: np.ndarray) -> np.ndarray:
    # Calculate angles of bones to axes
    magnitudes = np.linalg.norm(mat, axis=-1) + 0.0001
    result = np.arccos(mat / magnitudes[..., np.newaxis] % 1)
    return result


def to_angles(mat: np.ndarray, skeleton_type: str) -> np.ndarray:
    # Calculate angles in the body
    # Coco angles have shoulder points set to 0
    angles, to_zero = angles_map.get(skeleton_type)
    if angles is None:
        raise ValueError

    angle_array = __calculate_angles(mat, angles)
    angle_array[..., to_zero] = 0
    angle_array /= (2 * np.pi)

    return angle_array[..., np.newaxis]


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
    center_func = center_pos_map.get(skeleton_type)
    centers = center_func(mat)
    result = mat - centers[..., np.newaxis, :]
    return result
