import os
from typing import Union, List

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

from shared.structs import SkeletonData, Body

enable_iterative_imputer
os.environ['NUMEXPR_NUM_THREADS'] = '2'

def keypoint_interpolation_fill(bodies: List[Body], keypoint_index: int, threshold: float = 0.4):
    ki = keypoint_index
    coords = np.stack([body.poseXY[ki, :] for body in bodies])
    confs = np.stack([body.poseConf[ki, 0] for body in bodies])
    to_fill = confs < threshold

    if to_fill.all():
        to_fill = confs == 0
        # ggs
        for i in range(coords.shape[1]):
            interp_coords(coords, i, to_fill)
        for b in range(len(bodies)):
            bodies[b].poseXY[ki, :] = coords[b, :]
    elif to_fill.any():
        for i in range(coords.shape[1]):
            interp_coords(coords, i, to_fill)
        for b in range(len(bodies)):
            bodies[b].poseXY[ki, :] = coords[b, :]


def interp_coords(coords, i, to_fill):
    fp = coords[~to_fill, i]
    xp = np.nonzero(~to_fill)[0]
    x = np.nonzero(to_fill)[0]
    coords[to_fill, i] = np.interp(x, xp, fp)


def interpolation_fill(data: SkeletonData, tid: int, threshold: float = 0.3):
    bodies = []
    for frame in data.frames:
        for body in frame.bodies:
            if body.tid == tid:
                bodies.append(body)
                break
    assert len(bodies) == data.length

    keypoint_count = bodies[0].poseXY.shape[0]
    for keypoint_index in range(keypoint_count):
        keypoint_interpolation_fill(bodies, keypoint_index, threshold)


def mice_fill(data: SkeletonData, tid: int, threshold: float = 0.3, max_iter: int = 5):
    bodies = []
    for frame in data.frames:
        for body in frame.bodies:
            if body.tid == tid:
                bodies.append(body)
                break
    assert len(bodies) == data.length
    full_matrix = np.stack([body.poseXY for body in bodies], 0)
    missing = np.stack([body.poseConf.squeeze() for body in bodies], 0) < threshold
    full_matrix[missing] = np.nan

    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    for i in range(full_matrix.shape[2]):
        imputer.fit(full_matrix[:, :, i])
        full_matrix[:, ~missing.all(0), i] = imputer.transform(full_matrix[:, :, i])

    for i, body in enumerate(bodies):
        body.poseXY = full_matrix[i, :, :]


def knn_fill(data: SkeletonData, tid: int, threshold: float = 0.3, neighbours: int = 5):
    bodies = []
    for frame in data.frames:
        for body in frame.bodies:
            if body.tid == tid:
                bodies.append(body)
                break
    assert len(bodies) == data.length
    full_matrix = np.stack([body.poseXY for body in bodies], 0)
    missing = np.stack([body.poseConf.squeeze() for body in bodies], 0) < threshold
    full_matrix[missing] = np.nan

    imputer = KNNImputer(n_neighbors=neighbours)
    for i in range(full_matrix.shape[2]):
        full_matrix[:, ~missing.all(0), i] = imputer.fit_transform(full_matrix[:, :, i])

    for i, body in enumerate(bodies):
        body.poseXY = full_matrix[i, :, :]


def fill_missing_frames(skeleton_data: SkeletonData, tid: int):
    # Find any body for reference
    __added_count = 0

    ref_body: Union[Body, None] = None
    for frame in skeleton_data.frames:
        if frame.bodies:
            ref_body = frame.bodies[0]
            break
    if ref_body is None:
        raise ValueError("No body in skeleton frames")

    for frame in skeleton_data.frames:
        if tid not in [body.tid for body in frame.bodies]:
            new_body = Body(
                poseXY=np.zeros_like(ref_body.poseXY),
                poseConf=np.zeros_like(ref_body.poseConf),
                box=None,
                boxConf=None,
                poseXYZ=np.zeros_like(ref_body.poseXYZ) if ref_body.poseXYZ is not None else None,
                tid=tid,
            )
            frame.bodies.append(new_body)
            __added_count += 1
    return __added_count


def keypoint_fill(skeleton_data: SkeletonData, fill_type: str = "interpolation", threshold: float = 0.3,
                  max_iters: int = 5, neighbours: int = 5):
    assert fill_type in ["interpolation", "mice", "knn"]

    for tid in skeleton_data.get_all_tids():
        fill_missing_frames(skeleton_data, tid)
        if fill_type == "interpolation":
            interpolation_fill(skeleton_data, tid, threshold)
        elif fill_type == "mice":
            mice_fill(skeleton_data, tid, threshold, max_iters)
        elif fill_type == "knn":
            knn_fill(skeleton_data, tid, threshold, neighbours)
