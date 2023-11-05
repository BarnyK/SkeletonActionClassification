from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from shared.structs import SkeletonData, FrameData, Body


def skeleton_distance(skeleton1: np.ndarray, skeleton2: np.ndarray):
    distances = np.linalg.norm(skeleton1 - skeleton2, axis=1)
    return np.sum(distances)


def skeleton_bbox(skeleton: np.ndarray):
    min_x, max_x = np.min(skeleton[..., 0]), np.max(skeleton[..., 0])
    min_y, max_y = np.min(skeleton[..., 1]), np.max(skeleton[..., 1])
    return min_x, max_x, min_y, max_y


def body_bbox(body: Body):
    return skeleton_bbox(body.poseXY)


def skeleton_middle(skeleton: np.ndarray):
    min_x, max_x, min_y, max_y = skeleton_bbox(skeleton)
    return np.array([min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2], dtype=np.float32)


def calculate_movement_to_body_ratio(skeleton1: np.ndarray, skeleton2: np.ndarray, frames: int):
    min_x, max_x, min_y, max_y = skeleton_bbox(skeleton1)
    w = max_x - min_x
    h = max_y - min_y
    mid1 = skeleton_middle(skeleton1)
    mid2 = skeleton_middle(skeleton2)
    dist = np.abs(mid2 - mid1)
    dist_per_frame = dist / frames
    ratios = dist_per_frame / np.array([w, h])
    return ratios


def pose_track(frames: List[FrameData], threshold=60, width_ratio: float = 100.0, height_ratio: float = 50.0):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, frame in enumerate(frames):
        poses = [x.poseXY for x in frame.bodies]
        if len(poses) == 0:
            # skip frames with no bodies
            continue
        if num_joints is None:
            # Figure out how many joints in a body
            num_joints = poses[0].shape[0]

        # Create tracks.
        track_proposals = [t for t in tracks if t["data"][-1][0] > idx - threshold]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        # for each combination of track and pose calculate distances and solve linear sum assignment
        for i in range(n):
            for j in range(m):
                scores[i][j] = skeleton_distance(
                    track_proposals[i]["data"][-1][1], poses[j]
                )
        row, col = linear_sum_assignment(scores)
        additional_col = []
        # Save results
        for r, c in zip(row, col):
            move_ratios = calculate_movement_to_body_ratio(track_proposals[r]["data"][-1][1],
                                                           poses[c],
                                                           idx - track_proposals[r]["data"][-1][0])
            if (move_ratios > np.array([width_ratio, height_ratio])).any():
                # deny
                additional_col.append(c)
            track_proposals[r]["data"].append((idx, poses[c], c))

        # If there is more poses than tracks
        for j in range(m):
            if j not in col or j in additional_col:
                num_tracks += 1
                new_track = dict(data=[])
                new_track["track_id"] = num_tracks
                new_track["data"] = [(idx, poses[j], j)]
                tracks.append(new_track)

    # Assign tracking ids to bodies
    for i, track in enumerate(tracks):
        for item in track["data"]:
            idx, _, pos = item
            frames[idx].bodies[pos].tid = i


def get_valid_bodies(bodies: List[Body], ratio: float = 0.8) -> List[Body]:
    good_bodies = []
    for body in bodies:
        xs = body.poseXY[:, 0]
        min_x = xs.min()
        max_x = xs.max()

        ys = body.poseXY[:, 1]
        min_y = ys.min()
        max_y = ys.max()

        if (max_x - min_x) / (max_y - min_y) <= ratio:
            good_bodies.append(body)

    return good_bodies


def select_tracks_by_motion(data: SkeletonData, max_bodies: int = 2):
    motions = {}
    backup_motions = {}
    all_tids = data.get_all_tids()
    for tid in all_tids:
        bodies = data.get_all_bodies_for_tid(tid)
        good_bodies = get_valid_bodies(bodies)

        backup_motions[tid] = min(
            np.sum(np.var(np.vstack([x.poseXY for x in bodies]), axis=0)),
            np.sum(np.var(np.vstack([x.poseXY for x in good_bodies]), axis=0)) if good_bodies else np.inf
        )

        if len(good_bodies) > 5 and len(good_bodies) / len(bodies) >= 0.3:
            motions[tid] = backup_motions[tid]

    if not motions:
        motions = backup_motions
    # Select tracks with the highest motion value
    selected_tids = sorted([tid for tid in motions.keys()], key=lambda track_id: -motions[track_id])
    selected_tids = selected_tids[:max_bodies]
    for frame in data.frames:
        frame.bodies = [body for body in frame.bodies if body.tid in selected_tids]
        # Sort tids by motion
        frame.bodies = sorted(frame.bodies, key=lambda x: selected_tids.index(x.tid))

    # Reassign ids so that lowest has highest motion
    tid_mapping = {tid: i for i, tid in enumerate(selected_tids)}
    all_bodies = [body for frame in data.frames for body in frame.bodies]
    for body in all_bodies:
        body.tid = tid_mapping[body.tid]

    return selected_tids


def ntu_track_selection(data: SkeletonData, max_bodies: int = 2, length_threshold: int = 11):
    tids = data.get_all_tids()
    if len(data.get_all_tids()) == 1:
        return

    # Remove short sequences
    for tid in tids:
        if len(data.get_all_bodies_for_tid(tid)) < length_threshold:
            data.remove_bodies_for_tid(tid)
    if len(data.get_all_tids()) == 1:
        return

    # Spread denoising
    good_motions, all_motions = {}, {}
    for tid in tids:
        bodies = data.get_all_bodies_for_tid(tid)
        good_bodies = get_valid_bodies(bodies, 0.8)

        all_motions[tid] = min(
            np.sum(np.var(np.vstack([x.poseXY for x in bodies]), axis=0)),
            np.sum(np.var(np.vstack([x.poseXY for x in good_bodies]), axis=0)) if good_bodies else np.inf
        )
        if (1 - 0.69754) < + (len(good_bodies) / len(bodies)):
            good_motions[tid] = all_motions[tid]
        else:
            data.remove_bodies_for_tid(tid)

    good_tids = sorted([key for key in good_motions.keys()], key=lambda x: -good_motions[x])
    tid_mapping = {tid: i for i, tid in enumerate(good_tids)}
    all_bodies = [body for frame in data.frames for body in frame.bodies]
    for body in all_bodies:
        body.tid = tid_mapping[body.tid]
    if len(good_tids) == 1:
        return

    ## Wacky stuff with track combinations
    # Combine all tracks into max_bodies number of them
    # Priority by good_tids
    tid_dict = {tid: data.get_all_bodies_for_tid_with_seq(tid) for tid in good_tids}
    tid_starts = {tid: min([x[0] for x in tid_dict[tid]]) for tid in good_tids}
    tid_ends = {tid: max([x[0] for x in tid_dict[tid]]) for tid in good_tids}
    tracks = [[good_tids[0]], *([[]] * (max_bodies - 1))]
    track_bounds = [(tid_starts[good_tids[0]], tid_ends[good_tids[0]]), (0, 0)] * (max_bodies - 1)
    for tid in good_tids[1:]:
        s, e = tid_starts[tid], tid_ends[tid]
        for ti in range(max_bodies):
            st, et = track_bounds[ti]
            if max(st, s) >= min(et, e):
                tracks[ti].append(tid)
                track_bounds[ti] = min(s, st), max(e, et)

    mapping = {}
    for track_id in range(max_bodies):
        tids = tracks[track_id]
        if tids:
            mapping.update({tid: track_id for tid in tids })
    for good_tid in good_tids:
        if good_tid not in mapping.keys():
            data.remove_bodies_for_tid(good_tid)
    all_bodies = [body for frame in data.frames for body in frame.bodies]
    for body in all_bodies:
        body.tid = tid_mapping[body.tid]
    return good_tids


def select_by_size(data: SkeletonData, max_bodies: int = 2):
    for frame in data.frames:
        if not frame.bodies:
            continue
        bboxes = [body_bbox(body) for body in frame.bodies]
        sizes = [(bb[1] - bb[0]) * (bb[3] - bb[2]) for bb in bboxes]
        sizes = sorted([(i, bb) for i, bb in enumerate(sizes)], key=lambda x: x[1], )
        sizes = sizes[:max_bodies]
        frame.bodies = [body for i, body in enumerate(frame.bodies) if i in [x[0] for x in sizes]]


def select_by_confidence(data: SkeletonData, max_bodies: int = 2):
    for frame in data.frames:
        if not frame.bodies:
            continue
        confidences = [np.sum(body.poseConf) for body in frame.bodies]
        confidences = sorted([(i, bb) for i, bb in enumerate(confidences)], key=lambda x: x[1], )
        confidences = confidences[:max_bodies]
        frame.bodies = [body for i, body in enumerate(frame.bodies) if i in [x[0] for x in confidences]]


def select_by_order(data: SkeletonData, max_bodies: int = 2):
    for frame in data.frames:
        frame.bodies = frame.bodies[:max_bodies]


def assign_tids_by_order(data: SkeletonData):
    for frame in data.frames:
        for i, body in enumerate(frame.bodies):
            body.tid = i
