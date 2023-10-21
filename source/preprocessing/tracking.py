from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from shared.structs import SkeletonData, FrameData, Body


def skeleton_distance(skeleton1: np.ndarray, skeleton2: np.ndarray):
    distances = np.linalg.norm(skeleton1 - skeleton2, axis=1)
    return np.sum(distances)


def pose_track(frames: List[FrameData], threshold=60):
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

        # Save results
        for r, c in zip(row, col):
            track_proposals[r]["data"].append((idx, poses[c], c))

        # If there is more poses than tracks
        if m > n:
            for j in range(m):
                if j not in col:
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


def get_valid_frame_count(bodies: List[Body], ratio: float = 0.8) -> List[Body]:
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
        good_bodies = get_valid_frame_count(bodies)

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
    return selected_tids
