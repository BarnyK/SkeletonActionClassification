from shared.structs import SkeletonData


def remove_by_max_pose_confidence(data: SkeletonData, threshold: float = 0.3):
    max_conf = 0
    for frame in data.frames:
        confidences = [b.poseConf.sum() for b in frame.bodies]
        max_conf = max(max_conf, *confidences)
    min_conf = max_conf * threshold

    for frame in data.frames:
        to_remove = []
        for b in range(len(frame.bodies)):
            if frame.bodies[b].poseConf.sum() < min_conf:
                to_remove = [b] + to_remove
        for b in to_remove:
            frame.bodies.pop(b)


def remove_by_max_possible_pose_confidence(data: SkeletonData, threshold: float = 0.3):
    for frame in data.frames:
        if frame.bodies:
            min_conf = frame.bodies[0].poseXY.shape[0] * threshold
        to_remove = []
        for b in range(len(frame.bodies)):
            if frame.bodies[b].poseConf.sum() < min_conf:
                to_remove = [b] + to_remove
        for b in to_remove:
            frame.bodies.pop(b)


def remove_bodies_by_box_confidence(data: SkeletonData, threshold: float = 0.5, max_total: float = 1.0,
                                    max_frames: float = 1.0):
    """
    Remove
    :param data: SkeletonData
    :param threshold: box confidence threshold
    :param max_total: maximal amount of bodies that should be left
    :param max_frames: maximal amount of frames with bodies that should be left
    :return:
    """
    total_count = 0
    frames_with_bodies = 0
    frames_with_all_bodies_removed = 0
    to_remove = []
    for fi, frame in enumerate(data.frames):
        frame_removed = 0
        for bi, _ in enumerate(frame.bodies):
            total_count += 1
            if frame.bodies[bi].boxConf.sum() < threshold:
                to_remove = [(fi, bi)] + to_remove
                frame_removed += 1
        if frame.bodies:
            frames_with_bodies += 1
        if len(frame.bodies) == frame_removed:
            frames_with_all_bodies_removed += 1

    # In case the amount of bodies removed is bigger than percent of total: don't
    if len(to_remove) > max_total * total_count:
        return False

    # In case the amount of frames fully emptied is bigger than percent of all frames with bodies don't
    if frames_with_all_bodies_removed > max_frames * frames_with_bodies:
        return False

    # Remove frames
    for fi, bi in to_remove:
        data.frames[fi].bodies.pop(bi)
    return True


def select_biggest(data, n):
    raise NotImplementedError
