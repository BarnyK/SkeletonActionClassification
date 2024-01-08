import numpy as np

from shared.structs import SkeletonData, FrameData

drawn_limbs = [
    (0, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 8), (13, 12),
    (14, 13)
]

bones = (
    (0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 8),
    (13, 12), (14, 13))

angles = (
    (0, 1, 5), (4, 0, 14), (2, 1, 0), (3, 2, 1), (4, 3, 2), (5, 1, 0), (6, 5, 1), (7, 6, 5), (4, 0, 14), (9, 8, 1),
    (10, 9, 8), (11, 10, 9), (12, 8, 1), (13, 12, 8), (14, 13, 12))
angles_to_zero = (1, 8)

num_nodes = 15

edges = [(0, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5),
         (7, 6), (8, 1), (9, 8), (10, 9), (11, 10),
         (12, 8), (13, 12), (14, 13)]

center = 1

parts = [
    np.array([2, 3, 4]),  # Left arm
    np.array([5, 5, 7]),  # right arm,
    np.array([9, 10, 11]),  # left leg
    np.array([12, 13, 14]),  # right leg
    np.array([0, 1, 8]),  # Torso
]


def prepare_draw_keypoints(points: np.ndarray) -> np.ndarray:
    return points


def center_position_func(mat):
    return mat[..., center, :]


def spine_size(mat: np.ndarray) -> np.ndarray:
    # Calculates the size of spine given a matrix with skeleton
    # Input should be at least 2-dimensional
    x = mat[..., 1, :]
    y = mat[..., 8, :]

    spine_sizes = np.linalg.norm(x - y, axis=-1)
    return spine_sizes


def alignment_keypoint_value(mat: np.ndarray) -> np.ndarray:
    return mat[..., center, :]


def from_coco(mat: np.ndarray) -> np.ndarray:
    *R, V, C = mat.shape
    new_mat = np.zeros((*R, 15, C))
    direct_translations = {5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4, 11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11}
    for src, dst in direct_translations.items():
        new_mat[..., dst, :] = mat[..., src, :]
    # spine parts
    new_mat[..., 1, :] = (new_mat[..., 2, :] + new_mat[..., 5, :]) / 2
    new_mat[..., 8, :] = (new_mat[..., 9, :] + new_mat[..., 12, :]) / 2
    # Maybe calculate average of head points for point 0
    new_mat[..., 0, :] = np.mean([mat[..., i, :] for i in range(0, 5)], axis=-2)
    new_mat[..., 0, :] = 0.2 * mat[..., 0, :] + 0.4 * mat[..., 3, :] + 0.4 * mat[..., 4, :]
    return new_mat


def from_ntu(mat: np.ndarray) -> np.ndarray:
    *R, V, C = mat.shape
    new_mat = np.zeros((*R, 15, C))
    direct_translations = {0: 8, 3: 0, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6, 10: 7, 12: 9, 13: 10, 14: 11, 16: 12, 17: 13,
                           18: 14, 20: 1}

    for src, dst in direct_translations.items():
        new_mat[..., dst, :] = mat[..., src, :]

    return new_mat


transform_map = {
    "coco17": from_coco,
    "ntu": from_ntu,
}


def from_frame(frame: FrameData, skeleton_type: str):
    func = transform_map.get(skeleton_type)
    if func is None:
        raise KeyError(f"not supported type {skeleton_type}")
    for body in frame.bodies:
        body.poseXY = func(body.poseXY)
        body.poseConf = func(body.poseConf)
    return frame, "ntu_coco"


def from_skeleton_data(data: SkeletonData) -> SkeletonData:
    func = transform_map.get(data.type)
    if func is None:
        raise KeyError(f"not supported type {data.type}")
    for frame in data.frames:
        for body in frame.bodies:
            body.poseXY = func(body.poseXY)
            body.poseConf = func(body.poseConf)
    data.type = "ntu_coco"
    return data


if __name__ == "__main__":
    from pose_estimation.ntu_loader import read_file

    xd = read_file("/media/barny/SSD4/MasterThesis/Data/nturgb+d_skeletons/S001C001P001R001A001.skeleton")
    pose = xd.frames[0].bodies[0].poseXY
    newpose = from_ntu(pose)
    pass
