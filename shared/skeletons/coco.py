import numpy as np

drawn_limbs = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (17, 11),
    (17, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

bones = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
         (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))

angles = ((6, 0, 5), (1, 0, 5), (2, 0, 6), (0, 1, 3), (0, 2, 4), (5, 14, 5), (5, 14, 5), (7, 5, 11), (8, 6, 12),
          (5, 7, 9), (6, 8, 10), (0, 5, 11), (0, 6, 12), (5, 11, 13), (6, 12, 14), (11, 13, 15),
          (12, 14, 16))
angles_to_zero = (5, 6)
num_nodes = 17

edges = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
         (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
         (1, 0), (3, 1), (2, 0), (4, 2)]

center = 0

parts = [
    np.array([6, 8, 10]),  # Left arm
    np.array([5, 7, 9]),  # right arm,
    np.array([11, 13, 15]),  # left leg
    np.array([12, 14, 16]),  # right leg
    np.array([0, 1, 2, 3, 4]),  # Head
]


def prepare_draw_keypoints(points: np.ndarray) -> np.ndarray:
    # Add points between shoulders
    points = np.concatenate((points, (points[5:6, :] + points[6:7, :]) / 2))
    return points


def center_position_func(mat):
    return (mat[..., 5, :] + mat[..., 6, :]) / 2


def spine_size(mat: np.ndarray) -> np.ndarray:
    # Calculates the size of spine given a matrix with skeleton
    # Input should be at least 2-dimensional
    x = (mat[..., 5, :] + mat[..., 6, :]) / 2
    y = (mat[..., 11, :] + mat[..., 12, :]) / 2

    spine_sizes = np.linalg.norm(x - y, axis=-1)
    return spine_sizes


def alignment_keypoint_value(mat: np.ndarray) -> np.ndarray:
    return mat[..., 5, :]
