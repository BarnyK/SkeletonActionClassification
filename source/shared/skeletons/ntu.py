import numpy as np

drawn_limbs = [
    (0, 1),
    (1, 20),
    (2, 20),
    (2, 3),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 24),
    (23, 24),
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 22),
    (22, 21),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15)
]

bones = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
         (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
         (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))

angles = ((7, 3, 19),  # zeroed
          (0, 1, 20),
          (2, 20, 4),
          (3, 2, 20),
          (4, 20, 1),
          (5, 4, 20),
          (6, 5, 4),
          (7, 6, 5),
          (8, 20, 1),
          (9, 8, 20),
          (10, 9, 8),
          (11, 10, 9),
          (12, 0, 1),
          (13, 12, 0),
          (14, 13, 12),
          (15, 14, 13),
          (16, 0, 1),
          (17, 16, 0),
          (18, 17, 16),
          (19, 18, 17),
          (7, 3, 19),  # zeroed
          (21, 7, 6),
          (22, 7, 21),
          (23, 11, 10),
          (24, 11, 23)
          )
angles_to_zero = (0, 20)

num_nodes = 25

edges = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
         (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 7), (22, 7), (23, 11), (24, 11)]

center = 20


def prepare_draw_keypoints(points: np.ndarray) -> np.ndarray:
    return points


def center_position_func(mat):
    return mat[..., center, :]


def spine_size(mat: np.ndarray) -> np.ndarray:
    # Calculates the size of spine given a matrix with skeleton
    # Input should be at least 2-dimensional
    x = mat[..., 0, :]
    y = mat[..., 20, :]

    spine_sizes = np.linalg.norm(x - y, axis=-1)
    return spine_sizes


def alignment_keypoint_value(mat: np.ndarray) -> np.ndarray:
    return mat[..., center, :]
