import cv2
import numpy as np

from shared.read_skeleton import read_skeleton
from shared.structs import SkeletonData

limbs = {
    "coco_test_adjacency": [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                            (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                            (1, 0), (3, 1), (2, 0), (4, 2)],
    "coco_test_bones": [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                        (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14)],
    "coco17": [
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
    ],
    "mpii": [
        (8, 9),
        (11, 12),
        (11, 10),
        (2, 1),
        (1, 0),
        (13, 14),
        (14, 15),
        (3, 4),
        (4, 5),
        (8, 7),
        (7, 6),
        (6, 2),
        (6, 3),
        (8, 12),
        (8, 13),
    ],
    "ntu": [
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
    ],
    "halpe": [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # Head
        (5, 18),
        (6, 18),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # Body
        (17, 18),
        (18, 19),
        (19, 11),
        (19, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
        (20, 24),
        (21, 25),
        (23, 25),
        (22, 24),
        (15, 24),
        (16, 25),  # Foot
    ],
}

tracking_colors = {
    0: (128, 128, 0),
    1: (0, 165, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (147, 20, 255),
}


def draw_text_with_outline(image, point, text, font_size):
    point = point[0] + 10, point[1]
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def visualize(skeleton_data: SkeletonData, video_file: str, wait_key: int = 0, window_name: str = "visualization"):
    skeleton_type = skeleton_data.type
    limb_pairs = limbs.get(skeleton_type)
    if limb_pairs is None:
        raise ValueError(f"Skeleton type not available {skeleton_type}")

    if video_file is None:
        video_file = skeleton_data.video_file

    video_stream = cv2.VideoCapture(video_file)
    frame_size = (
        int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    length = skeleton_data.length

    for i in range(length):
        skel = skeleton_data.frames[i]
        (grabbed, frame) = video_stream.read()
        if not grabbed:
            break

        skeleton_color = (255, 0, 255)
        for body in skel.bodies:
            skeleton_color = (skeleton_color[1], skeleton_color[2], skeleton_color[0])
            if body.tid is not None:
                skeleton_color = tracking_colors[body.tid % len(tracking_colors)]

            # Boxes
            if body.box is not None:
                p1, p2 = (int(body.box[0]), int(body.box[1])), (
                    int(body.box[2]),
                    int(body.box[3]),
                )
                frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                conf = body.boxConf
                if conf is not None:
                    conf = conf[0]
                    draw_text_with_outline(frame, p1, str(round(float(conf), 3)), 6 / 10)
                conf = body.poseConf.mean()
                if conf is not None:
                    p1 = max(p1[0], 0), max(p1[1] - 15, 0)
                    draw_text_with_outline(frame, p1, str(round(float(conf), 3)), 6 / 10)

            # Points
            points = body.poseXY
            if skeleton_type == "coco17" or len(points) == 17:
                # Additional point between shoulders
                points = np.concatenate((points, (points[5:6, :] + points[6:7, :]) / 2))

            for i, (p1, p2) in enumerate(limb_pairs):
                point1 = (int(points[p1][0]), int(points[p1][1]))
                point2 = (int(points[p2][0]), int(points[p2][1]))
                cv2.line(frame, point1, point2, skeleton_color, 2)

            for ii, point in enumerate(points):
                point = (int(point[0]), int(point[1]))
                draw_text_with_outline(frame, point, str(ii), 5 / 10)

                if ii in []:
                    cv2.circle(frame, point, 10, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, point, 4, skeleton_color, -1)

            # Limbs

            # Confidence

        cv2.imshow(window_name, frame)
        cv2.waitKey(wait_key)

    video_stream.release()
    cv2.destroyAllWindows()


def visualize_alphapose(skeleton_file: str, wait_key: int = 0):
    skeleton_data = read_skeleton(skeleton_file)
    visualize(skeleton_data, skeleton_data.video_file, wait_key)


def visualize_ntu(skeleton_file: str, video_file: str, wait_key: int = 0):
    skeleton_data = read_skeleton(skeleton_file)
    visualize(skeleton_data, video_file, wait_key)


def visualize_data(data: SkeletonData, wait_key: int = 0):
    all_bodies = [body for frame in data.frames for body in frame.bodies]
    stack = np.stack([body.poseXY for body in all_bodies])
    x_min, x_max = stack[:, :, 0].min(), stack[:, :, 0].max()
    y_min, y_max = stack[:, :, 1].min(), stack[:, :, 1].max()
    im_w, im_h = 1920, 1080

    xt_scale = (im_w * 0.8) / (x_max - x_min)

    yt_scale = (im_h * 0.8) / (y_max - y_min)

    limb_pairs = limbs.get(data.type)

    for frame in data.frames:
        image = np.zeros((im_h, im_w, 3))
        skeleton_color = (255, 0, 255)

        for body in frame.bodies:
            skeleton_color = (skeleton_color[1], skeleton_color[2], skeleton_color[0])
            if body.tid is not None:
                skeleton_color = tracking_colors[body.tid % len(tracking_colors)]

            points = body.poseXY
            if data.type == "coco17" or len(points) == 17:
                # Additional point between shoulders
                points = np.concatenate((points, (points[5:6, :] + points[6:7, :]) / 2))

            points = (np.array([im_w * 0.1, im_h * 0.1]) + (points - np.array([x_min, y_min])) *
                      np.array([xt_scale, yt_scale]))

            for _, point in enumerate(points):
                point = (int(point[0]), int(point[1]))
                cv2.circle(image, point, 4, skeleton_color, -1)

            # Limbs
            for i, (p1, p2) in enumerate(limb_pairs):
                point1 = (int(points[p1][0]), int(points[p1][1]))
                point2 = (int(points[p2][0]), int(points[p2][1]))
                cv2.line(image, point1, point2, skeleton_color, 2)

            x = (points[5, :] + points[6, :]) / 2
            y = (points[11, :] + points[12, :]) / 2
            point1 = (int(x[0]), int(x[1]))
            point2 = (int(y[0]), int(y[1]))
            cv2.line(image, point1, point2, (255, 255, 255), 4)
            cv2.putText(
                image,
                f"{round(np.linalg.norm(x - y), 3)}",
                (point1[0] + 100, point1[1] + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("visualization", image)
        cv2.waitKey(wait_key)

    cv2.destroyAllWindows()
