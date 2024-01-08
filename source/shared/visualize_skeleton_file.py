from queue import Queue
from threading import Thread
from typing import Union

import cv2
import numpy as np

from pose_estimation import every_nth_frame
from shared.read_skeleton import read_skeleton
from shared.skeletons import drawn_limbs_map, draw_preparation_func_map
from shared.structs import SkeletonData, FrameData

tracking_colors = {
    0: (128, 128, 0),
    1: (0, 165, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (147, 20, 255),
}


def draw_text_with_outline(image, point, text, font_size):
    point = point[0] + 10, point[1] + 50
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visualize(skeleton_data: SkeletonData, video_file: str, wait_key: int = 0, window_name: str = "visualization",
              draw_bbox: bool = False, draw_frame_number: bool = False, draw_confidences: bool = False,
              skip_frames: bool = False, save_file: str = None, draw_point_number: bool = False,
              print_frame_text: bool = False):
    skeleton_type = skeleton_data.type
    limb_pairs = drawn_limbs_map.get(skeleton_type)
    if limb_pairs is None:
        raise ValueError(f"Skeleton type not available {skeleton_type}")

    prep_keypoints = draw_preparation_func_map.get(skeleton_type, lambda x: x)

    if video_file is None:
        video_file = skeleton_data.video_file

    video_stream = cv2.VideoCapture(video_file)
    frame_size = (
        int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    if skip_frames:
        wrapped_stream = every_nth_frame(video_stream, skeleton_data.frame_interval)
        interval = 1
    else:
        wrapped_stream = every_nth_frame(video_stream, 1)
        interval = skeleton_data.frame_interval

    out_stream = None
    if save_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_stream = cv2.VideoWriter(save_file, fourcc, 24, frame_size)
        out_stream.set(cv2.VIDEOWRITER_PROP_QUALITY, 1)

    i = -1
    while True:
        i += 1
        (grabbed, frame) = next(wrapped_stream)
        if not grabbed:
            break

        if draw_frame_number:
            draw_text_with_outline(frame, (50, 50), str(i), 1)

        # Skip skeleton frame
        if not skip_frames and i % interval != 0:
            if save_file:
                out_stream.write(frame)
                continue
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait_key)
            continue

        # Frames from skeleton data ended
        if i // interval >= len(skeleton_data.frames):
            if save_file:
                out_stream.write(frame)
                continue
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait_key)
            continue

        frame_data = skeleton_data.frames[i // interval]
        frame = draw_frame_data(frame, frame_data, limb_pairs, prep_keypoints, draw_bbox, draw_confidences,
                                draw_point_number, print_frame_text)

        if save_file:
            out_stream.write(frame)
            continue
        cv2.imshow(window_name, frame)
        cv2.waitKey(wait_key)

    video_stream.release()
    if save_file:
        out_stream.release()
    cv2.destroyAllWindows()


def draw_frame_data(frame, frame_data, limb_pairs, prep_keypoints, draw_bbox, draw_confidences, draw_point_number,
                    print_frame_text):
    if print_frame_text and frame_data.text:
        draw_text_with_outline(frame, (50, 50), frame_data.text, 3)
    skeleton_color = (255, 0, 255)
    for body in frame_data.bodies:
        skeleton_color = (skeleton_color[1], skeleton_color[2], skeleton_color[0])
        if body.tid is not None:
            skeleton_color = tracking_colors[body.tid % len(tracking_colors)]

        # Boxes
        if draw_bbox and body.box is not None:
            p1, p2 = (int(body.box[0]), int(body.box[1])), (
                int(body.box[2]),
                int(body.box[3]),
            )
            frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            if draw_confidences:
                conf = body.boxConf
                if conf is not None:
                    conf = conf[0]
                    draw_text_with_outline(frame, p1, str(round(float(conf), 3)), 6 / 10)
                conf = body.poseConf.mean()
                if conf is not None:
                    p1 = max(p1[0], 0), max(p1[1] - 17, 0)
                    draw_text_with_outline(frame, p1, str(round(float(conf), 3)), 6 / 10)

        # Points
        points = body.poseXY

        # Additional points if needed
        points = prep_keypoints(points)

        # Draw limbs
        for ii, (p1, p2) in enumerate(limb_pairs):
            point1 = (int(points[p1][0]), int(points[p1][1]))
            point2 = (int(points[p2][0]), int(points[p2][1]))
            cv2.line(frame, point1, point2, skeleton_color, 2)

        # Draw points
        for ii, point in enumerate(points):
            point = (int(point[0]), int(point[1]))
            if draw_point_number:
                draw_text_with_outline(frame, point, str(ii), 5 / 10)

            if ii in []:
                cv2.circle(frame, point, 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame, point, 4, skeleton_color, -1)
    return frame


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

    limb_pairs = drawn_limbs_map.get(data.type)
    prep_keypoints = draw_preparation_func_map.get(data.type, lambda mat: mat)

    for frame in data.frames:
        image = np.zeros((im_h, im_w, 3))
        skeleton_color = (255, 0, 255)

        for body in frame.bodies:
            skeleton_color = (skeleton_color[1], skeleton_color[2], skeleton_color[0])
            if body.tid is not None:
                skeleton_color = tracking_colors[body.tid % len(tracking_colors)]

            points = body.poseXY
            points = prep_keypoints(points)

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


# def visualize(skeleton_data: SkeletonData, video_file: str, wait_key: int = 0, window_name: str = "visualization",
#               draw_bbox: bool = False, draw_frame_number: bool = False, draw_confidences: bool = False,
#               skip_frames: bool = False, save_file: str = None, draw_point_number: bool = False,
#               print_frame_text: bool = False):
class Visualizer:
    def __init__(self, video_file: str, skeleton_type: str, frame_interval: int, skip_frames: bool,
                 save_file: str):
        self.queue = Queue()
        self.video_file = video_file

        self.skeleton_type = skeleton_type
        self.frame_interval = frame_interval
        self.skip_frames = skip_frames

        self.limb_pairs = drawn_limbs_map.get(self.skeleton_type)
        if self.limb_pairs is None:
            raise ValueError(f"Skeleton type not available {self.skeleton_type}")
        self.prep_keypoints = draw_preparation_func_map.get(self.skeleton_type, lambda x: x)

        self.draw_frame_number = True
        self.save_file = save_file
        self.stopped = False
        self.c = 0
        self.uniques = set()
        pass

    def stop(self):
        self.stopped = True
        self.put(None)

    def put(self, data: Union[FrameData, None]):
        if data is None:
            self.queue.put(data)
            return
        if data.seqId in self.uniques:
            return
        self.uniques.add(data.seqId)
        self.queue.put(data)

    def _get(self) -> Union[FrameData, None]:
        return self.queue.get()

    def visualize(self):
        video_stream = cv2.VideoCapture(self.video_file)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        if self.skip_frames:
            wrapped_stream = every_nth_frame(video_stream, self.frame_interval)
            interval = 1
            fps = fps / self.frame_interval
        else:
            wrapped_stream = every_nth_frame(video_stream, 1)
            interval = self.frame_interval
        wait_key_value = int(1000 / fps)

        frame_size = (
            int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        out_stream = None
        if self.save_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_stream = cv2.VideoWriter(self.save_file, fourcc, fps, frame_size)
            out_stream.set(cv2.VIDEOWRITER_PROP_QUALITY, 1)

        i = -1
        window_name = "visualization"
        while not self.stopped:
            i += 1
            (grabbed, frame) = next(wrapped_stream)
            if not grabbed:
                break
            if self.draw_frame_number:
                point = (frame.shape[1] - 100, 50)
                draw_text_with_outline(frame, point, str(i), 1)

            if not self.skip_frames and i % interval != 0:
                if self.save_file:
                    out_stream.write(frame)
                    continue
                cv2.imshow(window_name, frame)
                cv2.waitKey(wait_key_value)

            frame_data: FrameData = self._get()
            if frame_data is None:
                if self.save_file:
                    out_stream.write(frame)
                    continue
                cv2.imshow(window_name, frame)
                cv2.waitKey(wait_key_value)
                continue

            draw_frame_data(frame, frame_data, self.limb_pairs, self.prep_keypoints, True, False, False, True)

            if self.save_file:
                out_stream.write(frame)
                continue
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait_key_value)
        if self.save_file:
            out_stream.release()
        video_stream.release()
        if not self.save_file:
            cv2.destroyWindow(window_name)

    def run_visualize(self) -> Thread:
        self.vis_thread = Thread(
            target=self.visualize,
        )
        self.vis_thread.start()
        return self.vis_thread
