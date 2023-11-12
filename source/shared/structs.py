import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import numpy as np

from shared.dataset_info import DatasetInfo


@dataclass
class WindowProperties:
    length: int
    interlace: int
    step: int


@dataclass
class NtuNameData:
    set: int
    camera: int
    person: int
    replication: int
    action: int

    def to_dict(self):
        return self.__dict__


@dataclass
class Body:
    poseXY: np.ndarray
    poseConf: np.ndarray
    box: Optional[np.ndarray] = None
    boxConf: Optional[np.ndarray] = None
    poseXYZ: Optional[np.ndarray] = None
    tid: Optional[int] = None

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class FrameData:
    seqId: int
    count: int
    bodies: List[Body]

    def __post_init__(self):
        def assert_body(x) -> Body:
            if isinstance(x, dict):
                return Body(**x)
            elif isinstance(x, Body):
                return x

        self.bodies = [assert_body(x) for x in self.bodies]

    def to_dict(self) -> dict:
        return {
            "seqId": self.seqId,
            "count": self.count,
            "bodies": [x.to_dict() for x in self.bodies],
        }


@dataclass
class SkeletonData:
    source: str
    type: str
    dataset_info: DatasetInfo
    video_file: str
    length: int
    frames: List[FrameData]
    lengthB: Optional[int] = None
    original_image_shape: Tuple[int, int] = None
    frame_interval: int = 1

    def __post_init__(self):
        if isinstance(self.dataset_info, dict):
            self.dataset_info = DatasetInfo(**self.dataset_info)

        def assert_frame_data(x) -> FrameData:
            if isinstance(x, dict):
                return FrameData(**x)
            elif isinstance(x, FrameData):
                return x

        self.frames = [assert_frame_data(x) for x in self.frames]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "type": self.type,
            "dataset_info": self.dataset_info.to_dict(),
            "video_file": self.video_file,
            "length": self.length,
            "frames": [fd.to_dict() for fd in self.frames],
            "lengthB": self.lengthB,
            "original_image_shape": self.original_image_shape,
            "frame_interval": self.frame_interval,
        }

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def load(filename) -> 'SkeletonData':
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return SkeletonData(**data)

    def get_all_tids(self):
        return {body.tid for frame in self.frames for body in frame.bodies}

    def get_all_bodies_for_tid(self, tid: int) -> List[Body]:
        ret = []
        for frame in self.frames:
            for body in frame.bodies:
                if body.tid == tid:
                    ret.append(body)
                    break
        return ret

    def get_joints_count(self):
        for body in [body for frame in self.frames for body in frame.bodies]:
            return body.poseXY.shape[0]
        return 0

    def get_points_shape(self):
        for body in [body for frame in self.frames for body in frame.bodies]:
            return body.poseXY.shape
        raise ValueError("No bodies found")

    def to_matrix(self) -> Union[np.ndarray, None]:
        tids = self.get_all_tids()
        if len(tids) == 0:
            return None
        M, T, (V, N) = len(tids), self.lengthB, self.get_points_shape()

        mat = np.zeros((M, T, V, N), dtype=np.float32)
        for i, tid in enumerate(tids):
            tid_bodies = [body for frame in self.frames for body in frame.bodies]
            tid_bodies = [body for body in tid_bodies if body.tid == tid]
            if len(tid_bodies) != T:
                raise ValueError(f"Missing bodies for tid:{tid}")
            mat[i, ...] = np.stack([body.poseXY for body in tid_bodies])

        return mat

    def no_bodies(self):
        if [body for frame in self.frames for body in frame.bodies]:
            return False
        return True

    def to_matrix_confidences(self) -> np.ndarray:
        tids = self.get_all_tids()
        M, T, (V, _) = len(tids), self.lengthB, self.get_points_shape()

        mat = np.zeros((M, T, V, 1), dtype=np.float32)
        for i, tid in enumerate(tids):
            tid_bodies = [body for frame in self.frames for body in frame.bodies]
            tid_bodies = [body for body in tid_bodies if body.tid == tid]
            if len(tid_bodies) != T:
                raise ValueError(f"Missing bodies for tid:{tid}")
            mat[i, ...] = np.stack([body.poseConf for body in tid_bodies])

        return mat

    def get_frame_body(self, frame_id: int, body_tid: int):
        frame = self.frames[frame_id]
        for body in frame.bodies:
            if body.tid == body_tid:
                return body
        return None

    def remove_bodies_for_tid(self, tid: int):
        for frame in self.frames:
            frame.bodies = [body for body in frame.bodies if body.tid != tid]

    def get_all_bodies_for_tid_with_seq(self, tid: int):
        return [(frame.seqId, body) for frame in self.frames for body in frame.bodies if body.tid == tid]
