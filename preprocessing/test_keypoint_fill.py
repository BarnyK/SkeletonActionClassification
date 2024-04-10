import os
import time
from unittest import TestCase

import numpy as np

from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion
from shared.structs import SkeletonData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(TestCase):
    data: SkeletonData = None

    def setUp(cls):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S009C003P025R001A060.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        skeleton_filters.remove_bodies_by_box_confidence(data, 0.7)
        skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
        nms(data)
        pose_track(data.frames, threshold=90)
        select_tracks_by_motion(data, 2)
        cls.data = data

    def test_interpolation_fill(self):
        keypoint_fill(self.data, "interpolation")

    def test_mice_fill(self):
        keypoint_fill(self.data, "mice")

    def test_none_fill(self):
        keypoint_fill(self.data, "none")

    def test_knn_fill(self):
        keypoint_fill(self.data, "knn")

    def test_some(self):
        keypoint_fill(self.data, "interpolation")
        st = time.time()
        tids = self.data.get_all_tids()
        tids_map = {tid: i for i, tid in enumerate(tids)}
        shape = (len(tids), self.data.length, 17, 2)
        matrix = np.zeros(shape)
        for fi, frame in enumerate(self.data.frames):
            for body in frame.bodies:
                matrix[tids_map[body.tid], fi, :, :] = body.poseXY
        et = time.time()
