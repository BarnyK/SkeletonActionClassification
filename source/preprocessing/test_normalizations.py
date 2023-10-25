import os
from unittest import TestCase

import numpy as np

from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.normalizations import screen_normalization, relative_normalization, spine_normalization, \
    mean_spine_normalization, joints_to_bones, to_angles, to_motion, bone_angles, relative_joints
from preprocessing.tracking import pose_track, select_tracks_by_motion
from shared.structs import SkeletonData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(TestCase):
    vis = False

    def setUp(cls):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S009C003P025R001A060.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        skeleton_filters.remove_bodies_by_box_confidence(data, 0.7)
        skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
        nms(data, True)
        pose_track(data.frames, threshold=90)
        select_tracks_by_motion(data, 2)
        keypoint_fill(data, "interpolation")
        mat = data.to_matrix()
        cls.data = data
        cls.mat = mat

    def test_screen_normalization(self):
        mat = screen_normalization(self.mat, self.data.original_image_shape)

    def test_relative_normalization(self):
        mat = relative_normalization(self.mat)

    def test_spine_normalization(self):
        mat = spine_normalization(self.mat, self.data.type)

    def test_mean_spine_normalization(self):
        mat = mean_spine_normalization(self.mat, self.data.type)

    def body_stats(self, skeleton_data: SkeletonData, name: str):
        all_bodies = [body for frame in skeleton_data.frames for body in frame.bodies]
        stack = np.stack([body.poseXY for body in all_bodies], axis=0)
        mean = np.mean(stack)
        x_min, x_max = stack[:, :, 0].min(), stack[:, :, 0].max()
        y_min, y_max = stack[:, :, 1].min(), stack[:, :, 1].max()
        var = np.var(stack)
        print(name, mean, var, x_min, x_max, y_min, y_max)

    def test_to_bones(self):
        mat = joints_to_bones(self.mat, self.data.type)

    def test_to_angles(self):
        mat = to_angles(self.mat, self.data.type)

    def test_to_motion(self):
        mat = to_motion(self.mat)

    def test_bone_to_motion(self):
        mat = joints_to_bones(self.mat, self.data.type)
        mat = to_motion(mat)

    def test_bone_angles(self):
        mat = joints_to_bones(self.mat, self.data.type)
        mat = bone_angles(mat)

    def test_relative_joints(self):
        mat = relative_joints(self.mat, self.data.type)

