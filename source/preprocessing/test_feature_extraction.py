import os
from unittest import TestCase

import numpy as np

from preprocessing import skeleton_filters
from preprocessing.feature_extraction import joints_to_bones, to_angles, to_motion, bone_angles, relative_joints
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import nms
from preprocessing.tracking import pose_track, select_tracks_by_motion
from shared.structs import SkeletonData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestFeatures(TestCase):
    def setUp(cls):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S009C003P025R001A060.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        skeleton_filters.remove_bodies_by_box_confidence(data, 0.7)
        skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
        nms(data)
        pose_track(data.frames, threshold=90)
        select_tracks_by_motion(data, 2)
        keypoint_fill(data, "interpolation")
        mat = data.to_matrix()
        cls.data = data
        cls.mat = mat

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

    def test_swapped_to_bones(self):
        mat = joints_to_bones(self.mat, self.data.type)
        swapped_mat = np.stack([self.mat[1], self.mat[0]])
        mat2 = joints_to_bones(swapped_mat, self.data.type)
        swapped_mat2 = np.stack([mat2[1], mat2[0]])
        x = (mat - swapped_mat2)

    def test_swapped_to_angles(self):
        mat = to_angles(self.mat, self.data.type)
        swapped_mat = np.stack([self.mat[1], self.mat[0]])
        mat2 = to_angles(swapped_mat, self.data.type)
        swapped_mat2 = np.stack([mat2[1], mat2[0]])
        x = (mat - swapped_mat2)

    def test_swapped_to_motion(self):
        mat = to_motion(self.mat)
        swapped_mat = np.stack([self.mat[1], self.mat[0]])
        mat2 = to_motion(swapped_mat)
        swapped_mat2 = np.stack([mat2[1], mat2[0]])
        x = (mat - swapped_mat2)

    def test_swapped_bone_to_motion(self):
        mat = joints_to_bones(self.mat, self.data.type)
        mat = to_motion(mat)

    def test_swapped_bone_angles(self):
        mat = joints_to_bones(self.mat, self.data.type)
        mat = bone_angles(mat)

    def test_swapped_relative_joints(self):
        mat = relative_joints(self.mat, self.data.type)
        swapped_mat = np.stack([self.mat[1], self.mat[0]])
        mat2 = relative_joints(swapped_mat, self.data.type)
        swapped_mat2 = np.stack([mat2[1], mat2[0]])
        x = (mat - swapped_mat2)
