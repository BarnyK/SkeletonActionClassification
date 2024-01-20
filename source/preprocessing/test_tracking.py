import os
from unittest import TestCase

from preprocessing import skeleton_filters
from preprocessing.nms import nms
from preprocessing.tracking import select_tracks_by_motion, pose_track, assign_tids_by_order, select_by_size, \
    select_by_confidence, select_by_order
from shared.structs import SkeletonData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test_Tracking(TestCase):
    def test_select_tracks_by_motion_single(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A010.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        skeleton_filters.remove_bodies_by_box_confidence(data, 0.7)
        skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
        nms(data)
        pose_track(data.frames, threshold=90)
        select_tracks_by_motion(data, 1)
        assert len(data.get_all_tids()) == 1

    def test_select_tracks_by_motion_mutual(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A057.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        skeleton_filters.remove_bodies_by_box_confidence(data, 0.7)
        skeleton_filters.remove_by_max_possible_pose_confidence(data, 0.5)
        nms(data)
        pose_track(data.frames, threshold=90)
        select_tracks_by_motion(data, 2)
        assert len(data.get_all_tids()) == 2

    def test_assign_tids_by_order(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A057.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        assign_tids_by_order(data)

    def test_select_by_confidence(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A057.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        select_by_confidence(data)

    def test_select_by_size(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A057.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        select_by_size(data)

    def test_select_by_order(self):
        data_path = os.path.join(THIS_DIR, os.pardir, "sample_files/S001C001P001R001A057.coco17.apskel.pkl")
        data = SkeletonData.load(data_path)
        select_by_order(data)
