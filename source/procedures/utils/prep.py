from __future__ import annotations

import time

from preprocessing import skeleton_filters
from preprocessing.keypoint_fill import keypoint_fill
from preprocessing.nms import single_frame_nms, nms
from preprocessing.tracking import select_by_size, select_by_confidence, select_by_order, pose_track, \
    select_tracks_by_motion, assign_tids_by_order, ntu_track_selection, select_by_order_frame, \
    select_by_confidence_frame, select_by_size_frame
from procedures.config import GeneralConfig, PreprocessConfig
from shared import FrameData, SkeletonData
from shared.skeletons import ntu_coco


def preprocess_per_frame(frame: FrameData, gcfg: GeneralConfig):
    cfg = gcfg.prep_config
    if cfg.transform_to_combined:
        frame = ntu_coco.from_frame(frame, gcfg.skeleton_type)
    if cfg.use_box_conf:
        skeleton_filters.remove_bodies_by_box_confidence_frame(frame, cfg.box_conf_threshold)
    if cfg.use_max_pose_conf:
        skeleton_filters.remove_by_max_possible_pose_confidence_frame(frame, cfg.max_pose_conf_threshold)
    if cfg.use_nms:
        single_frame_nms(frame, True)
    if cfg.use_size_selection:
        select_by_size_frame(frame, cfg.max_body_count)
    elif cfg.use_confidence_selection:
        select_by_confidence_frame(frame, cfg.max_body_count)
    elif cfg.use_order_selection:
        select_by_order_frame(frame, cfg.max_body_count)
    return frame


def preprocess_data_rest(data: SkeletonData, cfg: PreprocessConfig):
    if cfg.use_tracking:
        pose_track(data.frames,
                   threshold=cfg.pose_tracking_threshold,
                   width_ratio=cfg.pose_tracking_width_ratio,
                   height_ratio=cfg.pose_tracking_height_ratio)
        if cfg.use_motion_selection:
            select_tracks_by_motion(data, cfg.max_body_count)
    else:
        assign_tids_by_order(data)
    keypoint_fill(data, cfg.keypoint_fill_type)
    return data


def preprocess_data_ap(data: SkeletonData, cfg: PreprocessConfig) -> SkeletonData:
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)
    if cfg.use_box_conf:
        skeleton_filters.remove_bodies_by_box_confidence(data, cfg.box_conf_threshold, cfg.box_conf_max_total,
                                                         cfg.box_conf_max_frames)
    if cfg.use_max_pose_conf:
        skeleton_filters.remove_by_max_possible_pose_confidence(data, cfg.max_pose_conf_threshold)
    if cfg.use_nms:
        nms(data, True)

    if cfg.use_size_selection:
        select_by_size(data, cfg.max_body_count)
    elif cfg.use_confidence_selection:
        select_by_confidence(data, cfg.max_body_count)
    elif cfg.use_order_selection:
        select_by_order(data, cfg.max_body_count)

    if cfg.use_tracking:
        pose_track(data.frames,
                   threshold=cfg.pose_tracking_threshold,
                   width_ratio=cfg.pose_tracking_width_ratio,
                   height_ratio=cfg.pose_tracking_height_ratio)
        if cfg.use_motion_selection:
            select_tracks_by_motion(data, cfg.max_body_count)
    else:
        assign_tids_by_order(data)
    keypoint_fill(data, cfg.keypoint_fill_type)
    return data


def preprocess_data_ap_timed(data: SkeletonData, cfg: PreprocessConfig):
    st = time.time()
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)
    transform_time = time.time()
    if cfg.use_box_conf:
        skeleton_filters.remove_bodies_by_box_confidence(data, cfg.box_conf_threshold, cfg.box_conf_max_total,
                                                         cfg.box_conf_max_frames)
    if cfg.use_max_pose_conf:
        skeleton_filters.remove_by_max_possible_pose_confidence(data, cfg.max_pose_conf_threshold)
    filter_time = time.time()
    if cfg.use_nms:
        nms(data, True)
    nms_time = time.time()

    if cfg.use_size_selection:
        select_by_size(data, cfg.max_body_count)
    elif cfg.use_confidence_selection:
        select_by_confidence(data, cfg.max_body_count)
    elif cfg.use_order_selection:
        select_by_order(data, cfg.max_body_count)

    if cfg.use_tracking:
        pose_track(data.frames,
                   threshold=cfg.pose_tracking_threshold,
                   width_ratio=cfg.pose_tracking_width_ratio,
                   height_ratio=cfg.pose_tracking_height_ratio)
        if cfg.use_motion_selection:
            select_tracks_by_motion(data, cfg.max_body_count)
    else:
        assign_tids_by_order(data)
    track_time = time.time()
    keypoint_fill(data, cfg.keypoint_fill_type)
    fill_time = time.time()
    return data, [transform_time-st, filter_time-transform_time, nms_time-filter_time, track_time-nms_time, fill_time-track_time]


def preprocess_data_ntu(data: SkeletonData, cfg: PreprocessConfig):
    if cfg.transform_to_combined:
        data = ntu_coco.from_skeleton_data(data)

    if data.no_bodies():
        return data

    if cfg.use_3d_points:
        for tid in data.get_all_tids():
            for body in data.get_all_bodies_for_tid(tid):
                body.poseXY = body.poseXYZ

    ntu_track_selection(data, cfg.max_body_count)
    keypoint_fill(data, cfg.keypoint_fill_type)
    return data
