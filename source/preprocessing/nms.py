import time

import numpy as np

from shared.structs import SkeletonData, FrameData, Body

mu = 1.7
delta1 = 1
delta2 = 2.65
gamma = 22.48
matchThreds = 5
scoreThreds = 0.3


def pose_nms(
        boxes: np.ndarray,
        box_scores: np.ndarray,
        pose_joints: np.ndarray,
        pose_scores: np.ndarray,
):
    # pose_conf: [n x 17]
    # box_conf: [n x 1]
    # pose_loc: [n x 17 x 2]
    # box_loc: [n x 4]

    candidates = np.arange(boxes.shape[0])
    nums_candidates = len(candidates)

    pose_joints = np.reshape(pose_joints, [pose_joints.shape[0], -1, 2])

    merge_ids = {}
    choose_set = []
    while candidates.size > 0:
        choose_idx = np.argmax(box_scores[candidates])
        choose = candidates[choose_idx]

        keypoint_width = get_keypoint_width(pose_joints[choose])
        simi = get_pose_dis(
            choose_idx,
            pose_scores[candidates],
            pose_joints[candidates],
            keypoint_width=keypoint_width,
            delta1=delta1,
            delta2=delta2,
            mu=mu,
        )
        num_match_keypoints, _ = PCK_match(
            choose_idx, pose_joints[candidates], keypoint_width
        )

        delete_ids = np.arange(candidates.shape[0])[
            (simi > gamma) | (num_match_keypoints >= matchThreds)
            ]

        assert delete_ids.size > 0  # at least itself
        if delete_ids.size == 0:
            delete_ids = choose_idx

        merge_ids[choose] = candidates[delete_ids]
        choose_set.append(choose)

        # force to delete itself
        candidates = np.delete(candidates, np.append(delete_ids, choose_idx))

    # merge poses
    result_detections = []
    new_boxes = []
    new_box_scores = []
    new_poseXYs = []
    new_pose_scores = []
    for root_pose_idx in choose_set:
        simi_poses_idx = merge_ids[root_pose_idx]
        max_score = np.max(pose_scores[simi_poses_idx, :])
        if max_score < scoreThreds:
            continue
        keypoint_width = get_keypoint_width(pose_joints[root_pose_idx])

        merge_poses, merge_score = merge_pose(
            pose_joints[root_pose_idx],
            pose_joints[simi_poses_idx],
            pose_scores[simi_poses_idx],
            keypoint_width,
        )
        max_score = np.max(merge_poses)
        if max_score < scoreThreds:
            continue

        new_boxes.append(boxes[root_pose_idx])
        new_box_scores.append(box_scores[root_pose_idx])
        new_poseXYs.append(merge_poses)
        new_pose_scores.append(merge_score)

    return new_boxes, new_box_scores, new_poseXYs, new_pose_scores


def get_keypoint_width(pose_loc):
    """
    :param pose_loc: [17 x 2]
    :return:
    """
    alpha = 0.1
    body_width = max(
        np.max(pose_loc[:, 0]) - np.min(pose_loc[:, 0]),
        np.max(pose_loc[:, 1]) - np.min(pose_loc[:, 1]),
    )
    keypoint_width = body_width * alpha
    return keypoint_width


def get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width, delta1, delta2, mu):
    """
    <class 'numpy'>
    :param choose_idx:
    :param pose_conf: [n x 17]
    :param pose_loc: [n x 17 x 2]
    :return:
    """
    num_pose = pose_conf.shape[0]
    target_pose_conf = pose_conf[choose_idx]

    dist = (
            np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1))
            / keypoint_width
    )
    mask = dist <= 1

    score_dists = np.zeros([num_pose, pose_conf.shape[1]])
    pose_conf_tile = np.tile(target_pose_conf, [pose_conf.shape[0], 1])
    score_dists[mask] = np.tanh(pose_conf_tile[mask] / delta1) * np.tanh(
        pose_conf[mask] / delta1
    )

    point_dists = np.exp((-1) * dist / delta2)
    return np.sum(score_dists, axis=1) + mu * np.sum(point_dists, axis=1)


def PCK_match(choose_idx, pose_loc, keypoint_width):
    """
    :param choose_idx:
    :param pose_loc: [n x 17 x 2]
    :param keypoint_width: 0.1 x body_width
    :return:
    """
    dist = np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1))
    num_match_keypoints = np.sum(dist / min(keypoint_width, 7) <= 1, axis=1)
    face_index = np.zeros(dist.shape)
    face_index[:, :5] = 1

    face_match_keypoints = np.sum((dist / 10 <= 1) & (face_index == 1), axis=1)
    return num_match_keypoints, face_match_keypoints


def merge_pose(root_pose, cluster_pose_loc, cluster_pose_conf, keypoint_width):
    """
    root_pose must be in cluster_pose_loc.
    :param root_pose: [17 x 2]
    :param cluster_pose_loc: [n x 17 x 2]
    :param cluster_pose_conf: [n x 17]
    :param keypoint_width: 0.1 x body_width
    :return:
    """
    dist = (
            np.sqrt(np.sum(np.square(cluster_pose_loc - root_pose), axis=-1))
            / keypoint_width
    )
    keypoint_width = min(keypoint_width, 15)
    mask = dist <= keypoint_width
    # final_pose = np.zeros([17,2]); final_scores = np.zeros(17)

    pose_loc_t = cluster_pose_loc * np.expand_dims(mask, axis=-1)
    pose_conf_t = cluster_pose_conf * mask

    weighted_pose_conf = pose_conf_t * 1.0 / np.sum(pose_conf_t, axis=0)
    result_pose_conf = weighted_pose_conf * pose_conf_t
    final_scores = np.sum(result_pose_conf, axis=0)

    weighted_pose_loc = pose_loc_t[:, :, :] * np.expand_dims(weighted_pose_conf, -1)
    final_pose = np.sum(weighted_pose_loc, axis=0)

    return final_pose, final_scores


alpha = 0.1


def get_parametric_distance(
        i, all_preds, keypoint_scores, ref_dist,
):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(np.power(pick_preds[np.newaxis, :] - all_preds, 2), axis=2))
    mask = dist <= 1

    kp_nums = all_preds.shape[1]

    # Define a keypoints distance
    score_dists = np.zeros((all_preds.shape[0], kp_nums))
    keypoint_scores = keypoint_scores.squeeze()
    if keypoint_scores.ndim == 1:
        keypoint_scores = np.expand_dims(keypoint_scores, 0)
    if pred_scores.ndim == 1:
        pred_scores = np.expand_dims(pred_scores, 1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = np.tile(pred_scores, (1, all_preds.shape[0])).transpose((1, 0))

    score_dists[mask] = np.tanh(pred_scores[mask] / delta1) * np.tanh(
        keypoint_scores[mask] / delta1
    )

    point_dist = np.exp((-1) * dist / delta2)
    final_dist = np.sum(score_dists, axis=1) + mu * np.sum(point_dist, axis=1)

    return final_dist


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    """
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [kp_num, 2]
        cluster_preds:  redundant poses         -- [n, kp_num, 2]
        cluster_scores: redundant poses score   -- [n, kp_num, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [kp_num, 2]
        final_score:    merged score            -- [kp_num]
    """
    dist = np.sqrt(np.sum(np.power(ref_pose[np.newaxis, :] - cluster_preds, 2), axis=2))

    kp_num = ref_pose.shape[0]
    ref_dist = min(ref_dist, 15)

    mask = dist <= ref_dist

    final_pose = np.zeros((kp_num, 2))
    final_score = np.zeros(kp_num)

    if cluster_preds.ndim == 2:
        cluster_preds = np.expand_dims(cluster_preds, 0)
        cluster_scores = np.expand_dims(cluster_scores, 0)
    if mask.ndim == 1:
        mask = np.expand_dims(mask, 0)

    # Weighted Merge
    masked_scores = cluster_scores * np.expand_dims(mask, 2)
    normed_scores = masked_scores / np.sum(masked_scores, axis=0)

    final_pose = np.multiply(cluster_preds, np.tile(normed_scores, (1, 1, 2))).sum(
        axis=0
    )
    final_score = np.multiply(masked_scores, normed_scores).sum(axis=0)
    return final_pose, final_score


def PCK_match2(pick_pred, all_preds, ref_dist):
    dist = np.sqrt(np.sum(np.power(pick_pred[np.newaxis, :] - all_preds, 2), axis=2))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = np.sum(dist / ref_dist <= 1, axis=1)

    return num_match_keypoints


def pose_nms_body(
        bboxes: np.ndarray,
        bbox_scores: np.ndarray,
        bbox_ids: np.ndarray,
        pose_preds: np.ndarray,
        pose_scores,
        areaThres=0,
):
    """
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n, 1)
    bbox_ids:       bbox tracking ids list (n, 1)
    pose_preds:     pose locations list (n, kp_num, 2)
    pose_scores:    pose scores list    (n, kp_num, 1)
    """
    # global ori_pose_preds, ori_pose_scores, ref_dists
    bbox_ids = np.arange(pose_preds.shape[0], like=bbox_scores)
    pose_scores[pose_scores == 0] = 1e-5
    kp_nums = pose_preds.shape[1]
    (
        res_bboxes,
        res_bbox_scores,
        res_bbox_ids,
        res_pose_preds,
        res_pose_scores,
        res_pick_ids,
    ) = ([], [], [], [], [], [])

    ori_bboxes = bboxes.copy()
    ori_bbox_scores = bbox_scores.copy()
    ori_bbox_ids = bbox_ids.copy()
    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()[..., np.newaxis]

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(axis=1)

    human_ids = np.arange(nsamples)
    mask = np.ones(len(human_ids)).astype(bool)

    # Do pPose-NMS
    pick = []
    merge_ids = []
    while mask.any():
        tensor_mask = mask == True
        # Pick the one with highest score
        pick_id = np.argmax(human_scores[tensor_mask])
        pick.append(human_ids[mask][pick_id])

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[mask][pick_id]]
        simi = get_parametric_distance(
            pick_id, pose_preds[tensor_mask], pose_scores[tensor_mask], ref_dist
        )
        num_match_keypoints = PCK_match2(
            pose_preds[tensor_mask][pick_id], pose_preds[tensor_mask], ref_dist
        )

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = np.arange(human_scores[tensor_mask].shape[0])[
            ((simi > gamma) | (num_match_keypoints >= matchThreds))
        ]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id

        merge_ids.append(human_ids[mask][delete_ids])
        newmask = mask[mask]
        newmask[delete_ids] = False
        mask[mask] = newmask

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bboxes_pick = ori_bboxes[pick]
    bbox_ids_pick = ori_bbox_ids[pick]
    # final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    # final_result = [item for item in final_result if item is not None]

    for j in range(len(pick)):
        ids = np.arange(kp_nums)
        max_score = np.max(scores_pick[j, ids, 0])

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j],
            ori_pose_preds[merge_id],
            ori_pose_scores[merge_id],
            ref_dists[pick[j]],
        )

        max_score = np.max(merge_score[ids])
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])
        bbox = bboxes_pick[j]
        bbox_score = bbox_scores_pick[j]

        if 1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres:
            continue

        res_bboxes.append(bbox)
        res_bbox_scores.append(bbox_score)
        res_pose_preds.append(merge_pose)
        res_pose_scores.append(merge_score)
        res_pick_ids.append(pick[j])

    return res_bboxes, res_bbox_scores, res_pose_preds, res_pose_scores, res_pick_ids


def single_frame_nms(frame: FrameData, ap: bool = True):
    # boxes, boxConf, poseXY, poseConf
    if len(frame.bodies) == 0:
        return
    boxes = np.stack([x.box for x in frame.bodies], 0)
    boxConfs = np.stack([x.boxConf for x in frame.bodies], 0)
    poseXYs = np.stack([x.poseXY for x in frame.bodies], 0)
    poseConfs = np.stack([x.poseConf for x in frame.bodies], 0).squeeze(2)

    if ap:
        new_boxes, new_boxConfs, new_poseXYs, new_poseConfs, _ = pose_nms_body(
            boxes, boxConfs, [], poseXYs, poseConfs, 0
        )
    else:
        new_boxes, new_boxConfs, new_poseXYs, new_poseConfs = pose_nms(
            boxes, boxConfs, poseXYs, poseConfs
        )
    new_bodies = [
        Body(new_poseXYs[i], new_poseConfs[i], new_boxes[i], new_boxConfs[i])
        for i in range(len(new_boxes))
    ]
    frame.bodies = new_bodies


import concurrent.futures


def nms(data: SkeletonData, ap: bool):
    for frame in data.frames:
        single_frame_nms(frame, ap)


def concurrent_nms(data: SkeletonData, ap: bool):
    # it's slower than loops because ye
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use executor to run single_frame_nms for each frame concurrently
        futures = [executor.submit(single_frame_nms, frame, ap) for frame in data.frames]

        # Wait for all futures to complete
        concurrent.futures.wait(futures)


def test_nms(skeleton_file):
    data = SkeletonData.load(skeleton_file)
    # visualize(frames, frames.video_file, 1000//30)
    # frames = SkeletonData.load(skeleton_file)
    # nms(frames, False)
    # # visualize(frames, frames.video_file, 1000//30, "The other one")

    data = SkeletonData.load(skeleton_file)
    st = time.time()
    nms(data, True)
    et = time.time()
    print(et - st)

    data = SkeletonData.load(skeleton_file)
    st = time.time()
    concurrent_nms(data, True)
    et = time.time()
    print(et - st)
    # visualize(frames, frames.video_file, 1000//30, "Alphapose ppose nms")


if __name__ == "__main__":
    test_nms(
        "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_halpe/S009C003P025R001A060_rgb.halpe.apskel.pkl")
    test_nms(
        "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_halpe/S001C001P001R001A053_rgb.halpe.apskel.pkl")
    test_nms(
        "/media/barny/SSD4/MasterThesis/Data/alphapose_skeletons/ntu_halpe/S001C002P001R001A037_rgb.halpe.apskel.pkl")
