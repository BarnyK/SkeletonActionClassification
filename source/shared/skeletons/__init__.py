from shared.skeletons import coco

drawn_limbs_map = {
    "coco17": coco.drawn_limbs,
}

draw_preparation_func_map = {
    "coco17": coco.prepare_draw_keypoints
}

bones_map = {
    "coco17": coco.bones
}

angles_map = {
    "coco17": coco.angles
}

center_pos_map = {
    "coco17": coco.center_position_func
}

spine_size_func_map = {
    "coco17": coco.spine_size
}

align_func_map = {
    "coco17": coco.alignment_keypoint_value
}
