from shared.skeletons import coco
from shared.skeletons import ntu

drawn_limbs_map = {
    "coco17": coco.drawn_limbs,
    "ntu": ntu.drawn_limbs
}

draw_preparation_func_map = {
    "coco17": coco.prepare_draw_keypoints,
    "ntu": ntu.prepare_draw_keypoints
}

bones_map = {
    "coco17": coco.bones,
    "ntu": ntu.bones
}

angles_map = {
    "coco17": (coco.angles, coco.angles_to_zero),
    "ntu": (ntu.angles, coco.angles_to_zero)
}

center_pos_map = {
    "coco17": coco.center_position_func,
    "ntu": ntu.center_position_func
}

spine_size_func_map = {
    "coco17": coco.spine_size,
    "ntu": ntu.spine_size
}

align_func_map = {
    "coco17": coco.alignment_keypoint_value,
    "ntu": ntu.alignment_keypoint_value
}
