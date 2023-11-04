from shared.skeletons import coco
from shared.skeletons import ntu
from shared.skeletons import ntu_coco
from shared.skeletons.ntu_coco import from_skeleton_data

drawn_limbs_map = {
    "coco17": coco.drawn_limbs,
    "ntu": ntu.drawn_limbs,
    "ntu_coco": ntu_coco.drawn_limbs
}

draw_preparation_func_map = {
    "coco17": coco.prepare_draw_keypoints,
    "ntu": ntu.prepare_draw_keypoints,
    "ntu_coco": ntu_coco.prepare_draw_keypoints,
}

bones_map = {
    "coco17": coco.bones,
    "ntu": ntu.bones,
    "ntu_coco": ntu_coco.bones,
}

angles_map = {
    "coco17": (coco.angles, coco.angles_to_zero),
    "ntu": (ntu.angles, coco.angles_to_zero),
    "ntu_coco": (ntu_coco.angles, coco.angles_to_zero),
}

center_pos_map = {
    "coco17": coco.center_position_func,
    "ntu": ntu.center_position_func,
    "ntu_coco": ntu_coco.center_position_func,
}

spine_size_func_map = {
    "coco17": coco.spine_size,
    "ntu": ntu.spine_size,
    "ntu_coco": ntu_coco.spine_size,
}

align_func_map = {
    "coco17": coco.alignment_keypoint_value,
    "ntu": ntu.alignment_keypoint_value,
    "ntu_coco": ntu_coco.alignment_keypoint_value,
}
