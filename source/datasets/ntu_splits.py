from __future__ import annotations

import numpy as np

NTU_ACTIONS = set([i + 1 for i in range(60)])
NTU120_ACTIONS = set([i + 1 for i in range(120)])
MUTUAL_ACTIONS = set([i for i in range(50, 61)] + [i for i in range(106, 121)])

XSUB = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
}
XVIEW = {2, 3}
XSET = set(range(2, 33, 2))


def __tuplist_to_dict(lista: list[tuple], keys: list[str]):
    res = {}
    for i, key in enumerate(keys):
        res[key] = [x[i] for x in lista]
    return res


def ntu_split_factory(ntu_ver: int, param_key: str, train_set: set[int], mutual: bool = False):
    if ntu_ver == 60:
        ACTION_SET = NTU_ACTIONS
    elif ntu_ver == 120:
        ACTION_SET = NTU120_ACTIONS
    else:
        raise ValueError(f"ntu_ver should be 60 or 120")
    if mutual:
        ACTION_SET = ACTION_SET & MUTUAL_ACTIONS

    def split(data: list[list[dict, np.ndarray, np.ndarray]]):
        train_split = []
        test_split = []
        keys = ["action", "dataset_info", "poseXY", "poseConf"]
        for dataset_info, poseXY, poseConf, *_ in data:
            action_id = dataset_info.info['action']
            if action_id not in ACTION_SET:
                continue
            param_id = dataset_info.info[param_key]
            tup_data = (action_id, dataset_info.to_filename(), poseXY, poseConf)
            if param_id in train_set:
                train_split.append(tup_data)
            else:
                test_split.append(tup_data)

        train_split = __tuplist_to_dict(train_split, keys)
        train_split['skeleton_type'] = data[0][3]
        train_split['im_shape'] = data[0][4]
        test_split = __tuplist_to_dict(test_split, keys)
        test_split['skeleton_type'] = data[0][3]
        test_split['im_shape'] = data[0][4]
        return train_split, test_split

    return split


ntu_xsub_split = ntu_split_factory(60, "person", XSUB)
ntu_xview_split = ntu_split_factory(60, "camera", XVIEW)
ntu120_xset_split = ntu_split_factory(120, "set", XSET)
ntu120_xsub_split = ntu_split_factory(120, "person", XSUB)

ntu_mutual_xsub_split = ntu_split_factory(60, "person", XSUB, True)
ntu_mutual_xview_split = ntu_split_factory(60, "camera", XVIEW, True)
ntu120_mutual_xset_split = ntu_split_factory(120, "set", XSET, True)
ntu120_mutual_xsub_split = ntu_split_factory(120, "person", XSUB, True)
