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


def ntu_split_factory(ntu_ver: int, param_key: str, train_set: set[int]):
    if ntu_ver == 60:
        ACTION_SET = NTU_ACTIONS
    elif ntu_ver == 120:
        ACTION_SET = NTU120_ACTIONS

    def split(data: list[list[dict, np.ndarray, np.ndarray]]):
        train_split = []
        test_split = []
        for dataset_info, poseXY, poseConf in data:
            action_id = dataset_info['info']['action']
            if action_id not in ACTION_SET:
                continue
            param_id = dataset_info['info'][param_key]
            if param_id in train_set:
                train_split.append("")
            else:
                test_split.append("")

        train_split = __tuplist_to_dict(train_split, [])
        test_split = __tuplist_to_dict(test_split, [])
        return train_split, test_split

    return split


ntu_xsub = ntu_split_factory(60, "person", XSUB)
ntu_xview = ntu_split_factory(60, "view", XVIEW)
ntu120_xset = ntu_split_factory(120, "set", XSET)
ntu120_xsub = ntu_split_factory(120, "person", XSUB)
