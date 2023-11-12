from __future__ import annotations

import numpy as np

from shared.helpers import tuple_list_to_dict

UT_ACTIONS = set([i for i in range(6)])


def ut_split_factory(param_key: str, train_set: set[int]):
    ACTION_SET = UT_ACTIONS

    # subject, camera, action
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

        train_split = tuple_list_to_dict(train_split, keys)
        train_split['skeleton_type'] = data[0][3]
        train_split['im_shape'] = data[0][4]
        test_split = tuple_list_to_dict(test_split, keys)
        test_split['skeleton_type'] = data[0][3]
        test_split['im_shape'] = data[0][4]
        return train_split, test_split

    return split


ut_whole_train = ut_split_factory("action", UT_ACTIONS)
ut_whole_test = ut_split_factory("subject", set())
