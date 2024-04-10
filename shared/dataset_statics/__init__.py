from shared.dataset_statics import ntu_splits, ut_splits, ntu_actions, ut_actions

split_map = {
    "ntu_xsub": ntu_splits.ntu_xsub_split,
    "ntu_xview": ntu_splits.ntu_xview_split,
    "ntu120_xset": ntu_splits.ntu120_xset_split,
    "ntu120_xsub": ntu_splits.ntu120_xsub_split,
    "ntu_mutual_xsub": ntu_splits.ntu_mutual_xsub_split,
    "ntu_mutual_xview": ntu_splits.ntu_mutual_xview_split,
    "ntu120_mutual_xset": ntu_splits.ntu120_mutual_xset_split,
    "ntu120_mutual_xsub": ntu_splits.ntu120_mutual_xsub_split,
    "ntu_whole_train": ntu_splits.ntu_whole_train,
    "ntu_whole_test": ntu_splits.ntu_whole_test,
    "ntu120_whole_train": ntu_splits.ntu120_whole_train,
    "ntu120_whole_test": ntu_splits.ntu120_whole_test,
    "ut_whole_train": ut_splits.ut_whole_train,
    "ut_whole_test": ut_splits.ut_whole_test,
}
all_splits = [x for x in split_map.keys()]

actions_maps = {
    "ntu": ntu_actions.ntu_actions,
    "ntu120": ntu_actions.all_actions,
    "ntu_mutual": ntu_actions.ntu_mutual,
    "ntu120_mutual": ntu_actions.mutual,
    "ut": ut_actions.all_actions
}


def zero_adjust_map(mapping: dict):
    keys = sorted(list(mapping.keys()))
    return {keys.index(x): y for x, y in mapping.items()}


adjusted_actions_maps = {x: zero_adjust_map(y) for x, y in actions_maps.items()}
