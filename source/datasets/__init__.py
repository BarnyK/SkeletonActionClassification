import datasets.ntu_splits
from datasets.transform_wrappers import TransformsList, TransformsDict, TransformsNameList

split_map = {
    "ntu_xsub": ntu_splits.ntu_xsub_split,
    "ntu_xview": ntu_splits.ntu_xview_split,
    "ntu120_xset": ntu_splits.ntu120_xset_split,
    "ntu120_xsub": ntu_splits.ntu120_xsub_split,
    "ntu_mutual_xsub": ntu_splits.ntu_mutual_xsub_split,
    "ntu_mutual_xview": ntu_splits.ntu_mutual_xview_split,
    "ntu120_mutual_xset": ntu_splits.ntu120_mutual_xset_split,
    "ntu120_mutual_xsub": ntu_splits.ntu120_mutual_xsub_split,
}
all_splits = [x for x in split_map.keys()]
