from __future__ import annotations

import pickle
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.sampler import Sampler
from datasets.transform_wrappers import TransformsDict, PoseTransform, TransformsList
from preprocessing.normalizations import no_norm
from shared.helpers import flatten_list


def solve_feature_transform_requirements(feature_list):
    required_transforms = set()
    queue = flatten_list(feature_list)[:]
    while queue:
        feature_name = queue.pop(0)
        required_transforms.add(feature_name)
        transform = TransformsDict[feature_name]
        required = transform.requires
        for req in required:
            if req.name not in required_transforms:
                queue.append(req.name)
            required_transforms.add(req.name)
    required_transforms = sorted(required_transforms, key=lambda x: TransformsList.index(TransformsDict[x]))
    return required_transforms


def transform_to_stgcn_input(feature_dictionary, feature_list):
    features = [feature_dictionary[k] for k in feature_list]
    features = np.concatenate(features, axis=-1)
    # Pad empty dimension
    if features.ndim == 4:
        M, T, V, C = features.shape
        if M == 1:
            padded = np.zeros((2, T, V, C), dtype=np.float32)
            padded[0, ...] = features
            features = padded
    elif features.ndim > 4:
        *R, M, T, V, C = features.shape
        if M == 1:
            padded = np.zeros((*R, 2, T, V, C), dtype=np.float32)
            padded[..., 0, np.newaxis, :, :, :] = features
            features = padded
    return features


def transform_to_tpgcn_input(feature_dictionary, feature_list, symmetry):
    og_features = feature_dictionary['joints']
    if og_features.ndim == 4:
        return create_tpgcn_branches(feature_dictionary, feature_list, og_features, symmetry)
    else:
        # Sampling creates new dimension
        out_data = []
        for s in range(og_features.shape[0]):
            new_features = create_tpgcn_branches(feature_dictionary, feature_list, og_features, symmetry, s)
            out_data.append(new_features)
        features = np.stack(out_data)
        return features


def create_tpgcn_branches(feature_dictionary, feature_list, og_features, symmetry, s=None):
    # Number of branches
    B = len(feature_list)
    # Number of channels in branch
    C = sum([feature_dictionary[k].shape[-1] for k in feature_list[0]])
    *_, T, V, _ = og_features.shape
    M = 2 if symmetry else 1
    new_features = np.zeros((B, C, T, V * 2, M), dtype=np.float32)
    for bi, branch in enumerate(feature_list):
        if s is not None:
            branch_features = [feature_dictionary[k][s] for k in branch]
        else:
            branch_features = [feature_dictionary[k] for k in branch]
        branch_features = np.concatenate(branch_features, axis=-1)
        branch_features = branch_features.transpose(3, 1, 2, 0)  # C, T, V, M
        new_features[bi, :, :, :V, 0] = branch_features[..., :, 0]
        new_features[bi, :, :, V:, 0] = branch_features[..., :, 1]
        if symmetry:
            new_features[bi, :, :, V:, 1] = branch_features[..., :, 0]
            new_features[bi, :, :, :V, 1] = branch_features[..., :, 1]
    return new_features


class PoseDataset(Dataset):
    def __init__(self, data_file: str, feature_list: Union[list[str], list[list[str]]],
                 sampler: Sampler, augments: list = (), symmetry: bool = False, norm_func=no_norm,
                 return_info: bool = False):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        # Data should be a dict with keys "labels", "points", "confidences", "image_shape"
        self.labels = data['action']
        self.points = data['poseXY']
        self.dataset_info = data['dataset_info']

        # Remove mutuals that do not have both bodies
        if isinstance(feature_list[0], list):
            for i in range(len(self.points), 0, -1):
                i -= 1
                if self.points[i].shape[0] != 2:
                    self.points.pop(i)
                    self.labels.pop(i)
                    self.dataset_info.pop(i)
                    #print(f"Removed {i}")

        self.confidences = data['poseConf']
        self.image_shape = data.get("im_shape", (1920, 1080))
        self.skeleton_type = data.get("skeleton_type", "coco17")

        self.sampler = sampler
        self.augments = augments

        self.symmetry = symmetry  # Symmetry processing for 2P-gcn
        self.feature_list = feature_list
        self.flat_feature_list = flatten_list(feature_list)

        assert len(self.labels) == len(self.points)
        self.transforms: dict[str, PoseTransform] = {k: v(self.skeleton_type) for k, v in TransformsDict.items()}

        self.required_transforms = solve_feature_transform_requirements(self.feature_list)

        for feature in self.flat_feature_list:
            if feature not in self.transforms.keys():
                raise ValueError(f"feature {feature} is not supported")

        unique_labels = list(sorted(set(self.labels)))
        self.label_translation = {x: i for i, x in enumerate(unique_labels)}
        self.norm_func = norm_func
        self.return_info = return_info

    def num_classes(self):
        return len(self.label_translation)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        points = self.points[idx]
        # Normalize
        points = self.norm_func(points)

        # Augments
        for augment in self.augments:
            points = augment(points)

        # Sample
        features = self.sampler(points)
        if isinstance(features, list):
            features = np.stack(features)

        # Calculate features
        feature_dictionary = {"joints": features}
        for feat in self.required_transforms:
            self.transforms[feat](feature_dictionary)

        # Combine features
        if isinstance(self.feature_list[0], str):
            features = transform_to_stgcn_input(feature_dictionary, self.feature_list)

        if isinstance(self.feature_list[0], list):
            features = transform_to_tpgcn_input(feature_dictionary, self.feature_list, self.symmetry)

        features = torch.from_numpy(features).float()
        if self.return_info:
            return features, self.label_translation[label], label, self.dataset_info[idx]
        else:
            return features, self.label_translation[label], label
# def matrix_to_skeleton_body(mat):
#     B, T, V, C = mat.shape
#     data = SkeletonData("ababa", "coco17", None, None, T, [], T, (1920, 1080))
#     for fi in range(T):
#         frame = FrameData(fi, B, [])
#         for b in range(B):
#             frame.bodies.append(Body(mat[b, fi, :, :], None, None, None, None, b))
#         data.frames.append(frame)
#     return data
#
#
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm
#
#     test_sampler = Sampler(64, 64, False, 5)
#     test_set = PoseDataset(
#         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl",
#         ["joints"],
#         test_sampler,
#         []
#     )
#     test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=0, pin_memory=True)
#     norm = SpineNormalization("/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl", use_mean=False, align=True)
#     for x, y, yy in tqdm(test_loader):
#         xx = norm(x[0])
#
#         break
#         x = screen_normalization(x, (1920, 1080))
#         sd = matrix_to_skeleton_body(x[0])
#         visualize_data(sd, 1000 // 30)
#         visualize_skeleton(
#             f"/home/barny/MasterThesis/Data/alphapose_skeletons/ntu_coco/S008C002P030R001A037.coco17.apskel.pkl")
#         ['S008C002P030R001A037']
#         break
#         pass
