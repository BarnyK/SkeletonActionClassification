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


def solve_feature_transform_requirements(feature_list: Union[list[str], list[list[str]]]):
    """
    Given a list of features it returns the ordered list of required feature transformations
    """
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


def transform_to_stgcn_input(feature_dictionary: dict, feature_list: list):
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


def transform_to_tpgcn_input(feature_dictionary: dict, feature_list: list, symmetry: bool, copy_pad: bool):
    og_features = feature_dictionary['joints']
    if og_features.ndim == 4:
        return create_tpgcn_branches(feature_dictionary, feature_list, og_features, symmetry, copy_pad)
    else:
        # Sampling creates new dimension
        out_data = []
        for s in range(og_features.shape[0]):
            new_features = create_tpgcn_branches(feature_dictionary, feature_list, og_features, symmetry, copy_pad, s)
            out_data.append(new_features)
        features = np.stack(out_data)
        return features


def create_tpgcn_branches(feature_dictionary: dict, feature_list: list, og_features,
                          symmetry: bool, copy_pad: bool, s: int = None):
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
        if branch_features.shape[-1] == 1:
            if copy_pad:
                branch_features = np.concatenate([branch_features, branch_features], -1)
            else:
                # Zero pad
                padded = np.zeros((C, T, V, 2), dtype=np.float32)
                padded[..., 0] = branch_features[..., 0]
                branch_features = padded
        new_features[bi, :, :, :V, 0] = branch_features[..., :, 0]
        new_features[bi, :, :, V:, 0] = branch_features[..., :, 1]
        if symmetry:
            new_features[bi, :, :, V:, 1] = branch_features[..., :, 0]
            new_features[bi, :, :, :V, 1] = branch_features[..., :, 1]
    return new_features


class PoseDataset(Dataset):
    def __init__(self, data_file: str, feature_list: Union[list[str], list[list[str]]],
                 sampler: Sampler, augments: list = (), symmetry: bool = False, norm_func=no_norm,
                 return_info: bool = False, copy_pad: bool = False):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        # Data should be a dict with keys "labels", "points", "confidences", "image_shape"
        self.labels = data['action']
        self.points = data['poseXY']
        self.dataset_info = data['dataset_info']

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
        self.copy_pad = copy_pad

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
            features = transform_to_tpgcn_input(feature_dictionary, self.feature_list, self.symmetry, self.copy_pad)

        features = torch.from_numpy(features).float()
        if self.return_info:
            return features, self.label_translation[label], label, idx, self.dataset_info[idx]
        else:
            return features, self.label_translation[label], label
