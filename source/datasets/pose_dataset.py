from __future__ import annotations

import pickle
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets.sampler import Sampler
from datasets.transform_wrappers import TransformsDict, PoseTransform, TransformsList, TransformsNameList
from models import create_stgcnpp
from preprocessing.normalizations import screen_normalization


def flatten_list(in_list):
    if len(in_list) == 0 or not isinstance(in_list[0], list):
        return in_list
    return [x for hid_list in in_list for x in hid_list]


class PoseDataset(Dataset):
    def __init__(self, data_file: str, feature_list: Union[list[str], list[list[str]]],
                 sampler: Sampler, symmetry: bool = False):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        # Data should be a dict with keys "labels", "points", "confidences", "image_shape"
        self.labels = data['action']
        self.points = data['poseXY']

        self.confidences = data['poseConf']
        self.image_shape = data.get("im_shape", (1920, 1080))
        self.skeleton_type = data.get("skeleton_type", "coco17")

        self.sampler = sampler

        self.symmetry = symmetry  # Symmetry processing for 2P-gcn
        self.feature_list = feature_list
        self.flat_feature_list = flatten_list(feature_list)

        assert len(self.labels) == len(self.points)
        self.transforms: dict[str, PoseTransform] = {k: v(self.skeleton_type) for k, v in TransformsDict.items()}

        self.required_transforms = self.solve_feature_transform_requirements()

        for feature in self.flat_feature_list:
            if feature not in self.transforms.keys():
                raise ValueError(f"feature {feature} is not supported")

    def solve_feature_transform_requirements(self):
        required_transforms = set()
        queue = flatten_list(self.feature_list)[:]
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        points = self.points[idx]
        points = np.float32(points)
        # Normalize
        points = screen_normalization(points, (1920, 1080))

        # Sample
        features = self.sampler(points)
        listed = False
        if isinstance(features, list):
            listed = True
            features = np.stack(features)

        # Calculate features
        feature_dictionary = {"joints": features}
        for feat in self.required_transforms:
            self.transforms[feat](feature_dictionary)

        # Combine features
        if isinstance(self.feature_list[0], str):
            features = [feature_dictionary[k] for k in self.feature_list]
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
                    padded[..., 0,np.newaxis, :, :, :] = features
                    features = padded

        if isinstance(self.feature_list[0], list):
            out_data = []
            for s in range(features.shape[0]):
                # Number of branches
                B = len(self.feature_list)
                # Number of channels in branch
                C = sum([feature_dictionary[k].shape[-1] for k in self.feature_list[0]])
                *_, T, V, _ = features.shape
                M = 2 if self.symmetry else 1
                new_features = np.zeros((B, C, T, V * 2, M), dtype=np.float32)
                for bi, branch in enumerate(self.feature_list):
                    branch_features = [feature_dictionary[k][s] for k in branch]
                    branch_features = np.concatenate(branch_features, axis=-1)
                    branch_features = branch_features.transpose(3, 1, 2, 0)  # C, T, V, M
                    new_features[bi, :, :, :V, 0] = branch_features[..., :, 0]
                    new_features[bi, :, :, V:, 0] = branch_features[..., :, 1]
                    if self.symmetry:
                        new_features[bi, :, :, V:, 1] = branch_features[..., :, 0]
                        new_features[bi, :, :, :V, 1] = branch_features[..., :, 1]
                out_data.append(new_features)
            features = np.stack(features)

        features = torch.from_numpy(features)
        return features, label - 1, label


# if __name__ == "__main__":
#     test_sampler = Sampler(64, 32, True, 8)
#     # test_set = PoseDataset(
#     #     "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl",
#     #     TransformsNameList,
#     #     test_sampler
#     # )
#     # test_loader = DataLoader(test_set, 2, shuffle=False, num_workers=1, pin_memory=True)
#     # for x, labels, labels_real in tqdm(test_loader):
#     #     print(x.shape)
#     #     break
#
#     test_set = PoseDataset(
#         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_mutual_xsub.test.pkl",
#         [["joints", "angles"], ["joint_motion", "angles"]],
#         test_sampler
#     )
#     test_loader = DataLoader(test_set, 2, shuffle=False, num_workers=1, pin_memory=True)
#     for x, labels, labels_real in tqdm(test_loader):
#         print(x.shape)
#         break
#
# if __name__ == "__main__2":
#     test_sampler = Sampler(64, 32, True, 10)
#     test_set = PoseDataset(
#         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl",
#         ["joints", "joint_motion", "angles"],
#         test_sampler
#     )
#     test_loader = DataLoader(test_set, 2, shuffle=False, num_workers=4, pin_memory=True)
#     for x, labels in tqdm(test_loader):
#         print(x.shape)
#         break
#
# if __name__ == "__main__2":
#     import torch.nn.functional as F
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#
#     train_sampler = Sampler(64, 32)
#     train_set = PoseDataset(
#         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl",
#         ["joints", "joint_motion", "angles"],
#         train_sampler
#     )
#     train_loader = DataLoader(train_set, 16, True, num_workers=4, pin_memory=True)
#
#     test_sampler = Sampler(64, 32, True, 10)
#     test_set = PoseDataset(
#         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl",
#         ["joints", "joint_motion", "angles"],
#         test_sampler
#     )
#     test_loader = DataLoader(test_set, 2, shuffle=False, num_workers=4, pin_memory=True)
#
#     device = torch.device('cuda:0')
#     model = create_stgcnpp(60, 5)
#     model.to(device)
#
#     epochs = 40
#     loss_func = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
#     scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
#
#     for epoch in range(epochs):
#         running_loss = 0.0
#         current_lr = optimizer.param_groups[0]['lr']
#         for x, labels, labels_real in (tq := tqdm(train_loader)):
#             x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad()
#
#             y_pred = model(x)
#
#             loss = F.cross_entropy(y_pred, labels - 61)
#
#             tq.set_description(f"LR: {round(current_lr, 4):0<6} Loss: {round(float(loss), 4):0<7}")
#
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         tq.set_description(f"LR: {round(current_lr, 4):0<6} Loss: {round(float(running_loss / len(loader)), 4):0<7}")
#         scheduler.step()
