from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets.sampler import sampler
from datasets.transform_wrappers import TransformsDict, PoseTransform, TransformsList
from models import create_stgcnpp
from preprocessing.normalizations import spine_normalization, screen_normalization


def flatten_list(in_list):
    if len(in_list) == 0 or not isinstance(in_list[0], list):
        return in_list
    return [x for hid_list in in_list for x in hid_list]


class PoseDataset(Dataset):
    def __init__(self, data_file: str, feature_list: list[str], window_size: int, samples_per_window: int,
                 symmetry: bool = False):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        # Data should be a dict with keys "labels", "points", "confidences", "image_shape"
        self.labels = data['labels']
        self.labels = [x['info']['action'] for x in data['labels']]
        self.points = data['points']

        self.confidences = data.get("confidences")
        self.image_shape = data.get("image_shape", (1920, 1080))
        self.skeleton_type = data.get("skeleton_type", "coco17")

        self.window_size = window_size
        self.samples_per_window = samples_per_window

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
        features = sampler(points, self.window_size, self.samples_per_window)

        # Calculate features
        feature_dictionary = {"joints": features}
        for feat in self.required_transforms:
            self.transforms[feat](feature_dictionary)

        # Combine features
        if isinstance(self.feature_list[0], str):
            features = [feature_dictionary[k] for k in self.feature_list]
            features = np.concatenate(features, axis=-1)

            # Pad empty dimension
            M, T, V, C = features.shape
            if M == 1:
                padded = np.zeros((2, T, V, C), dtype=np.float32)
                padded[0, ...] = features
                features = padded

        if isinstance(self.feature_list[0], list):
            # Number of branches
            B = len(self.feature_list)
            # Number of channels in branch
            C = sum([feature_dictionary[k].shape[-1] for k in self.feature_list[0]])
            _, T, V, _ = features.shape
            M = 2 if self.symmetry else 1
            features = np.zeros((B, C, T, V * 2, M), dtype=np.float32)
            for bi, branch in enumerate(self.feature_list):
                branch_features = [feature_dictionary[k] for k in branch]
                branch_features = np.concatenate(branch_features, axis=-1)
                branch_features = branch_features.transpose(3, 1, 2, 0)  # C, T, V, M
                features[bi, :, :, :V, 0] = branch_features[..., :, 0]
                features[bi, :, :, V:, 0] = branch_features[..., :, 1]
                if self.symmetry:
                    features[bi, :, :, V:, 1] = branch_features[..., :, 0]
                    features[bi, :, :, :V, 1] = branch_features[..., :, 1]

        features = torch.from_numpy(features)
        return features, label


if __name__ == "__main__2":
    dataset = PoseDataset(
        "/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.f1.combined",
        [["joints", "joint_motion"], ["bones", "bone_accel"], ["bones", "bone_motion"]],
        64,
        64)
    loader = DataLoader(dataset, 16, False, num_workers=1, pin_memory=True)
    for x, labels in tqdm(loader):
        print(x.shape)
        break

if __name__ == "__main__":
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import CosineAnnealingLR
    dataset = PoseDataset(
        "/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.f1.combined",
        ["joints", "joint_motion", "angles"],
        64,
        64)
    loader = DataLoader(dataset, 16, True, num_workers=4, pin_memory=True)
    device = torch.device('cuda:0')
    model = create_stgcnpp(60, 5)
    model.to(device)

    epochs = 40
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    for epoch in range(epochs):
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        for x, labels in (tq := tqdm(loader)):
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            y_pred = model(x)

            loss = F.cross_entropy(y_pred, labels-61)

            tq.set_description(f"LR: {round(current_lr,4):0<6} Loss: {round(float(loss),4):0<7}")

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        tq.set_description(f"LR: {round(current_lr, 4):0<6} Loss: {round(float(running_loss/len(loader)), 4):0<7}")
        scheduler.step()

