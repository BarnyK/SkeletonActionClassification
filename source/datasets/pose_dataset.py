import pickle

import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets.sampler import sampler
import torch

from preprocessing.feature_extraction import to_angles


class PoseDataset(Dataset):
    def __init__(self, data_file: str, window_size: int, samples_per_window: int):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        # Data should be a dict with keys "labels", "points", "confidences", "image_shape"
        self.labels = data['labels']
        self.points = data['points']
        self.confidences = data.get("confidences")
        self.image_shape = data.get("image_shape", (1920, 1080))
        self.skeleton_type = data.get("skeleton_type", "coco17")
        self.window_size = window_size
        self.samples_per_window = samples_per_window

        assert len(self.labels) == len(self.points)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        points = self.points[idx]
        # Normalize

        # Sample
        data = sampler(points, self.window_size, self.samples_per_window)
        # data = torch.from_numpy(data)
        # Data is of shape [N, T, point_count, 2]

        # Features: "joints", "bones", "m_joints", "m_bones", "angles", "bone_angles", "bone_lengths"
        #              2         2         2           2         1            2              1
        # featTODO: "relative_joint", "accel_joints", "accel_bones"
        #               2                   2               2
        # Calculate features
        feature_dictionary = {
            "joints": data, "angles": to_angles(data, self.skeleton_type),
        }

        # Combine features
        combine_list = ["joints", "angles", "bones", "bone_lengths"]
        if isinstance(combine_list[0], list):
            # TPGCN
            raise NotImplementedError
            pass
        else:
            # STGCN
            features = [v for v in feature_dictionary.values()]
            features = np.stack(features,axis=-1)
            channels = [v.shape[-1] for v in features]
            channel_count = sum(channels)

            pass

        # Reshape

        #
        # Output should be [2, window_size, point_count, channels]
        # 2P-GCN output should be [pipes, channels, window_size, 2*point_count, 1/2]

        return data, label


if __name__ == "__main__":
    dataset = PoseDataset("/media/barny/SSD4/MasterThesis/Data/ntu_120_coco.f1.combined", 64, 64)
    loader = DataLoader(dataset, 1, False, num_workers=4, pin_memory=True)

    for x, y in loader:
        print(x.shape)
        pass
