import numpy as np
from tqdm import tqdm

from datasets.sampler import Sampler


class RandomScale:
    def __init__(self, scale=0.2):
        assert isinstance(scale, tuple) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, skeleton):
        scale = self.scale
        scale = (scale,) * skeleton.shape[-1]
        scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
        return skeleton * scale


if __name__ == "__main__":
    import datasets.pose_dataset
    from torch.utils.data import DataLoader

    test_sampler = Sampler(64, 32, False, 5)
    test_set = datasets.pose_dataset.PoseDataset(
        "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xview.train.pkl",
        ["joints"],
        test_sampler,
        [RandomScale(0.2)]
    )
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=0, pin_memory=True)

    scaler = RandomScale(0.2)
    for x, y, yy in tqdm(test_loader):
        pass
