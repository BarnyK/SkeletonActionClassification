import numpy as np


class RandomScale:
    def __init__(self, scale=0.2):
        assert isinstance(scale, tuple) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, skeleton):
        scale = self.scale
        scale = (scale,) * skeleton.shape[-1]
        scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
        return skeleton * scale
