from unittest import TestCase

import numpy as np

from datasets.sampler import sampler


class Test(TestCase):
    def test_sampler_shorter(self):
        data = np.random.random((2, 55, 17, 2))
        win = sampler(data, 64, 32)
        assert win.shape[1] == 32

    def test_sampler_longer(self):
        data = np.random.random((2, 255, 17, 2))
        win = sampler(data, 64, 16)
        assert win.shape[1] == 16

    def test_sampler_bad_params(self):
        data = np.random.random((2, 255, 17, 2))
        with self.assertRaises(AssertionError) as context:
            win = sampler(data, 64, 31)
