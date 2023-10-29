from unittest import TestCase

import numpy as np

from datasets.sampler import legacy_sampler, Sampler


class Test(TestCase):
    def test_sampler_shorter(self):
        data = np.random.random((2, 55, 17, 2))
        win = legacy_sampler(data, 64, 32)
        assert win.shape[1] == 32

    def test_sampler_longer(self):
        data = np.random.random((2, 255, 17, 2))
        win = legacy_sampler(data, 64, 16)
        assert win.shape[1] == 16

    def test_sampler_bad_params(self):
        data = np.random.random((2, 255, 17, 2))
        with self.assertRaises(AssertionError) as context:
            win = legacy_sampler(data, 64, 31)


class TestSampler(TestCase):
    def setUp(self):
        pass

    def test_train_sample_exact(self):
        sampler = Sampler(64, 32, False, 5, 111)
        data = np.random.rand(1, 64, 17, 2)
        sample = sampler(data)

    def test_train_sample_smaller(self):
        sampler = Sampler(64, 32, False, 5, 111)
        data = np.random.rand(1, 53, 17, 2)
        sample = sampler(data)

    def test_train_sample_bigger(self):
        sampler = Sampler(64, 32, False, 5, 111)
        data = np.random.rand(1, 100, 17, 2)
        sample = sampler(data)

    def test_eval_sample_exact(self):
        sampler = Sampler(64, 32, True, 12, 111)
        data = np.random.rand(1, 64, 17, 2)
        sample = sampler(data)

    def test_eval_sample_smaller(self):
        sampler = Sampler(64, 32, True, 25, 111)
        data = np.random.rand(1, 53, 17, 2)
        sample = sampler(data)

    def test_eval_sample_bigger(self):
        sampler = Sampler(64, 32, True, 15, 111)
        data = np.random.rand(1, 100, 17, 2)
        sample = sampler(data)
