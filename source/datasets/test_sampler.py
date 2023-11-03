from unittest import TestCase

import numpy as np

from datasets.sampler import Sampler


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
