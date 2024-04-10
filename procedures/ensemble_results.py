from __future__ import annotations

import os.path

import numpy as np
import torch

from procedures.config import GeneralConfig


def ensemble_results(files: list[str]):
    all_data = []
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{file} does not exist")
        data = torch.load(file)
        if set(data.keys()) != {'config', 'results', 'labels', 'real_labels', 'names', 'top1', 'top5'}:
            raise KeyError("Loaded keys mismatch")
        data['config'] = GeneralConfig.from_yaml(data['config'])
        print(f"{file:30}\t{data['top1']:<8.4}{data['config'].features}")
        all_data.append(data)

    stacked_results = np.stack([x['results'] for x in all_data], 0)
    summed_results = np.sum(stacked_results, 0)
    labels_results = np.argmax(summed_results, 1)
    top1 = np.mean(labels_results == all_data[0]['labels'])
    print(f"Ensemble accuracy: {top1:17.4}")
    return top1


if __name__ == '__main__':
    ensemble_results(["/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jomo-jore_ntu_xsub_0/results.pkl",
                      "/media/barny/SSD4/MasterThesis/Data/logs/feature_test/stgcn_jo-jore_ntu_xsub_0/results.pkl"])
