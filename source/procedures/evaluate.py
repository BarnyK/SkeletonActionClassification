from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp
from preprocessing.normalizations import create_norm_func, setup_norm_func
from procedures.config import EvalConfig, GeneralConfig
from procedures.training import test_epoch, load_model


def evaluate(cfg: GeneralConfig):
    norm_func = create_norm_func(cfg.normalization_type)
    test_sampler = Sampler(cfg.window_length, cfg.samples_per_window, True, cfg.eval_config.test_clips_count)
    test_set = PoseDataset(
        cfg.eval_config.test_file,
        cfg.features,
        test_sampler,
        [],
        cfg.symmetry_processing,
        norm_func
    )
    test_loader = DataLoader(test_set, cfg.eval_config.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(cfg.device)
    channels = calculate_channels(cfg.features, 2)
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(test_set.num_classes(), channels, cfg.skeleton_type)
    else:
        raise ValueError("2p-gcn not supported yet")

    # load
    state_dict = load_model(cfg.best_model_path, model, None, None, device)
    if norm_state_dict := state_dict.get("normalization"):
        setup_norm_func(norm_func, state_dict=norm_state_dict)
    else:
        setup_norm_func(norm_func, train_file=cfg.train_config.train_file)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    stats = test_epoch(model, test_loader, loss_func, device)
    print(stats)


if __name__ == "__main__":
    cfg = GeneralConfig.from_yaml_file("/media/barny/SSD4/MasterThesis/Data/logs/default_64_32_0/config.yaml")
    print(cfg)
    cfg.device = "cuda"
    cfg.eval_config.test_batch_size = 1
    evaluate(cfg)
    #cfg = EvalConfig("stgcnpp", model_file, "cuda:0", ["joints"], 64, 32, )
