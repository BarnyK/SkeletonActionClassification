from __future__ import annotations

import os
from argparse import Namespace
from typing import Union

import torch
from torch.utils.data import DataLoader

from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp, create_tpgcn
from preprocessing.normalizations import create_norm_func, setup_norm_func
from procedures.config import GeneralConfig
from procedures.training import test_epoch, load_model


def evaluate(cfg: GeneralConfig, model_path: Union[str, None] = None, out_path: Union[str, None] = None,
             test_file: str = None):
    if test_file is None:
        test_file = cfg.eval_config.test_file
    norm_func = create_norm_func(cfg.normalization_type)
    test_sampler = Sampler(cfg.window_length, cfg.samples_per_window, True, cfg.eval_config.test_clips_count)
    test_set = PoseDataset(
        test_file,
        cfg.features,
        test_sampler,
        [],
        cfg.symmetry_processing,
        norm_func,
        True,
        copy_pad=cfg.copy_pad
    )
    test_loader = DataLoader(test_set, cfg.eval_config.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(cfg.device)
    channels = calculate_channels(cfg.features, test_set.points[0].shape[-1])
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(test_set.num_classes(), channels, cfg.skeleton_type)
    elif cfg.model_type == "2pgcn":
        model = create_tpgcn(test_set.num_classes(), len(cfg.features), channels,
                             cfg.skeleton_type, cfg.labeling, cfg.graph_type)
    else:
        raise KeyError(f"model type {cfg.model_type} not supported")

    # load
    if not model_path:
        model_path = cfg.best_model_path
    state_dict = load_model(model_path, model, None, None, device)
    if norm_state_dict := state_dict.get("normalization"):
        setup_norm_func(norm_func, state_dict=norm_state_dict)
    else:
        setup_norm_func(norm_func, train_file=cfg.train_config.train_file)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    (
        mean_loss,
        top1_accuracy,
        top5_accuracy,
        results,
        labels,
        real_labels,
        names
    ) = test_epoch(model, test_loader, loss_func, device, save_results=True)

    texts = ["Mean loss", "Top1 accuracy", "Top5 accuracy"]
    values = [mean_loss, top1_accuracy, top5_accuracy]
    for i, text in enumerate(texts):
        print(f"{text + ':':15}{values[i]:10.6}")

    results_data = {"config": cfg.to_yaml(), "results": results, "labels": labels, "real_labels": real_labels,
                    "names": names, "top1": top1_accuracy, "top5": top5_accuracy}
    if out_path:
        torch.save(results_data, out_path)
    return results_data


def handle_eval(args: Namespace):
    cfg = GeneralConfig.from_yaml_file(args.config)
    if args.model and not os.path.isfile(args.model):
        print(f"{args.model} does not exist")
        return False
    if args.save_file and not os.path.isdir(os.path.split(args.save_file)[0]):
        print(f"Folder for {args.save_file} does not exist")
        return False
    evaluate(cfg, args.model, args.save_file)


def evaluate_folder(folder_path: str):
    files = [(root, os.path.join(root, file)) for root, subdir, files in os.walk(folder_path) for file in files if
             file == "config.yaml"]
    files = [x for x in files if not os.path.isfile(os.path.join(x[0], "results.pkl"))]
    for i, (root, config_file) in enumerate(files):
        print(f"{i:<4}{len(files):<8}{config_file}")
        cfg = GeneralConfig.from_yaml_file(config_file)
        file_path = os.path.join(root, "results.pkl")
        model_path = os.path.join(root, "best.pth")
        if not os.path.isfile(model_path):
            print(f"{model_path} doesn't exist")
            continue
        if os.path.isfile(file_path):
            print("Already done")
            continue
        evaluate(cfg, None, file_path)
