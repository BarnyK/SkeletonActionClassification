from __future__ import annotations

import logging
import os
import re
import shutil
import time
from datetime import timedelta
from typing import Union

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.augments import RandomScale
from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp, create_tpgcn
from preprocessing.normalizations import create_norm_func, SpineNormalization, setup_norm_func, ScreenNormalization
from procedures.config import GeneralConfig
from shared.errors import DifferentConfigException
from shared.helpers import parse_training_log

logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO (or other level of your choice)


def train_epoch(model, loss_func, loader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    current_lr = optimizer.param_groups[0]['lr']
    top1_count, top5_count, sample_count = 0, 0, 0
    for x, labels, labels_real in (tq := tqdm(loader)):
        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        out: torch.Tensor = model(x)

        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        sample_count += x.shape[0]
        with torch.no_grad():
            top1 = out.max(1)[1]
            top1_count += (top1 == labels).sum().item()

            top5 = torch.topk(out, 5)[1]
            top5_count += sum([labels[n] in top5[n, :] for n in range(x.shape[0])])

            top1_accuracy = top1_count / sample_count
        tq.set_description(
            f"Training, LR: {current_lr:.4} Loss: {loss.item():.4} Accuracy: {top1_accuracy:.2%}")

    mean_loss = float(running_loss / len(loader))
    top1_accuracy = top1_count / sample_count
    top5_accuracy = top5_count / sample_count
    tq.set_description(
        f"Training, LR: {current_lr:.4} Loss: {mean_loss:.4} Accuracy: {top1_accuracy:.2%}")
    scheduler.step()
    return mean_loss, top1_accuracy, top5_accuracy


def test_epoch(model, loader, loss_func, device, save_results=False):
    model.eval()
    running_loss = 0.0
    top1_count = 0
    top5_count = 0
    sample_count = 0
    results, all_labels, all_real_labels, names = [], [], [], []
    for iter_data in (tq := tqdm(loader)):
        if len(iter_data) == 5:
            x, labels, labels_real, idx, dataset_info = iter_data
        else:
            x, labels, labels_real, *_ = iter_data
            dataset_info = ""
        tq.set_description("Testing")
        batch_size, num_samples, *rest = x.shape
        x = x.reshape(batch_size * num_samples, *rest)
        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            out: torch.Tensor = model(x)
            out = out.reshape(batch_size, num_samples, -1)
            out = out.mean(dim=1)
            if save_results:
                res_out = out.cpu().numpy()
                results.append(res_out)
                all_labels.extend(labels.cpu().tolist())
                all_real_labels.extend(labels_real.cpu().tolist())
                names.extend(dataset_info)

            loss = loss_func(out, labels)
            sample_count += batch_size

            top1 = out.max(1)[1]
            top1_count += (top1 == labels).sum().item()

            top5 = torch.topk(out, 5)[1]
            top5_count += sum([labels[n] in top5[n, :] for n in range(batch_size)])
        running_loss += loss.item()
    top1_accuracy = top1_count / sample_count
    top5_accuracy = top5_count / sample_count
    mean_loss = running_loss / len(loader)

    if save_results:
        results = np.concatenate(results, 0)
        return mean_loss, top1_accuracy, top5_accuracy, results, all_labels, all_real_labels, names
    return mean_loss, top1_accuracy, top5_accuracy


def write_log(logs_path, texts: Union[str, list[str]]):
    with open(os.path.join(logs_path, "training.log", ), "a+") as f:
        if isinstance(texts, str):
            f.write(texts + '\n')
        elif isinstance(texts, list):
            for text in texts:
                f.write(text + '\n')


def save_model(filename, model, optimizer, scheduler, norm_func):
    data = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if isinstance(norm_func, (SpineNormalization, ScreenNormalization)):
        data['normalization'] = norm_func.state_dict()
    torch.save(data, filename)


def load_model(filename, model, optimizer, scheduler, device):
    data = torch.load(filename, map_location=device)
    if "net" in data.keys():
        if model:
            model.load_state_dict(data['net'])
        if optimizer:
            optimizer.load_state_dict(data['optimizer'])
        if scheduler:
            scheduler.load_state_dict(data['scheduler'])
    else:
        if model:
            model.load_state_dict(data)
    return data


def keep_best_models(logs_path, keep_best_n):
    # List epoch models with their numbers
    models_dir = os.path.join(logs_path, "models")
    epoch_models = [x for x in os.listdir(models_dir) if x.endswith(".pth") and ".epoch" not in x]
    numbers = [int(re.search(r'\d+', filename).group()) for filename in epoch_models]
    # Get evaluation stats
    log_file = os.path.join(logs_path, "training.log")
    train_stats, train_times, eval_stats, eval_times = parse_training_log(log_file)
    eval_stats = sorted(eval_stats, key=lambda x: x[3], reverse=True)
    best_epoch_ids = [x[0] for x in eval_stats[:keep_best_n]]

    # Remove weak files
    for epoch_id, file in zip(numbers, epoch_models):
        if epoch_id not in best_epoch_ids:
            # Remove
            os.remove(os.path.join(models_dir, file))
        else:
            # Rename
            old_file = os.path.join(models_dir, file)
            new_file = os.path.join(models_dir, f"{best_epoch_ids.index(epoch_id) + 1}.{file}")
            os.rename(old_file, new_file)

    pass


def train_network(cfg: GeneralConfig):
    t_cfg = cfg.train_config
    e_cfg = cfg.eval_config

    logger = logging.getLogger(cfg.name)
    logger.info("Starting training")
    logger.info(f"Using {cfg.features}")

    # Resume handling
    start_epoch = 0
    finished = False
    load_file = None
    logs_path = os.path.join(cfg.log_folder, cfg.name)
    cfg_path = os.path.join(logs_path, "config.yaml")
    if os.path.exists(logs_path):
        existing_cfg_path = cfg_path
        existing_cfg = GeneralConfig.from_yaml_file(existing_cfg_path)
        if existing_cfg != cfg:
            diffs = GeneralConfig.compare(existing_cfg, cfg)
            if len(diffs) > 0:
                raise DifferentConfigException(f"Existing config is different to the current one - {', '.join(diffs)}")
        if os.path.exists(os.path.join(cfg.best_model_path)):
            finished = True
        # load
        models_folder = os.path.join(logs_path, "models")
        if not finished and os.path.isdir(models_folder):
            epoch_files = os.listdir(models_folder)
            if epoch_files:
                load_file = max([os.path.join(models_folder, x) for x in epoch_files], key=os.path.getctime)
                match = re.findall("([0-9]*).pth", load_file)[-1]
                start_epoch = int(match) + 1

    # Create directories
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "models"), exist_ok=True)

    # Write config
    cfg.to_yaml_file(cfg_path)

    # Return if training already complete
    if start_epoch >= t_cfg.epochs or finished:
        logger.info("Training completed")
        return cfg_path

    # Create dataloaders
    test_loader, train_loader, test_set, _, norm_func = create_dataloaders(cfg)

    # Create model
    device = torch.device(cfg.device)
    channels = calculate_channels(cfg.features, test_set.points[0].shape[-1])
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(test_set.num_classes(), channels, cfg.skeleton_type)
        model.to(device)
    elif cfg.model_type == "2pgcn":
        model = create_tpgcn(test_set.num_classes(), len(cfg.features), channels,
                             cfg.skeleton_type, cfg.labeling, cfg.graph_type)
        model.to(device)
    else:
        raise KeyError(f"model type {cfg.model_type} not supported")

    # Create training loss, optimizer and scheduler
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=t_cfg.sgd_lr, momentum=t_cfg.sgd_momentum,
                                weight_decay=t_cfg.sgd_weight_decay, nesterov=t_cfg.sgd_nesterov)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs,
                                  eta_min=t_cfg.cosine_shed_eta_min)

    # Load data if exists
    if load_file is not None:
        state_dict = load_model(load_file, model, optimizer, scheduler, device)
        if norm_state_dict := state_dict.get("normalization"):
            setup_norm_func(norm_func, state_dict=norm_state_dict)
        else:
            setup_norm_func(norm_func, train_file=cfg.train_config.train_file)
    else:
        setup_norm_func(norm_func, train_file=cfg.train_config.train_file)

    # Start training loop
    start_time = time.time()
    all_eval_stats = []
    for epoch in range(start_epoch, t_cfg.epochs):
        log_queue = []
        logger.info(f"Current epoch: {epoch + 1}/{t_cfg.epochs}")
        train_start_time = time.time()
        training_stats = train_epoch(model, loss_func, train_loader, optimizer, scheduler, device)
        train_end_time = time.time()

        log_queue.append(f"[{epoch}] - training stats - {','.join([str(x) for x in training_stats])}")
        log_queue.append(f"[{epoch}] - training time - {timedelta(seconds=train_end_time - train_start_time)}")

        if (epoch + 1) % e_cfg.eval_interval == 0 or t_cfg.epochs - epoch - 1 < e_cfg.eval_last_n:
            eval_start_time = time.time()
            eval_stats = test_epoch(model, test_loader, loss_func, device)
            eval_end_time = time.time()
            all_eval_stats.append((epoch, *eval_stats))
            logger.info(f"Top1 accuracy: {eval_stats[1]:.2%}")
            logger.info(f"Top5 accuracy: {eval_stats[2]:.2%}")
            logger.info(f"Mean loss: {eval_stats[0]:.2}")

            log_queue.append(f"[{epoch}] - eval stats - {','.join([str(x) for x in eval_stats])}")
            log_queue.append(f"[{epoch}] - eval time - {timedelta(seconds=eval_end_time - eval_start_time)}")

        # Save model
        model_path = os.path.join(logs_path, "models", f"epoch_{epoch}.pth")
        save_model(model_path, model, optimizer, scheduler, norm_func)

        write_log(logs_path, log_queue)

        # Print ETA
        remaining_time = ((time.time() - start_time) / (epoch - start_epoch + 1)) * (t_cfg.epochs - (epoch + 1))
        logger.info(f"Estimated remaining time: {timedelta(seconds=remaining_time)}")

    end_time = time.time()
    best_eval_epoch = max(all_eval_stats, key=lambda x: x[2])
    logger.info(f"Best top1 accuracy {best_eval_epoch[2]:.2%} at epoch {best_eval_epoch[0]}")
    logger.info(f"Training took {timedelta(seconds=end_time - start_time)}")
    write_log(logs_path, f"Top1 accuracy {best_eval_epoch[2]:.2%} at epoch {best_eval_epoch[0]}")

    best_epoch_file = os.path.join(logs_path, "models", f"epoch_{best_eval_epoch[0]}.pth")
    shutil.copy(best_epoch_file, cfg.best_model_path)
    logger.info("Training completed")
    keep_best_models(logs_path, cfg.train_config.keep_best_n)
    return cfg_path


def create_dataloaders(cfg: GeneralConfig):
    augments = []
    if cfg.train_config.use_scale_augment:
        augments.append(RandomScale(cfg.train_config.scale_value))

    norm_func = create_norm_func(cfg.normalization_type)

    train_sampler = Sampler(cfg.window_length, cfg.samples_per_window)
    train_set = PoseDataset(
        cfg.train_config.train_file,
        cfg.features,
        train_sampler,
        augments,
        cfg.symmetry_processing,
        norm_func,
        False,
        cfg.copy_pad,
    )
    train_loader = DataLoader(train_set, cfg.train_config.train_batch_size, True, num_workers=4, pin_memory=True)

    test_sampler = Sampler(cfg.window_length, cfg.samples_per_window, True, cfg.eval_config.test_clips_count)
    test_set = PoseDataset(
        cfg.eval_config.test_file,
        cfg.features,
        test_sampler,
        [],
        cfg.symmetry_processing,
        norm_func,
        False,
        cfg.copy_pad,
    )
    test_loader = DataLoader(test_set, cfg.eval_config.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader, train_loader, test_set, train_set, norm_func
