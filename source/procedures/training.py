from __future__ import annotations

import logging
import os
import re
import shutil
import time
from datetime import datetime, timedelta

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.augments import RandomScale
from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp
from preprocessing.normalizations import create_norm_func
from procedures.config import TrainingConfig, GeneralConfig

logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO (or other level of your choice)
logger = logging.getLogger(__name__)


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


def test_epoch(model, loader, loss_func, device):
    model.eval()
    running_loss = 0.0
    top1_count = 0
    top5_count = 0
    sample_count = 0
    for x, labels, labels_real in (tq := tqdm(loader)):
        tq.set_description("Testing")
        batch_size, num_samples, *rest = x.shape
        x = x.reshape(batch_size * num_samples, *rest)
        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            out: torch.Tensor = model(x)
            out = out.reshape(batch_size, num_samples, -1)
            out = out.mean(dim=1)
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
    logger.info(f"Top1 accuracy: {top1_accuracy:.2%}")
    logger.info(f"Top5 accuracy: {top5_accuracy:.2%}")
    logger.info(f"Mean loss: {mean_loss:.2}")
    return mean_loss, top1_accuracy, top5_accuracy


def write_log(logs_path, text):
    with open(os.path.join(logs_path, "training.log", ), "a+") as f:
        f.write(text + '\n')


def save_model(filename, model, optimizer, scheduler):
    data = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(data, filename)


def load_model(filename, model, optimizer, scheduler, device):
    data = torch.load(filename, map_location=device)
    if "net" in data.keys():
        model.load_state_dict(data['net'])
        if optimizer:
            optimizer.load_state_dict(data['optimizer'])
        if scheduler:
            scheduler.load_state_dict(data['scheduler'])
    else:
        model.load_state_dict(data)


def train_network(cfg: GeneralConfig):
    t_cfg = cfg.train_config
    e_cfg = cfg.eval_config
    if cfg.name is None:
        now = datetime.now()
        cfg.name = now.strftime("%H_%M_%d_%m_%Y")
    logger.info("Starting training")
    logger.info(f"Using {cfg.features}")

    test_loader, train_loader, test_set, _ = create_dataloaders(cfg)

    device = torch.device(cfg.device)
    channels = calculate_channels(cfg.features, 2)
    if cfg.model_type == "stgcnpp":
        model = create_stgcnpp(test_set.num_classes(), channels)
        model.to(device)
    else:
        raise ValueError("2p-gcn not supported yet")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=t_cfg.sgd_lr, momentum=t_cfg.sgd_momentum,
                                weight_decay=t_cfg.sgd_weight_decay, nesterov=t_cfg.sgd_nesterov)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg.epochs,
                                  eta_min=t_cfg.cosine_shed_eta_min)
    start_epoch = 0

    # Resume handling
    logs_path = os.path.join(cfg.log_folder, cfg.name)
    if os.path.exists(logs_path):
        existing_cfg_path = os.path.join(logs_path, "config.yaml")
        existing_cfg = GeneralConfig.from_yaml_file(existing_cfg_path)
        if existing_cfg != cfg:
            raise ValueError("Existing config is different to the current one")
        # load 
        models_folder = os.path.join(logs_path, "models")
        if os.path.isdir(models_folder):
            epoch_files = os.listdir(models_folder)
            if epoch_files:
                newest_file = max([os.path.join(models_folder, x) for x in epoch_files], key=os.path.getctime)
                match = re.findall("([0-9]*).pth", newest_file)[-1]
                start_epoch = int(match) + 1
                load_model(newest_file, model, optimizer, scheduler, device)

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "models"), exist_ok=True)

    cfg.to_yaml_file(os.path.join(logs_path, "config.yaml"))
    write_log(logs_path, f"Training with features: {', '.join(cfg.features)}")

    start_time = time.time()
    all_eval_stats = []
    for epoch in range(start_epoch, t_cfg.epochs):
        logger.info(f"Current epoch: {epoch + 1}/{t_cfg.epochs}")
        train_start_time = time.time()
        training_stats = train_epoch(model, loss_func, train_loader, optimizer, scheduler, device)
        train_end_time = time.time()

        write_log(logs_path, f"[{epoch}] - training stats - {','.join([str(x) for x in training_stats])}")
        write_log(logs_path, f"[{epoch}] - training time - {timedelta(seconds=train_end_time - train_start_time)}")

        if (epoch + 1) % e_cfg.eval_interval == 0 or t_cfg.epochs - epoch - 1 < e_cfg.eval_last_n:
            eval_start_time = time.time()
            eval_stats = test_epoch(model, test_loader, loss_func, device)
            eval_end_time = time.time()
            all_eval_stats.append((epoch, *eval_stats))

            write_log(logs_path, f"[{epoch}] - eval stats - {','.join([str(x) for x in eval_stats])}")
            write_log(logs_path, f"[{epoch}] - eval time - {timedelta(seconds=eval_end_time - eval_start_time)}")

        # Save model
        model_path = os.path.join(logs_path, "models", f"epoch_{epoch}.pth")
        save_model(model_path, model, optimizer, scheduler)

        # Print ETA
        estimated_remaining_time = ((time.time() - start_time) / (epoch + 1)) * (t_cfg.epochs - (epoch + 1))
        logger.info(f"Estimated remaining time: {timedelta(seconds=estimated_remaining_time)}")

    end_time = time.time()
    best_eval_epoch = max(all_eval_stats, key=lambda x: x[2])
    logger.info(f"Best top1 accuracy {best_eval_epoch[2]:.2%} at epoch {best_eval_epoch[0]}")
    logger.info(f"Training took {timedelta(seconds=end_time - start_time)}")
    write_log(logs_path, f"Top1 accuracy {best_eval_epoch[2]:.2%} at epoch {best_eval_epoch[0]}")

    best_epoch_file = os.path.join(logs_path, "models", f"epoch_{best_eval_epoch[0]}.pth")
    best_file = os.path.join(logs_path, f"best.pth")
    shutil.copy(best_epoch_file, best_file)


def create_dataloaders(cfg: GeneralConfig):
    augments = []
    if cfg.train_config.use_scale_augment:
        augments.append(RandomScale(cfg.train_config.scale_value))

    norm_func = create_norm_func(cfg.normalization_type, cfg.train_config.train_file)

    train_sampler = Sampler(cfg.window_length, cfg.sampler_per_window)
    train_set = PoseDataset(
        cfg.train_config.train_file,
        cfg.features,
        train_sampler,
        augments,
        cfg.symmetry_processing,
        norm_func
    )
    train_loader = DataLoader(train_set, cfg.train_config.train_batch_size, True, num_workers=4, pin_memory=True)

    test_sampler = Sampler(cfg.window_length, cfg.sampler_per_window, True, cfg.eval_config.test_clips_count)
    test_set = PoseDataset(
        cfg.eval_config.test_file,
        cfg.features,
        test_sampler,
        [],
        cfg.symmetry_processing,
        norm_func
    )
    test_loader = DataLoader(test_set, cfg.eval_config.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader, train_loader, test_set, train_set


def main():
    cfg = TrainingConfig("base_joints", "stgcnpp", 80, "cuda:0", ["joints"], 64, 32,
                         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl", 64,
                         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl", 128, 8, 1, 0.1,
                         0.9, 0.0002, True, 0.00001)
    train_network(cfg)
