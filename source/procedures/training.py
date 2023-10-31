from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import torch
from numpy import argmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp

logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO (or other level of your choice)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    name: str
    model_type: str
    epochs: int
    device: str
    features: list[str]
    window_length: int
    sampler_per_window: int

    train_file: str
    train_batch_size: int

    test_file: str
    test_batch_size: int
    test_clips_count: int = 8

    eval_interval: int = 1

    sgd_lr: float = 0.1
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 0.0002
    sgd_nesterov: bool = True

    cosine_shed_eta_min: float = 0.0001
    log_folder: str = "logs"


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


def train_network(cfg: TrainingConfig):
    if cfg.name is None:
        now = datetime.now()
        training_name = now.strftime("%H_%M_%d_%m_%Y")
    logger.info("Starting training")
    logger.info(f"Using {cfg.features}")
    train_sampler = Sampler(cfg.window_length, cfg.sampler_per_window)
    train_set = PoseDataset(
        cfg.train_file,
        cfg.features,
        train_sampler
    )
    train_loader = DataLoader(train_set, cfg.train_batch_size, True, num_workers=4, pin_memory=True)

    test_sampler = Sampler(cfg.window_length, cfg.sampler_per_window, True, cfg.test_clips_count)
    test_set = PoseDataset(
        cfg.test_file,
        cfg.features,
        test_sampler
    )
    test_loader = DataLoader(test_set, cfg.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(cfg.device)
    channels = calculate_channels(cfg.features, 2)
    model = create_stgcnpp(60, channels)
    model.to(device)

    logs_path = os.path.join(cfg.log_folder, cfg.name)
    os.mkdir(logs_path)

    write_log(logs_path, f"Training with features: {', '.join(cfg.features)}")
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr, momentum=cfg.sgd_momentum,
                                weight_decay=cfg.sgd_weight_decay, nesterov=cfg.sgd_nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.cosine_shed_eta_min)

    all_eval_stats = []
    for epoch in range(cfg.epochs):
        logger.info(f"Current epoch: {epoch+1}/{cfg.epochs}")
        train_start_time = time.time()
        training_stats = train_epoch(model, loss_func, train_loader, optimizer, scheduler, device)
        train_end_time = time.time()

        write_log(logs_path, f"[{epoch}] - training stats - {','.join([str(x) for x in training_stats])}")
        write_log(logs_path, f"[{epoch}] - training time - {timedelta(seconds=train_end_time - train_start_time)}")

        if (epoch+1) % cfg.eval_interval == 0:
            eval_start_time = time.time()
            eval_stats = test_epoch(model, test_loader, loss_func, device)
            eval_end_time = time.time()
            all_eval_stats.append(eval_stats)

            write_log(logs_path, f"[{epoch}] - eval stats - {','.join([str(x) for x in eval_stats])}")
            write_log(logs_path, f"[{epoch}] - eval time - {timedelta(seconds=eval_end_time - eval_start_time)}")
        # Save model
        torch.save(model.state_dict(), os.path.join(logs_path, f"epoch_{epoch}.pth"))
    best_eval_epoch = argmax([x[1] for x in all_eval_stats])
    best_acc = all_eval_stats[best_eval_epoch][1]
    logger.info(f"Best top1 accuracy {best_acc:.2%} at epoch {best_eval_epoch}")


def main():
    cfg = TrainingConfig("base_joints", "stgcnpp", 80, "cuda:0", ["joints"], 64, 32,
                         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl", 64,
                         "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl", 128, 8, 1, 0.1,
                         0.9, 0.0002, True, 0.00001)
    train_network(cfg)
