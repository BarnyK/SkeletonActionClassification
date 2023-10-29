import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pose_dataset import PoseDataset
from datasets.sampler import Sampler
from datasets.transform_wrappers import calculate_channels
from models import create_stgcnpp

logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO (or other level of your choice)
logger = logging.getLogger(__name__)


def train_epoch(model, loss_func, loader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    current_lr = optimizer.param_groups[0]['lr']
    top1_count = 0
    sample_count = 0
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

        mean_loss = float(running_loss / len(loader))
        accuracy = top1_count / sample_count * 100.0
        tq.set_description(
            f"LR: {round(current_lr, 4):0<6} Loss: {round(mean_loss, 4):0<7} Accuracy: {round(accuracy, 2)}")

    mean_loss = float(running_loss / len(loader))
    accuracy = top1_count / sample_count * 100.0
    tq.set_description(
        f"LR: {round(current_lr, 4):0<6} Loss: {round(mean_loss, 4):0<7} Accuracy: {round(accuracy, 2)}")
    scheduler.step()


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

            reco_top5 = torch.topk(out, 5)[1]
            top5_count += sum([labels[n] in reco_top5[n, :] for n in range(batch_size)])
        running_loss += loss.item()
    top1_accuracy = top1_count / sample_count
    top5_accuracy = top5_count / sample_count
    mean_loss = running_loss / len(loader)
    logger.info(f"Top1 accuracy: {top1_accuracy:.2%}")
    logger.info(f"Top5 accuracy: {top5_accuracy:.2%}")
    logger.info(f"Mean loss: {mean_loss:.2}")



def train_model(model, train_loder, test_loader, device, epochs: int):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    for epoch in range(epochs):
        logger.info(f"Current epoch: {epoch}")
        train_epoch(model, loss_func, train_loder, optimizer, scheduler, device)
        test_epoch(model, test_loader, loss_func, device)


def main_test():
    feature_set = ["joints", "joint_motion", "angles"]
    train_sampler = Sampler(64, 64)
    train_set = PoseDataset(
        "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.train.pkl",
        feature_set,
        train_sampler
    )
    train_loader = DataLoader(train_set, 16, True, num_workers=4, pin_memory=True)

    test_sampler = Sampler(64, 64, True, 8)
    test_set = PoseDataset(
        "/media/barny/SSD4/MasterThesis/Data/prepped_data/test1/ntu_xsub.test.pkl",
        feature_set,
        test_sampler
    )
    test_loader = DataLoader(test_set, 8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda:0')
    channels = calculate_channels(feature_set, 2)
    model = create_stgcnpp(60, channels)
    model.to(device)

    epochs = 40
    train_model(model, train_loader, test_loader, device, epochs)

    pass


if __name__ == "__main__":
    main_test()
