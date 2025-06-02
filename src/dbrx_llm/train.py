import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_duration: Optional[int] = None,
) -> float:
    """Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        data_loader (DataLoader): DataLoader providing training batches.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        epoch (int): Current epoch number.
        log_interval (int): How many batches to wait before logging training status.
        max_duration (Optional[int]): Maximum number of batches to process. If None, process all batches.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        total_loss += loss

        mlflow.log_metric("train_loss", loss)
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(input_ids),
                    len(data_loader) * len(input_ids),
                    100.0 * batch_idx / len(data_loader),
                    loss,
                )
            )
        if max_duration is not None:
            if batch_idx == max_duration:
                break
    return total_loss / len(data_loader)
