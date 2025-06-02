import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class AverageMeter:
    """Computes and stores the average and current value for loss and accuracy metrics."""

    def __init__(self) -> None:
        """Initialize the AverageMeter with zeroed metrics."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.loss = 0
        self.correct = 0
        self.total = 0

    def update(self, loss: float, correct: int, total: int) -> None:
        """Update the metrics with new values.

        Args:
            loss (float): Current loss value.
            correct (int): Number of correct predictions.
            total (int): Total number of predictions.
        """
        self.loss = loss
        self.correct = correct
        self.total = total

    def all_reduce(self, device: torch.device) -> Tuple[float, float]:
        """Reduce metrics across all distributed processes.

        Args:
            device (torch.device): Device to perform reduction on.

        Returns:
            Tuple[float, float]: Reduced test loss and accuracy.
        """
        total = torch.tensor([self.loss, self.correct, self.total], device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        loss, correct, total = total.tolist()
        test_loss = loss / total
        test_acc = correct / total
        return test_loss, test_acc

    def reduce(self) -> Tuple[float, float]:
        """Calculate final metrics without distributed reduction.

        Returns:
            Tuple[float, float]: Test loss and accuracy.
        """
        test_loss = self.loss / self.total
        test_acc = self.correct / self.total
        return test_loss, test_acc


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    avg_meter: AverageMeter,
    log_interval: int,
    max_duration: Optional[int] = None,
) -> Tuple[float, int, int]:
    """Evaluate the model for one epoch.

    Args:
        model (nn.Module): The neural network model to evaluate.
        data_loader (DataLoader): DataLoader providing evaluation batches.
        avg_meter (AverageMeter): Meter to track evaluation metrics.
        log_interval (int): How many batches to wait before logging evaluation status.
        max_duration (Optional[int]): Maximum number of batches to process. If None, process all batches.

    Returns:
        Tuple[float, int, int]: Final loss, number of correct predictions, and total predictions.
    """
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(data_loader):
        device = torch.device("cuda")
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )
        outputs = model(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels).item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        avg_meter.update(loss, correct, total)
        if batch_idx % log_interval == 0:
            print(
                "Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(input_ids),
                    len(data_loader) * len(input_ids),
                    100.0 * batch_idx / len(data_loader),
                    loss,
                )
            )
        if max_duration is not None:
            if batch_idx == max_duration:
                break
    return loss, correct, total
