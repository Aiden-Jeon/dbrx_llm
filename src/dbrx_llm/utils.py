import os
from time import time
from typing import Any

import torch


def create_log_dir(log_volume_dir: str) -> str:
    """Create a new log directory with a timestamp.

    Args:
        log_volume_dir (str): Base directory path where logs will be stored.

    Returns:
        str: Path to the newly created log directory.
    """
    log_dir = os.path.join(log_volume_dir, str(time()))
    os.makedirs(log_dir)
    return log_dir


def create_run_name(run_name: str) -> str:
    """Create a unique run name by appending a timestamp.

    Args:
        run_name (str): Base name for the run.

    Returns:
        str: Unique run name with timestamp appended.
    """
    return f"{run_name}-{str(time())}"


def save_checkpoint(log_dir: str, model: torch.nn.Module, epoch: int) -> None:
    """Save a model checkpoint to disk.

    Args:
        log_dir (str): Directory path where the checkpoint will be saved.
        model (torch.nn.Module): The PyTorch model to save.
        epoch (int): Current epoch number for the checkpoint filename.
    """
    filepath = log_dir + "/checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    state = {
        "model": model.state_dict(),
    }
    torch.save(state, filepath)
