from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def create_logger(exp_dir: str, model_name: str) -> tb.SummaryWriter:
    """
    Create a tensorboard logger with timestamped directory.
    
    This code was written by GitHub Copilot
    
    Args:
        exp_dir: Base experiment directory
        model_name: Name of the model being trained
        
    Returns:
        SummaryWriter instance for logging
    """
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    return tb.SummaryWriter(log_dir)


def log_metrics(logger: tb.SummaryWriter, metrics: dict, step: int, prefix: str = ""):
    """
    Log multiple metrics to tensorboard.
    
    This code was written by GitHub Copilot
    
    Args:
        logger: TensorBoard logger
        metrics: Dictionary of metric name -> value
        step: Global step counter
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.add_scalar(full_name, value, step)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, checkpoint_dir: Path):
    """
    Save model checkpoint.
    
    This code was written by GitHub Copilot
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path