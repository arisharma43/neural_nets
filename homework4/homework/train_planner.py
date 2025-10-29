"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .datasets.road_dataset import load_data
from .logger import create_logger, log_metrics
from .metrics import PlannerMetric
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    dataset_path: str = "drive_data",
    **kwargs,
):
    """
    Train a planner model.
    
    This code was written by GitHub Copilot
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create logger
    logger = create_logger(exp_dir, model_name)

    # Load model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load data
    if model_name == "vit_planner":
        # ViT uses images
        transform_pipeline = "default"
    else:
        # MLP and Transformer use track points only
        transform_pipeline = "state_only"

    train_data = load_data(
        f"{dataset_path}/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    
    val_data = load_data(
        f"{dataset_path}/val",
        transform_pipeline=transform_pipeline,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    global_step = 0
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()
        epoch_train_losses = []

        for batch in train_data:
            # Move data to device
            if model_name == "vit_planner":
                image = batch["image"].to(device)
                inputs = {"image": image}
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                inputs = {"track_left": track_left, "track_right": track_right}
                
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            optimizer.zero_grad()
            pred_waypoints = model(**inputs)
            
            # Compute MSE loss only on valid waypoints
            # MSE loss: (pred - target)^2, shape: (B, n_waypoints, 2)  
            loss_per_element = (pred_waypoints - waypoints) ** 2
            # Apply mask to exclude invalid waypoints from loss
            # waypoints_mask: (B, n_waypoints) -> (B, n_waypoints, 1)
            mask_expanded = waypoints_mask.unsqueeze(-1).float()  # Convert to float for multiplication
            masked_loss_per_element = loss_per_element * mask_expanded
            
            # Average over all valid elements (both coordinates and waypoints)
            num_valid_elements = mask_expanded.sum() * 2  # multiply by 2 for x,y coordinates
            if num_valid_elements > 0:
                masked_loss = masked_loss_per_element.sum() / num_valid_elements
            else:
                masked_loss = torch.tensor(0.0, device=pred_waypoints.device, requires_grad=True)
            
            # Backward pass
            masked_loss.backward()
            optimizer.step()

            # Update metrics
            train_metric.add(pred_waypoints, waypoints, waypoints_mask)
            epoch_train_losses.append(masked_loss.item())

            # Log training loss every 100 steps
            if global_step % 100 == 0:
                logger.add_scalar("train/loss", masked_loss.item(), global_step)

            global_step += 1

        # Validation loop
        model.eval()
        val_metric.reset()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_data:
                # Move data to device
                if model_name == "vit_planner":
                    image = batch["image"].to(device)
                    inputs = {"image": image}
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    inputs = {"track_left": track_left, "track_right": track_right}
                    
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass
                pred_waypoints = model(**inputs)
                
                # Compute MSE loss only on valid waypoints
                # MSE loss: (pred - target)^2, shape: (B, n_waypoints, 2)
                loss_per_element = (pred_waypoints - waypoints) ** 2
                # Apply mask to exclude invalid waypoints from loss
                mask_expanded = waypoints_mask.unsqueeze(-1).float()
                masked_loss_per_element = loss_per_element * mask_expanded
                
                # Average over all valid elements
                num_valid_elements = mask_expanded.sum() * 2  # multiply by 2 for x,y coordinates
                if num_valid_elements > 0:
                    masked_loss = masked_loss_per_element.sum() / num_valid_elements
                else:
                    masked_loss = torch.tensor(0.0, device=pred_waypoints.device)
                
                # Update metrics
                val_metric.add(pred_waypoints, waypoints, waypoints_mask)
                epoch_val_losses.append(masked_loss.item())

        # Compute epoch metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()
        
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)

        # Log epoch metrics
        log_metrics(logger, {"loss": avg_train_loss, **train_metrics}, epoch, "epoch/train")
        log_metrics(logger, {"loss": avg_val_loss, **val_metrics}, epoch, "epoch/val")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model)

        # Print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} / {num_epoch:3d}: "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f} "
                f"val_long_err={val_metrics['longitudinal_error']:.4f} "
                f"val_lat_err={val_metrics['lateral_error']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True, 
                       choices=["mlp_planner", "transformer_planner", "vit_planner"])
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dataset_path", type=str, default="drive_data")

    # Model-specific hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MLP")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension for Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension for ViT")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ViT")

    args = parser.parse_args()
    train(**vars(args))
