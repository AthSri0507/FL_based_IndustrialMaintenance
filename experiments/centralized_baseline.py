"""Centralized Training Pipeline — Baseline for Federated Learning Comparison.

This module provides a centralized training pipeline that serves as a baseline
for comparing federated learning approaches. It trains models on all data
combined (as opposed to distributed across clients).

Usage:
    python -m experiments.centralized_baseline --config configs/centralized.yaml
    
    # Or with command-line args:
    python -m experiments.centralized_baseline \
        --data-dir data/raw \
        --task rul \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3

The pipeline supports:
- RUL (Remaining Useful Life) prediction using TCN-based regressor
- Fault classification using TCN-based classifier
- Configurable hyperparameters via YAML or CLI
- Logging, checkpointing, and reproducibility controls
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tcn import (
    RULPredictor,
    FaultClassifier,
    rul_mse_loss,
    compute_rul_metrics,
    fault_ce_loss,
    compute_classification_metrics,
)
from src.data.segment import segment_windows


# --------------------- Configuration ---------------------


@dataclass
class CentralizedConfig:
    """Configuration for centralized training pipeline."""

    # Data profile: "clean" for existing behavior, "non_iid_hard" for stress testing
    # NOTE: Changing data_profile ONLY affects data generation.
    # It does NOT alter: training loops, model architecture, aggregation,
    # metrics, logging, or random seed handling.
    data_profile: str = "clean"

    # Data settings
    data_dir: str = "data/raw"
    data_files: List[str] = field(default_factory=list)  # specific files to load
    window_size: int = 50  # W: sliding window length
    hop_size: int = 10  # H: hop between windows
    normalize_windows: bool = True
    val_split: float = 0.2  # fraction of data for validation
    test_split: float = 0.1  # fraction of data for testing

    # Task settings
    task: str = "rul"  # "rul" or "classification"
    num_classes: int = 2  # for classification task

    # Model architecture
    num_layers: int = 4
    hidden_dim: int = 64
    kernel_size: int = 3
    dropout: float = 0.2
    fc_hidden: int = 32

    # Training settings
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam" or "sgd"
    lr_scheduler: str = "none"  # "none", "step", "cosine", "plateau"
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Checkpointing and logging
    output_dir: str = "experiments/outputs/centralized"
    checkpoint_every: int = 5  # save checkpoint every N epochs
    log_every: int = 1  # log metrics every N batches (0 = per epoch only)
    save_best: bool = True

    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: Optional[str] = None) -> CentralizedConfig:
    """Load configuration from YAML file or return defaults."""
    config = CentralizedConfig()

    if config_path and os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            for key, value in cfg_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except ImportError:
            # Fallback to JSON if YAML not available
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    cfg_dict = json.load(f)
                for key, value in cfg_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

    return config


# --------------------- Data Loading ---------------------


def load_data_from_directory(
    data_dir: str,
    file_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and combine data from multiple files in a directory.

    Args:
        data_dir: Path to directory containing data files
        file_patterns: Optional list of specific filenames to load

    Returns:
        X: Features array of shape (N, T, C) or (N, C)
        y: Labels array of shape (N,)
    """
    import pandas as pd

    data_path = Path(data_dir)
    all_X = []
    all_y = []

    # Find data files
    if file_patterns:
        files = [data_path / f for f in file_patterns if (data_path / f).exists()]
    else:
        files = list(data_path.glob("*.parquet")) + list(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    for fpath in sorted(files):
        try:
            if str(fpath).endswith(".parquet"):
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath)

            # Expect last column to be label (or column named 'label' / 'rul')
            if "label" in df.columns:
                y_col = "label"
            elif "rul" in df.columns:
                y_col = "rul"
            elif "RUL" in df.columns:
                y_col = "RUL"
            else:
                # Assume last column is target
                y_col = df.columns[-1]

            feature_cols = [c for c in df.columns if c != y_col and c not in {"index", "timestamp", "client_id", "unit_id"}]

            X = df[feature_cols].values.astype(np.float32)
            y = df[y_col].values.astype(np.float32)

            all_X.append(X)
            all_y.append(y)

        except Exception as e:
            logging.warning(f"Could not load {fpath}: {e}")
            continue

    if not all_X:
        raise ValueError("No data could be loaded from the specified directory")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    return X_combined, y_combined


def create_synthetic_data(
    num_samples: int = 1000,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing the pipeline.

    Args:
        num_samples: Number of samples to generate
        seq_length: Length of each time series
        num_channels: Number of sensor channels
        task: "rul" or "classification"
        num_classes: Number of classes for classification task
        seed: Random seed

    Returns:
        X: Features of shape (num_samples, seq_length, num_channels)
        y: Labels of shape (num_samples,)
    """
    rng = np.random.RandomState(seed)

    # Generate realistic-looking sensor data with trends
    X = np.zeros((num_samples, seq_length, num_channels), dtype=np.float32)

    for i in range(num_samples):
        # Base signals with different frequencies
        t = np.linspace(0, 4 * np.pi, seq_length)
        for c in range(num_channels):
            freq = 0.5 + c * 0.2
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = 1.0 + rng.uniform(-0.3, 0.3)
            noise = rng.randn(seq_length) * 0.1

            # Add degradation trend for some samples
            degradation = 0.0
            if i > num_samples // 2:
                degradation = np.linspace(0, rng.uniform(0.5, 2.0), seq_length)

            X[i, :, c] = amplitude * np.sin(freq * t + phase) + noise + degradation

    if task == "rul":
        # Generate RUL values: higher degradation = lower RUL
        y = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            # RUL based on signal variance and trend
            signal_var = np.var(X[i])
            trend = np.mean(X[i, -10:]) - np.mean(X[i, :10])
            y[i] = max(0, 100 - 20 * signal_var - 10 * abs(trend) + rng.uniform(-5, 5))
    else:
        # Classification: assign classes based on signal characteristics
        y = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            # Simple rule: high variance = fault
            signal_var = np.var(X[i])
            y[i] = min(num_classes - 1, int(signal_var * num_classes))

    return X, y


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    config: CentralizedConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train, validation, and test data loaders.

    Args:
        X: Features of shape (N, W, C) or (N, T, C)
        y: Labels of shape (N,)
        config: Training configuration

    Returns:
        train_loader, val_loader, test_loader
    """
    # Apply sliding window segmentation if data is 2D (raw time series)
    if X.ndim == 2:
        X = segment_windows(X, W=config.window_size, H=config.hop_size, normalize=config.normalize_windows)
        # Adjust y to match number of windows
        num_windows = X.shape[0]
        if len(y) > num_windows:
            # Take labels at window end positions
            indices = list(range(config.window_size - 1, len(y), config.hop_size))[:num_windows]
            y = y[indices]

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if config.task == "rul":
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        y_tensor = torch.tensor(y, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train/val/test
    total_size = len(dataset)
    test_size = int(total_size * config.test_split)
    val_size = int(total_size * config.val_split)
    train_size = total_size - val_size - test_size

    # Use generator for reproducibility
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )

    return train_loader, val_loader, test_loader


# --------------------- Model Factory ---------------------


def create_model(num_channels: int, config: CentralizedConfig) -> nn.Module:
    """Create model based on task type and configuration.

    Args:
        num_channels: Number of input channels (sensor features)
        config: Training configuration

    Returns:
        Model instance (RULPredictor or FaultClassifier)
    """
    if config.task == "rul":
        model = RULPredictor(
            num_channels=num_channels,
            num_layers=config.num_layers,
            kernel_size=config.kernel_size,
            hidden=config.hidden_dim,
            dropout=config.dropout,
            fc_hidden=config.fc_hidden,
        )
    else:
        model = FaultClassifier(
            num_channels=num_channels,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            kernel_size=config.kernel_size,
            hidden=config.hidden_dim,
            dropout=config.dropout,
            fc_hidden=config.fc_hidden,
        )

    return model


def create_optimizer(model: nn.Module, config: CentralizedConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: CentralizedConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on configuration."""
    if config.lr_scheduler == "none":
        return None
    elif config.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    elif config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_gamma, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


# --------------------- Training Loop ---------------------


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update early stopping state.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class TrainingLogger:
    """Logger for training metrics and progress."""

    def __init__(self, output_dir: str, experiment_name: str = "centralized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: Dict[str, List] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": [],
            "time_per_epoch": [],
        }

        # Set up file logging
        log_file = self.output_dir / f"{experiment_name}_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        time_elapsed: float,
    ):
        """Log metrics for an epoch."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_metrics"].append(train_metrics)
        self.history["val_metrics"].append(val_metrics)
        self.history["lr"].append(lr)
        self.history["time_per_epoch"].append(time_elapsed)

        # Format metrics for logging
        train_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())

        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.2e} | Time: {time_elapsed:.2f}s"
        )
        self.logger.info(f"  Train Metrics: {train_str}")
        self.logger.info(f"  Val Metrics:   {val_str}")

    def save_history(self):
        """Save training history to JSON file."""
        history_file = self.output_dir / f"{self.experiment_name}_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_file}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: str,
    task: str,
) -> Tuple[float, Dict]:
    """Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        task: "rul" or "classification"

    Returns:
        avg_loss: Average training loss
        metrics: Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        if task == "rul":
            outputs = outputs.squeeze()
        loss = loss_fn(outputs, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        all_preds.append(outputs.detach().cpu())
        all_targets.append(y_batch.detach().cpu())

    avg_loss = total_loss / num_samples

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if task == "rul":
        metrics = compute_rul_metrics(all_preds, all_targets)
    else:
        metrics = compute_classification_metrics(all_preds, all_targets)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    device: str,
    task: str,
) -> Tuple[float, Dict]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to evaluate on
        task: "rul" or "classification"

    Returns:
        avg_loss: Average loss
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        if task == "rul":
            outputs = outputs.squeeze()
        loss = loss_fn(outputs, y_batch)

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        all_preds.append(outputs.cpu())
        all_targets.append(y_batch.cpu())

    avg_loss = total_loss / num_samples

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if task == "rul":
        metrics = compute_rul_metrics(all_preds, all_targets)
    else:
        metrics = compute_classification_metrics(all_preds, all_targets)

    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: CentralizedConfig,
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        val_loss: Validation loss
        config: Training configuration
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": asdict(config),
    }

    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)


def train(
    config: CentralizedConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_channels: int,
) -> Dict:
    """Main training loop.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_channels: Number of input channels

    Returns:
        results: Dictionary containing training history and final metrics
    """
    # Set up reproducibility
    if config.deterministic:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    device = config.device
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model, optimizer, scheduler
    model = create_model(num_channels, config).to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Loss function
    if config.task == "rul":
        loss_fn = rul_mse_loss
    else:
        loss_fn = fault_ce_loss

    # Training utilities
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    logger = TrainingLogger(output_dir, experiment_name="centralized")

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.logger.info(f"Starting centralized training for {config.task} task")
    logger.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.logger.info(f"Device: {device}")

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, config.task
        )

        # Validate
        val_loss, val_metrics = evaluate(
            model, val_loader, loss_fn, device, config.task
        )

        epoch_time = time.time() - epoch_start

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log progress
        logger.log_epoch(
            epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr, epoch_time
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config, output_dir, is_best=False)

        if config.save_best and is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, config, output_dir, is_best=True)

        # Early stopping
        if early_stopping.step(val_loss):
            logger.logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Final evaluation on test set
    logger.logger.info("Evaluating on test set...")
    
    # Load best model if available
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device, config.task)

    logger.logger.info(f"Test Loss: {test_loss:.4f}")
    logger.logger.info(f"Test Metrics: {test_metrics}")

    # Save final results
    results = {
        "config": asdict(config),
        "training_history": logger.history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.logger.info(f"Results saved to {results_path}")

    logger.save_history()

    return results


# --------------------- CLI Interface ---------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Centralized Training Pipeline for Federated Learning Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML/JSON configuration file"
    )

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=1000, help="Number of synthetic samples"
    )

    # Task arguments
    parser.add_argument(
        "--task", type=str, default="rul", choices=["rul", "classification"],
        help="Training task"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of classes for classification"
    )

    # Model arguments
    parser.add_argument("--num-layers", type=int, default=4, help="Number of TCN layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--lr-scheduler", type=str, default="none", help="LR scheduler")

    # Data processing
    parser.add_argument("--window-size", type=int, default=50, help="Window size W")
    parser.add_argument("--hop-size", type=int, default=10, help="Hop size H")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs/centralized",
        help="Output directory",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    """Main entry point for centralized training pipeline."""
    args = parse_args()

    # Load config from file or use defaults
    config = load_config(args.config)

    # Override with CLI arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.task:
        config.task = args.task
    if args.num_classes:
        config.num_classes = args.num_classes
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.kernel_size:
        config.kernel_size = args.kernel_size
    if args.dropout:
        config.dropout = args.dropout
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.optimizer:
        config.optimizer = args.optimizer
    if args.lr_scheduler:
        config.lr_scheduler = args.lr_scheduler
    if args.window_size:
        config.window_size = args.window_size
    if args.hop_size:
        config.hop_size = args.hop_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed:
        config.seed = args.seed

    # =================================================================
    # NON-IID HARD MODE DATA DISPATCH
    # NOTE: This ONLY affects data generation.
    # Training loops, model architecture, metrics, logging, and random 
    # seed handling remain COMPLETELY UNCHANGED.
    # =================================================================
    if config.data_profile == "non_iid_hard":
        from src.data.non_iid_generator import generate_non_iid_hard_centralized
        
        print("Using NON-IID HARD data profile (centralized)")
        print("  - Merging heterogeneous client data for centralized training")
        
        X, y = generate_non_iid_hard_centralized(
            num_clients=5,  # Default number of clients
            seq_length=100,
            num_channels=14,
            task=config.task,
            num_classes=config.num_classes,
            seed=config.seed,
        )
        num_channels = X.shape[2]
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Number of channels: {num_channels}")
    # =================================================================
    # END NON-IID HARD MODE DATA DISPATCH
    # =================================================================
    # =================================================================
    # CLEAN MODE (DEFAULT) - EXISTING BEHAVIOR UNCHANGED
    # =================================================================
    elif args.synthetic:
        print("Using synthetic data for testing...")
        X, y = create_synthetic_data(
            num_samples=args.synthetic_samples,
            seq_length=100,
            num_channels=14,
            task=config.task,
            num_classes=config.num_classes,
            seed=config.seed,
        )
        num_channels = X.shape[2]
    else:
        try:
            print(f"Loading data from {config.data_dir}...")
            X, y = load_data_from_directory(config.data_dir, config.data_files or None)
            
            # Apply windowing if needed
            if X.ndim == 2:
                X = segment_windows(X, W=config.window_size, H=config.hop_size, normalize=config.normalize_windows)
                # Adjust labels
                num_windows = X.shape[0]
                if len(y) > num_windows:
                    indices = list(range(config.window_size - 1, len(y), config.hop_size))[:num_windows]
                    y = y[indices]
            
            num_channels = X.shape[2] if X.ndim == 3 else X.shape[1]
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not load real data: {e}")
            print("Falling back to synthetic data...")
            X, y = create_synthetic_data(
                num_samples=1000,
                seq_length=100,
                num_channels=14,
                task=config.task,
                num_classes=config.num_classes,
                seed=config.seed,
            )
            num_channels = X.shape[2]

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of channels: {num_channels}")

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, config)

    # Run training
    results = train(config, train_loader, val_loader, test_loader, num_channels)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Metrics: {results['test_metrics']}")
    print(f"Total Epochs: {results['total_epochs']}")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
