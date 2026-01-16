"""Simulated Client Trainer for Federated Learning.

This module provides an industrial-grade simulated client for federated learning
experiments. It is designed to work with time-series data (sensor windows) and
supports both RUL regression and fault classification tasks.

Key features:
- Proper (N, W, C) time-series window support
- Client-side train/validation split with leakage protection
- Task-aware loss functions and metrics
- Per-channel normalization
- Comprehensive metric reporting (MAE, RMSE, Score, Accuracy, F1, etc.)
- Simulation of real-world client behaviors (dropout, delays, failures)
- Reproducible training with seed control

This is a **simulation** module for FL research. It does not include:
- Network communication
- Serialization/deserialization
- Real fault tolerance

For production FL deployments, use a proper FL framework (Flower, PySyft, etc.).
"""

import copy
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------- Enums and Configuration ----------------------


class TaskType(Enum):
    """Supported task types for client training."""
    RUL = "rul"  # Remaining Useful Life regression
    CLASSIFICATION = "classification"  # Fault classification
    MULTI_TASK = "multi_task"  # Combined RUL + classification


@dataclass
class ClientConfig:
    """Configuration for simulated client training.
    
    This dataclass encapsulates all training hyperparameters and behaviors
    for a simulated FL client.
    """
    # Task configuration
    task: TaskType = TaskType.RUL
    num_classes: int = 2  # For classification task
    
    # Data configuration
    val_split: float = 0.2  # Fraction of local data for validation
    normalize_per_channel: bool = True  # Per-channel normalization for windows
    normalize_rul: bool = True  # Normalize RUL targets to [0, 1] for stable training
    rul_max: Optional[float] = None  # Max RUL for normalization (auto-detected if None)
    
    # Training hyperparameters
    batch_size: int = 16
    local_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    
    # Early stopping (on validation loss)
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    
    # Learning rate scheduler
    lr_scheduler: str = "none"  # "none", "step", "plateau"
    lr_step_size: int = 5
    lr_gamma: float = 0.5
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Device
    device: str = "cpu"
    
    # Simulation behaviors
    simulate_dropout: bool = False  # Randomly fail to participate
    dropout_probability: float = 0.1
    simulate_delay: bool = False  # Add artificial delay
    delay_mean_seconds: float = 0.0
    delay_std_seconds: float = 0.0
    simulate_partial_work: bool = False  # Sometimes train fewer epochs
    partial_work_probability: float = 0.1
    
    # Multi-task weights (for MULTI_TASK mode)
    rul_weight: float = 1.0
    classification_weight: float = 1.0


# ---------------------- Metrics Functions ----------------------


def compute_rul_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor,
    rul_scale: float = 1.0,
) -> Dict[str, float]:
    """Compute comprehensive RUL regression metrics.
    
    Args:
        pred: Predicted RUL values (N,) - may be normalized
        target: True RUL values (N,) - may be normalized
        rul_scale: Scale factor to denormalize for metric reporting.
                   If training on normalized RUL (y/max_rul), pass max_rul here
                   to report metrics in original scale.
        
    Returns:
        Dictionary with mae, rmse, mse, score (asymmetric NASA scoring)
        All metrics are reported in ORIGINAL (denormalized) scale.
    """
    with torch.no_grad():
        pred = pred.float().flatten()
        target = target.float().flatten()
        
        # Denormalize for metric computation
        pred_denorm = pred * rul_scale
        target_denorm = target * rul_scale
        
        # MSE (in original scale)
        mse = F.mse_loss(pred_denorm, target_denorm).item()
        
        # MAE (in original scale)
        mae = (pred_denorm - target_denorm).abs().mean().item()
        
        # RMSE (in original scale)
        rmse = mse ** 0.5
        
        # Asymmetric scoring function (NASA turbofan style)
        # Penalizes late predictions more than early predictions
        # Note: score uses denormalized values
        d = pred_denorm - target_denorm  # positive = late (bad), negative = early (less bad)
        a1, a2 = 13.0, 10.0
        early = d < 0
        late = ~early
        score_tensor = torch.zeros_like(d)
        score_tensor[early] = torch.exp(-d[early] / a1) - 1
        score_tensor[late] = torch.exp(d[late] / a2) - 1
        score = score_tensor.sum().item()
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "score": score,
        }


def compute_classification_metrics(
    logits: torch.Tensor, 
    target: torch.Tensor,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        logits: Predicted logits (N, num_classes)
        target: True class labels (N,)
        num_classes: Number of classes
        
    Returns:
        Dictionary with accuracy, precision, recall, f1, per-class metrics
    """
    with torch.no_grad():
        if logits.dim() == 1:
            # Binary case with single logit
            preds = (logits > 0).long()
        else:
            preds = logits.argmax(dim=1)
        
        target = target.long().flatten()
        preds = preds.flatten()
        
        # Overall accuracy
        accuracy = (preds == target).float().mean().item()
        
        # Per-class precision, recall, F1 (macro average)
        precisions = []
        recalls = []
        f1s = []
        
        for c in range(num_classes):
            tp = ((preds == c) & (target == c)).sum().float()
            fp = ((preds == c) & (target != c)).sum().float()
            fn = ((preds != c) & (target == c)).sum().float()
            
            precision = (tp / (tp + fp + 1e-8)).item()
            recall = (tp / (tp + fn + 1e-8)).item()
            f1 = (2 * precision * recall / (precision + recall + 1e-8))
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # Macro averages
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        # For binary classification, also report positive class metrics
        if num_classes == 2:
            return {
                "accuracy": accuracy,
                "precision": precisions[1],  # Positive class
                "recall": recalls[1],
                "f1": f1s[1],
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        else:
            return {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }


# ---------------------- Data Utilities ----------------------


def normalize_windows_per_channel(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize windows per channel (zero mean, unit std).
    
    Args:
        X: Tensor of shape (N, W, C) - windows of sensor readings
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape
    """
    if X.dim() != 3:
        return X  # Only normalize 3D tensors
    
    # Compute mean and std per sample per channel
    # X: (N, W, C) -> compute over W dimension
    mean = X.mean(dim=1, keepdim=True)  # (N, 1, C)
    std = X.std(dim=1, keepdim=True)    # (N, 1, C)
    std = torch.clamp(std, min=eps)
    
    return (X - mean) / std


def split_by_time_series_units(
    X: torch.Tensor,
    y: torch.Tensor,
    unit_ids: Optional[torch.Tensor],
    val_fraction: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data by unit/machine IDs to prevent temporal leakage.
    
    In predictive maintenance, windows from the same unit should not appear
    in both train and validation sets, as this causes data leakage.
    
    Args:
        X: Features (N, W, C) or (N, features)
        y: Labels (N,)
        unit_ids: Unit/machine identifiers (N,). If None, falls back to random split.
        val_fraction: Fraction of units for validation
        seed: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    rng = np.random.RandomState(seed)
    
    if unit_ids is None:
        # Fallback: random split (may have leakage but better than nothing)
        n = len(X)
        indices = np.arange(n)
        rng.shuffle(indices)
        
        val_size = int(n * val_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]
    
    # Unit-level split
    unique_units = torch.unique(unit_ids).numpy()
    rng.shuffle(unique_units)
    
    num_val_units = max(1, int(len(unique_units) * val_fraction))
    val_units = set(unique_units[:num_val_units])
    
    unit_ids_np = unit_ids.numpy()
    val_mask = np.isin(unit_ids_np, list(val_units))
    train_mask = ~val_mask
    
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


# ---------------------- Loss Functions ----------------------


def get_loss_function(task: TaskType) -> Callable:
    """Get appropriate loss function for task type.
    
    Args:
        task: TaskType enum
        
    Returns:
        Loss function callable
    """
    if task == TaskType.RUL:
        return F.mse_loss
    elif task == TaskType.CLASSIFICATION:
        return F.cross_entropy
    elif task == TaskType.MULTI_TASK:
        # Multi-task loss is handled separately in training loop
        return None
    else:
        raise ValueError(f"Unknown task type: {task}")


# ---------------------- Simulated Client Trainer ----------------------


class SimulatedClientTrainer:
    """Industrial-grade simulated client for federated learning experiments.
    
    This class simulates a client in a federated learning setup. It handles:
    - Time-series window data (N, W, C) for sensor-based predictive maintenance
    - Task-aware training (RUL regression, fault classification, multi-task)
    - Proper train/validation split with temporal leakage protection
    - Comprehensive metrics reporting
    - Simulation of real-world client behaviors
    
    Example usage:
        >>> client = SimulatedClientTrainer(
        ...     client_id=0,
        ...     data=(X_windows, y_labels),  # (N, W, C), (N,)
        ...     config=ClientConfig(task=TaskType.RUL)
        ... )
        >>> result = client.train_local(global_state={"model": model})
    
    Note:
        This is a **simulation** for research purposes. It does not include
        networking, serialization, or real fault tolerance. For production
        FL deployments, consider using established FL frameworks.
    """
    
    def __init__(
        self,
        client_id: int,
        data: Optional[Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # X, y, unit_ids
            pd.DataFrame,
            str,  # File path
        ]] = None,
        config: Optional[ClientConfig] = None,
    ):
        """Initialize simulated client.
        
        Args:
            client_id: Unique identifier for this client
            data: Client's local data. Can be:
                - Tuple (X, y) where X is (N, W, C) or (N, features)
                - Tuple (X, y, unit_ids) for unit-aware splitting
                - DataFrame with features and target column
                - Path to parquet/csv file
            config: Client configuration (uses defaults if None)
        """
        self.client_id = int(client_id)
        self._raw_data = data
        self.config = config or ClientConfig()
        
        # Cached processed data
        self._X: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self._unit_ids: Optional[torch.Tensor] = None
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._X_val: Optional[torch.Tensor] = None
        self._y_val: Optional[torch.Tensor] = None
        self._data_loaded = False
        
        # RUL normalization tracking
        self._rul_scale: float = 1.0  # Scale factor for denormalization
        
        # Training state
        self._last_train_metrics: Dict = {}
        self._last_val_metrics: Dict = {}
    
    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        if self._X_train is not None:
            return len(self._X_train)
        return 0
    
    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        if self._X_val is not None:
            return len(self._X_val)
        return 0
    
    @property
    def rul_scale(self) -> float:
        """RUL scale factor for denormalization."""
        return self._rul_scale
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess client data.
        
        Returns:
            Tuple (X, y) of processed tensors
            
        Raises:
            ValueError: If data format is unsupported or missing
        """
        if self._data_loaded:
            return self._X, self._y
        
        X, y, unit_ids = self._parse_raw_data()
        
        # Ensure proper dtypes
        X = X.float()
        if self.config.task == TaskType.CLASSIFICATION:
            y = y.long()
        else:
            y = y.float()
            
            # Normalize RUL targets to [0, 1] for stable training
            if self.config.normalize_rul:
                if self.config.rul_max is not None:
                    self._rul_scale = float(self.config.rul_max)
                else:
                    # Auto-detect max RUL from data
                    self._rul_scale = float(y.max().item())
                    if self._rul_scale == 0:
                        self._rul_scale = 1.0  # Avoid division by zero
                
                y = y / self._rul_scale
                logger.debug(f"Client {self.client_id}: Normalized RUL by scale={self._rul_scale:.2f}")
        
        # Per-channel normalization for time-series windows
        if self.config.normalize_per_channel and X.dim() == 3:
            X = normalize_windows_per_channel(X)
        
        self._X = X
        self._y = y
        self._unit_ids = unit_ids
        
        # Create train/val split
        self._create_train_val_split()
        
        self._data_loaded = True
        return self._X, self._y
    
    def _parse_raw_data(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Parse raw data into tensors.
        
        Returns:
            X, y, unit_ids (unit_ids may be None)
        """
        unit_ids = None
        
        if isinstance(self._raw_data, tuple):
            if len(self._raw_data) == 2:
                X, y = self._raw_data
            elif len(self._raw_data) == 3:
                X, y, unit_ids = self._raw_data
                if not isinstance(unit_ids, torch.Tensor):
                    unit_ids = torch.tensor(unit_ids)
            else:
                raise ValueError(f"Tuple data must have 2 or 3 elements, got {len(self._raw_data)}")
            
            X = X.clone() if isinstance(X, torch.Tensor) else torch.tensor(X)
            y = y.clone() if isinstance(y, torch.Tensor) else torch.tensor(y)
            return X, y, unit_ids
        
        if isinstance(self._raw_data, pd.DataFrame):
            return self._parse_dataframe(self._raw_data)
        
        if isinstance(self._raw_data, str):
            return self._load_from_file(self._raw_data)
        
        raise ValueError(
            "Unsupported or missing data. Provide (X, y), (X, y, unit_ids), "
            "DataFrame, or file path."
        )
    
    def _parse_dataframe(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Parse DataFrame into tensors."""
        # Identify target column
        if "label" in df.columns:
            y_col = "label"
        elif "rul" in df.columns:
            y_col = "rul"
        elif "RUL" in df.columns:
            y_col = "RUL"
        else:
            y_col = df.columns[-1]
        
        # Identify unit column
        unit_ids = None
        unit_col = None
        for col in ["unit_id", "unit", "machine_id", "engine_id"]:
            if col in df.columns:
                unit_col = col
                unit_ids = torch.tensor(df[unit_col].values)
                break
        
        # Feature columns
        exclude_cols = {y_col, unit_col, "index", "timestamp", "time", "cycle"}
        exclude_cols.discard(None)
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = torch.tensor(df[feature_cols].values).float()
        y = torch.tensor(df[y_col].values)
        
        return X, y, unit_ids
    
    def _load_from_file(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Load data from file."""
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".pt") or path.lower().endswith(".pth"):
            # PyTorch tensor file
            data = torch.load(path)
            if isinstance(data, dict):
                X = data.get("X", data.get("features"))
                y = data.get("y", data.get("labels", data.get("targets")))
                unit_ids = data.get("unit_ids", data.get("units"))
                return X, y, unit_ids
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                return data[0], data[1], data[2] if len(data) > 2 else None
        else:
            # Try CSV as fallback
            df = pd.read_csv(path)
        
        return self._parse_dataframe(df)
    
    def _create_train_val_split(self):
        """Create train/validation split with leakage protection."""
        seed = self.config.seed
        if seed is not None:
            seed = seed + self.client_id  # Per-client seed variation
        
        self._X_train, self._y_train, self._X_val, self._y_val = split_by_time_series_units(
            self._X,
            self._y,
            self._unit_ids,
            self.config.val_split,
            seed=seed,
        )
        
        logger.debug(
            f"Client {self.client_id}: train={len(self._X_train)}, val={len(self._X_val)}"
        )
    
    def _should_dropout(self) -> bool:
        """Check if client should simulate dropout (fail to participate)."""
        if not self.config.simulate_dropout:
            return False
        
        rng = np.random.RandomState(
            (self.config.seed or 0) + self.client_id + int(time.time() * 1000) % 10000
        )
        return rng.random() < self.config.dropout_probability
    
    def _simulate_delay(self):
        """Simulate network/computation delay."""
        if not self.config.simulate_delay:
            return
        
        rng = np.random.RandomState(
            (self.config.seed or 0) + self.client_id + int(time.time() * 1000) % 10000
        )
        delay = max(0, rng.normal(self.config.delay_mean_seconds, self.config.delay_std_seconds))
        if delay > 0:
            time.sleep(delay)
    
    def _get_effective_epochs(self) -> int:
        """Get number of epochs, possibly reduced for partial work simulation."""
        epochs = self.config.local_epochs
        
        if not self.config.simulate_partial_work:
            return epochs
        
        rng = np.random.RandomState(
            (self.config.seed or 0) + self.client_id + int(time.time() * 1000) % 10000
        )
        if rng.random() < self.config.partial_work_probability:
            # Do only 1 to (epochs-1) epochs
            return rng.randint(1, max(2, epochs))
        
        return epochs
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        train_dataset = TensorDataset(self._X_train, self._y_train)
        val_dataset = TensorDataset(self._X_val, self._y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.config.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        opt_name = self.config.optimizer.lower()
        
        if opt_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create LR scheduler based on config."""
        sched_name = self.config.lr_scheduler.lower()
        
        if sched_name == "none":
            return None
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=2,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")
    
    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Returns:
            avg_loss, metrics_dict
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
            
            # Handle output shape
            if self.config.task == TaskType.RUL:
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = loss_fn(outputs, y_batch.float())
            else:
                # Classification
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                loss = loss_fn(outputs, y_batch.long())
            
            loss.backward()
            optimizer.step()
            
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            all_preds.append(outputs.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
        
        avg_loss = total_loss / max(num_samples, 1)
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if self.config.task == TaskType.RUL:
            # Denormalize for metric reporting in original scale
            metrics = compute_rul_metrics(all_preds, all_targets, rul_scale=self._rul_scale)
        else:
            metrics = compute_classification_metrics(
                all_preds, all_targets, self.config.num_classes
            )
        
        metrics["loss"] = avg_loss
        return avg_loss, metrics
    
    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Callable,
        device: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on validation set.
        
        Returns:
            avg_loss, metrics_dict
        """
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            if self.config.task == TaskType.RUL:
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = loss_fn(outputs, y_batch.float())
            else:
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                loss = loss_fn(outputs, y_batch.long())
            
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            all_preds.append(outputs.cpu())
            all_targets.append(y_batch.cpu())
        
        avg_loss = total_loss / max(num_samples, 1)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if self.config.task == TaskType.RUL:
            # Denormalize for metric reporting in original scale
            metrics = compute_rul_metrics(all_preds, all_targets, rul_scale=self._rul_scale)
        else:
            metrics = compute_classification_metrics(
                all_preds, all_targets, self.config.num_classes
            )
        
        metrics["loss"] = avg_loss
        return avg_loss, metrics
    
    def train_local(
        self,
        global_state: Dict,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Perform local training and return update.
        
        This is the main entry point for federated training. It:
        1. Loads and preprocesses local data
        2. Creates train/val split
        3. Trains local model starting from global state
        4. Returns model delta and comprehensive metrics
        
        Args:
            global_state: Dictionary containing:
                - "model": The global model (nn.Module)
                - Optionally "round": Current FL round number
            config: Optional config overrides (merged with self.config)
            
        Returns:
            Dictionary containing:
                - "client_id": This client's ID
                - "num_samples": Number of training samples
                - "num_val_samples": Number of validation samples
                - "delta": State dict of model updates (local - global)
                - "train_metrics": Final training metrics
                - "val_metrics": Final validation metrics
                - "epochs_completed": Number of epochs actually trained
                - "status": "success", "dropout", or "error"
                - "training_time_seconds": Wall-clock training time
        """
        start_time = time.time()
        
        # Check for simulated dropout
        if self._should_dropout():
            return {
                "client_id": self.client_id,
                "status": "dropout",
                "num_samples": 0,
                "num_val_samples": 0,
                "delta": {},
                "train_metrics": {},
                "val_metrics": {},
                "epochs_completed": 0,
                "training_time_seconds": 0.0,
            }
        
        # Simulate network delay
        self._simulate_delay()
        
        # Apply config overrides
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Get model from global state
        model = global_state.get("model")
        if model is None or not isinstance(model, nn.Module):
            raise ValueError("global_state must include a torch.nn.Module under 'model'")
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            seed = self.config.seed + self.client_id
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Load and prepare data
            self.load_data()
            
            if self.num_train_samples == 0:
                raise ValueError("No training samples after split")
            
            # Create local model copy
            device = self.config.device
            local_model = copy.deepcopy(model).to(device)
            
            # Get loss function
            loss_fn = get_loss_function(self.config.task)
            
            # Create optimizer and scheduler
            optimizer = self._create_optimizer(local_model)
            scheduler = self._create_scheduler(optimizer)
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders()
            
            # Training loop with early stopping
            best_val_loss = float("inf")
            patience_counter = 0
            effective_epochs = self._get_effective_epochs()
            epochs_completed = 0
            
            for epoch in range(effective_epochs):
                # Train one epoch
                train_loss, train_metrics = self._train_one_epoch(
                    local_model, train_loader, optimizer, loss_fn, device
                )
                
                # Validate
                val_loss, val_metrics = self._evaluate(
                    local_model, val_loader, loss_fn, device
                )
                
                epochs_completed += 1
                
                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Early stopping check (on validation loss)
                if self.config.early_stopping_enabled:
                    if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            logger.debug(
                                f"Client {self.client_id}: Early stopping at epoch {epoch + 1}"
                            )
                            break
            
            # Store final metrics
            self._last_train_metrics = train_metrics
            self._last_val_metrics = val_metrics
            
            # Compute delta (local - global)
            delta = {}
            global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            local_sd = {k: v.cpu() for k, v in local_model.state_dict().items()}
            
            for key in global_sd.keys():
                delta[key] = (local_sd[key] - global_sd[key]).detach().clone()
            
            training_time = time.time() - start_time
            
            return {
                "client_id": self.client_id,
                "status": "success",
                "num_samples": self.num_train_samples,
                "num_val_samples": self.num_val_samples,
                "delta": delta,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "epochs_completed": epochs_completed,
                "training_time_seconds": training_time,
            }
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            return {
                "client_id": self.client_id,
                "status": "error",
                "error_message": str(e),
                "num_samples": 0,
                "num_val_samples": 0,
                "delta": {},
                "train_metrics": {},
                "val_metrics": {},
                "epochs_completed": 0,
                "training_time_seconds": time.time() - start_time,
            }


# ---------------------- Backward Compatibility ----------------------


class Client(SimulatedClientTrainer):
    """Backward-compatible alias for SimulatedClientTrainer.
    
    .. deprecated::
        Use `SimulatedClientTrainer` instead. This alias exists for
        backward compatibility with existing code.
    """
    
    def __init__(
        self,
        client_id: int,
        data: Optional[Union[str, pd.DataFrame, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """Initialize client with minimal configuration.
        
        This constructor maintains backward compatibility with the original
        Client class API.
        """
        # Default to RUL task for backward compatibility
        config = ClientConfig(
            task=TaskType.RUL,
            val_split=0.2,
            early_stopping_enabled=False,  # Original didn't have proper early stopping
            normalize_per_channel=False,  # Original didn't normalize
        )
        super().__init__(client_id=client_id, data=data, config=config)
    
    def train_local(self, global_state: dict, config: dict) -> dict:
        """Train locally with backward-compatible interface.
        
        Args:
            global_state: Must contain "model" key with nn.Module
            config: Training config dictionary (mapped to ClientConfig)
            
        Returns:
            Dictionary with client_id, num_samples, delta, metrics, epochs_ran
        """
        # Map old config keys to new config
        config_mapping = {
            "batch_size": "batch_size",
            "local_epochs": "local_epochs",
            "lr": "lr",
            "device": "device",
            "seed": "seed",
        }
        
        mapped_config = {}
        for old_key, new_key in config_mapping.items():
            if old_key in config:
                mapped_config[new_key] = config[old_key]
        
        # Handle early stopping
        early_cfg = config.get("early_stopping", {})
        if early_cfg.get("enabled", False):
            mapped_config["early_stopping_enabled"] = True
            mapped_config["early_stopping_patience"] = early_cfg.get("patience", 3)
            mapped_config["early_stopping_min_delta"] = early_cfg.get("min_delta", 1e-4)
        
        # Call parent implementation
        result = SimulatedClientTrainer.train_local(self, global_state, mapped_config)
        
        # Map back to old response format for compatibility
        return {
            "client_id": result["client_id"],
            "num_samples": result["num_samples"],
            "delta": result["delta"],
            "metrics": result.get("train_metrics", {"loss": 0.0}),
            "epochs_ran": result["epochs_completed"],
        }
