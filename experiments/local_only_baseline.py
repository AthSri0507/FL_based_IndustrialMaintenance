"""Local-Only Training Baseline — Independent Client Models Without Federation.

This module provides a local-only training baseline where each client trains
its own independent model without any federated aggregation. This serves as
a key comparison point for federated learning experiments.

The local-only baseline answers: "How well can clients do with just their own data?"

Usage:
    python -m experiments.local_only_baseline --config configs/local_only.yaml
    
    # Or with command-line args:
    python -m experiments.local_only_baseline \
        --num-clients 5 \
        --task rul \
        --local-epochs 20 \
        --heterogeneity uniform

Comparison with other baselines:
- Centralized: All data combined (upper bound on performance)
- Local-only: Each client's own data only (this baseline)
- Federated: Collaborative learning with aggregation

Key metrics reported:
- Per-client local performance (on own test data)
- Per-client global performance (on held-out global test set)
- Average and std across clients
- Comparison to centralized baseline (if available)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tcn import RULPredictor, FaultClassifier
from src.clients.client import (
    SimulatedClientTrainer,
    ClientConfig,
    TaskType,
    compute_rul_metrics,
    compute_classification_metrics,
)
from src.data.segment import segment_windows


logger = logging.getLogger(__name__)


# --------------------- Configuration ---------------------


@dataclass
class LocalOnlyConfig:
    """Configuration for local-only training baseline."""

    # Data settings
    data_dir: str = "data/raw"
    data_files: List[str] = field(default_factory=list)
    window_size: int = 50
    hop_size: int = 10
    normalize_windows: bool = True
    global_test_split: float = 0.15  # Held-out global test set

    # Client settings
    num_clients: int = 5
    heterogeneity_mode: str = "uniform"  # "uniform", "dirichlet", "extreme"
    dirichlet_alpha: float = 0.5  # For dirichlet heterogeneity
    extreme_imbalance: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.3, 0.1])

    # Task settings
    task: str = "rul"  # "rul" or "classification"
    num_classes: int = 2

    # Model architecture (same as centralized for fair comparison)
    num_layers: int = 4
    hidden_dim: int = 64
    kernel_size: int = 3
    dropout: float = 0.2
    fc_hidden: int = 32

    # Training settings (per client)
    local_epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    
    # RUL normalization
    normalize_rul: bool = True
    rul_max: Optional[float] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Output
    output_dir: str = "experiments/outputs/local_only"
    save_client_models: bool = True
    compare_centralized: bool = True
    centralized_checkpoint: str = ""  # Path to centralized model for comparison

    # Hardware
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: Optional[str] = None) -> LocalOnlyConfig:
    """Load configuration from YAML file or return defaults."""
    config = LocalOnlyConfig()

    if config_path and os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            for key, value in cfg_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except ImportError:
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    cfg_dict = json.load(f)
                for key, value in cfg_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

    return config


# --------------------- Data Generation ---------------------


def create_synthetic_data(
    num_samples: int = 1000,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic time-series data for testing.

    Args:
        num_samples: Number of samples to generate
        seq_length: Length of each time series
        num_channels: Number of sensor channels
        task: "rul" or "classification"
        num_classes: Number of classes for classification
        seed: Random seed

    Returns:
        X: Features of shape (num_samples, seq_length, num_channels)
        y: Labels of shape (num_samples,)
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((num_samples, seq_length, num_channels), dtype=np.float32)

    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, seq_length)
        for c in range(num_channels):
            freq = 0.5 + c * 0.2
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = 1.0 + rng.uniform(-0.3, 0.3)
            noise = rng.randn(seq_length) * 0.1

            # Degradation trend for later samples
            degradation = 0.0
            if i > num_samples // 2:
                degradation = np.linspace(0, rng.uniform(0.5, 2.0), seq_length)

            X[i, :, c] = amplitude * np.sin(freq * t + phase) + noise + degradation

    if task == "rul":
        y = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            signal_var = np.var(X[i])
            trend = np.mean(X[i, -10:]) - np.mean(X[i, :10])
            y[i] = max(0, 100 - 20 * signal_var - 10 * abs(trend) + rng.uniform(-5, 5))
    else:
        y = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            signal_var = np.var(X[i])
            y[i] = min(num_classes - 1, int(signal_var * num_classes))

    return X, y


def load_real_data(
    data_dir: str,
    file_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load real data from directory.

    Args:
        data_dir: Path to data directory
        file_patterns: Optional list of specific files

    Returns:
        X, y arrays
    """
    import pandas as pd

    data_path = Path(data_dir)
    all_X = []
    all_y = []

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

            if "label" in df.columns:
                y_col = "label"
            elif "rul" in df.columns:
                y_col = "rul"
            elif "RUL" in df.columns:
                y_col = "RUL"
            else:
                y_col = df.columns[-1]

            exclude_cols = {"index", "timestamp", "client_id", "unit_id", y_col}
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            X = df[feature_cols].values.astype(np.float32)
            y = df[y_col].values.astype(np.float32)

            all_X.append(X)
            all_y.append(y)

        except Exception as e:
            logger.warning(f"Could not load {fpath}: {e}")
            continue

    if not all_X:
        raise ValueError("No data could be loaded")

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# --------------------- Data Partitioning ---------------------


def partition_data_for_clients(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    heterogeneity_mode: str = "uniform",
    dirichlet_alpha: float = 0.5,
    extreme_imbalance: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data across clients with configurable heterogeneity.

    Args:
        X: Features array (N, W, C)
        y: Labels array (N,)
        num_clients: Number of clients
        heterogeneity_mode: "uniform", "dirichlet", or "extreme"
        dirichlet_alpha: Alpha parameter for Dirichlet distribution
        extreme_imbalance: List of fault prevalences for extreme mode
        seed: Random seed

    Returns:
        List of (X_client, y_client) tuples
    """
    rng = np.random.RandomState(seed)
    n_samples = len(X)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    if heterogeneity_mode == "uniform":
        # Equal random split
        splits = np.array_split(indices, num_clients)
        return [(X[s], y[s]) for s in splits]

    elif heterogeneity_mode == "dirichlet":
        # Dirichlet-based non-IID split (label-aware)
        if len(np.unique(y)) <= 10:  # Classification or discrete RUL bins
            labels = y.astype(int) if y.dtype != np.int64 else y
            unique_labels = np.unique(labels)
            
            # Sample proportions for each client from Dirichlet
            label_distribution = rng.dirichlet([dirichlet_alpha] * num_clients, len(unique_labels))
            
            client_indices = [[] for _ in range(num_clients)]
            for label_idx, label in enumerate(unique_labels):
                label_mask = labels == label
                label_indices = indices[label_mask[indices]]
                
                # Distribute this label's samples according to Dirichlet
                n_label = len(label_indices)
                proportions = label_distribution[label_idx]
                split_sizes = (proportions * n_label).astype(int)
                split_sizes[-1] = n_label - split_sizes[:-1].sum()  # Ensure all samples used
                
                rng.shuffle(label_indices)
                pos = 0
                for client_id, size in enumerate(split_sizes):
                    client_indices[client_id].extend(label_indices[pos:pos + size])
                    pos += size
            
            return [(X[np.array(ci)], y[np.array(ci)]) for ci in client_indices]
        else:
            # Continuous RUL: partition by RUL value ranges
            sorted_idx = indices[np.argsort(y[indices])]
            # Create uneven partitions based on Dirichlet
            proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
            split_sizes = (proportions * n_samples).astype(int)
            split_sizes[-1] = n_samples - split_sizes[:-1].sum()
            
            splits = []
            pos = 0
            for size in split_sizes:
                splits.append(sorted_idx[pos:pos + size])
                pos += size
            
            return [(X[s], y[s]) for s in splits]

    elif heterogeneity_mode == "extreme":
        # Extreme imbalance with specified fault prevalences
        if extreme_imbalance is None:
            extreme_imbalance = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Extend or truncate to match num_clients
        if len(extreme_imbalance) < num_clients:
            extreme_imbalance = extreme_imbalance + [0.5] * (num_clients - len(extreme_imbalance))
        extreme_imbalance = extreme_imbalance[:num_clients]
        
        # For classification, partition by label distribution
        if len(np.unique(y)) <= 10:
            labels = y.astype(int) if y.dtype != np.int64 else y
            pos_idx = indices[labels[indices] == 1]
            neg_idx = indices[labels[indices] == 0]
            
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            
            client_data = []
            base_size = n_samples // num_clients
            pos_ptr = 0
            neg_ptr = 0
            
            for i, target_prev in enumerate(extreme_imbalance):
                n_pos = min(int(base_size * target_prev), len(pos_idx) - pos_ptr)
                n_neg = min(base_size - n_pos, len(neg_idx) - neg_ptr)
                
                client_idx = np.concatenate([
                    pos_idx[pos_ptr:pos_ptr + n_pos],
                    neg_idx[neg_ptr:neg_ptr + n_neg]
                ])
                rng.shuffle(client_idx)
                
                client_data.append((X[client_idx], y[client_idx]))
                pos_ptr += n_pos
                neg_ptr += n_neg
            
            return client_data
        else:
            # For RUL: partition by RUL value ranges (some clients see only high/low RUL)
            sorted_idx = indices[np.argsort(y[indices])]
            splits = np.array_split(sorted_idx, num_clients)
            return [(X[s], y[s]) for s in splits]

    else:
        raise ValueError(f"Unknown heterogeneity mode: {heterogeneity_mode}")


# --------------------- Model Creation ---------------------


def create_model(
    task: str,
    num_channels: int,
    config: LocalOnlyConfig,
) -> nn.Module:
    """Create a fresh model instance.

    Args:
        task: "rul" or "classification"
        num_channels: Number of input channels (sensor features)
        config: Configuration

    Returns:
        Model instance
    """
    if task == "rul":
        return RULPredictor(
            num_channels=num_channels,
            num_layers=config.num_layers,
            hidden=config.hidden_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            fc_hidden=config.fc_hidden,
        )
    else:
        return FaultClassifier(
            num_channels=num_channels,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            hidden=config.hidden_dim,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            fc_hidden=config.fc_hidden,
        )


# --------------------- Evaluation ---------------------


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    task: str,
    batch_size: int = 32,
    device: str = "cpu",
    rul_scale: float = 1.0,
) -> Dict[str, float]:
    """Evaluate a model on given data.

    Args:
        model: Trained model
        X: Features tensor (N, W, C)
        y: Labels tensor (N,)
        task: "rul" or "classification"
        batch_size: Batch size for evaluation
        device: Device to use
        rul_scale: Scale factor for RUL denormalization

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0

    loss_fn = nn.MSELoss() if task == "rul" else nn.CrossEntropyLoss()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        if task == "rul":
            outputs = outputs.squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            loss = loss_fn(outputs, y_batch.float())
        else:
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            loss = loss_fn(outputs, y_batch.long())

        batch_size_actual = X_batch.size(0)
        total_loss += loss.item() * batch_size_actual
        num_samples += batch_size_actual

        all_preds.append(outputs.cpu())
        all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    avg_loss = total_loss / max(num_samples, 1)

    if task == "rul":
        metrics = compute_rul_metrics(all_preds, all_targets, rul_scale=rul_scale)
    else:
        metrics = compute_classification_metrics(all_preds, all_targets, num_classes=2)

    metrics["loss"] = avg_loss
    return metrics


# --------------------- Client Training ---------------------


def train_local_client(
    client_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: LocalOnlyConfig,
    num_channels: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a single client's local model.

    Args:
        client_id: Client identifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration
        num_channels: Number of input channels (sensor features)

    Returns:
        Trained model and training history
    """
    logger.info(f"Training client {client_id} with {len(X_train)} samples...")

    # Create fresh model for this client
    model = create_model(config.task, num_channels, config)

    # Prepare data for SimulatedClientTrainer
    # Combine train and val for the trainer (it will do its own split)
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    # Create unit IDs to enable proper splitting
    n_train = len(X_train)
    n_val = len(y_val)
    unit_ids = np.concatenate([
        np.zeros(n_train, dtype=np.int64),  # Training units
        np.ones(n_val, dtype=np.int64),  # Validation units
    ])

    # Configure the client trainer
    task_type = TaskType.RUL if config.task == "rul" else TaskType.CLASSIFICATION
    client_config = ClientConfig(
        task=task_type,
        num_classes=config.num_classes,
        val_split=n_val / (n_train + n_val),  # Match the actual split
        batch_size=config.batch_size,
        local_epochs=config.local_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        early_stopping_enabled=True,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        normalize_rul=config.normalize_rul,
        rul_max=config.rul_max,
        seed=config.seed + client_id if config.seed else None,
        device=config.device,
    )

    # Create and train the client
    trainer = SimulatedClientTrainer(
        client_id=client_id,
        data=(
            torch.tensor(X_combined, dtype=torch.float32),
            torch.tensor(y_combined, dtype=torch.float32 if config.task == "rul" else torch.long),
            torch.tensor(unit_ids, dtype=torch.long),
        ),
        config=client_config,
    )

    # Get initial model state
    global_state = {"model": model}

    # Train locally
    start_time = time.time()
    result = trainer.train_local(global_state)
    train_time = time.time() - start_time

    # Apply the learned delta to get final model
    if result["status"] == "success":
        trained_model = model
        with torch.no_grad():
            for name, param in trained_model.named_parameters():
                if name in result["delta"]:
                    delta = result["delta"][name]
                    if isinstance(delta, np.ndarray):
                        delta = torch.from_numpy(delta)
                    param.add_(delta.to(param.device))
    else:
        trained_model = model
        logger.warning(f"Client {client_id} training failed: {result.get('reason', 'unknown')}")

    history = {
        "client_id": client_id,
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "epochs_completed": result.get("epochs_completed", 0),
        "train_time": train_time,
        "final_train_metrics": result.get("train_metrics", {}),
        "final_val_metrics": result.get("val_metrics", {}),
        "rul_scale": trainer.rul_scale,
        "status": result["status"],
    }

    return trained_model, history


# --------------------- Main Pipeline ---------------------


class LocalOnlyBaseline:
    """Local-only training baseline experiment runner."""

    def __init__(self, config: LocalOnlyConfig):
        self.config = config
        self.client_models: List[nn.Module] = []
        self.client_histories: List[Dict] = []
        self.global_test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.results: Dict[str, Any] = {}

    def run(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the complete local-only baseline experiment.

        Args:
            X: Optional pre-loaded features (N, W, C)
            y: Optional pre-loaded labels (N,)

        Returns:
            Dictionary with all results and metrics
        """
        self._setup_logging()
        self._set_seeds()

        logger.info("=" * 60)
        logger.info("Local-Only Baseline Experiment")
        logger.info("=" * 60)
        logger.info(f"Configuration: {asdict(self.config)}")

        # Load or generate data
        if X is None or y is None:
            X, y = self._load_data()

        # Ensure data is 3D (N, W, C)
        if X.ndim == 2:
            X = segment_windows(
                X,
                W=self.config.window_size,
                H=self.config.hop_size,
                normalize=self.config.normalize_windows,
            )
            # Adjust y to match windows
            num_windows = X.shape[0]
            if len(y) > num_windows:
                indices = list(range(
                    self.config.window_size - 1,
                    len(y),
                    self.config.hop_size
                ))[:num_windows]
                y = y[indices]

        logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")

        # Hold out global test set
        X_train_full, y_train_full, X_test_global, y_test_global = self._split_global_test(X, y)
        
        # Convert global test to tensors
        self.global_test_data = (
            torch.tensor(X_test_global, dtype=torch.float32),
            torch.tensor(
                y_test_global,
                dtype=torch.float32 if self.config.task == "rul" else torch.long
            ),
        )
        
        logger.info(f"Training data: {len(X_train_full)} samples")
        logger.info(f"Global test data: {len(X_test_global)} samples")

        # Partition data across clients
        client_partitions = partition_data_for_clients(
            X_train_full,
            y_train_full,
            num_clients=self.config.num_clients,
            heterogeneity_mode=self.config.heterogeneity_mode,
            dirichlet_alpha=self.config.dirichlet_alpha,
            extreme_imbalance=self.config.extreme_imbalance,
            seed=self.config.seed,
        )

        logger.info(f"\nClient data distribution:")
        for i, (X_c, y_c) in enumerate(client_partitions):
            if self.config.task == "classification":
                pos_frac = np.mean(y_c == 1) if len(y_c) > 0 else 0
                logger.info(f"  Client {i}: {len(X_c)} samples, {pos_frac:.1%} positive")
            else:
                logger.info(f"  Client {i}: {len(X_c)} samples, RUL range: [{y_c.min():.1f}, {y_c.max():.1f}]")

        # Train each client independently
        num_channels = X.shape[2]
        total_train_time = 0.0

        for client_id, (X_client, y_client) in enumerate(client_partitions):
            # Split client data into train/val
            n_client = len(X_client)
            n_val = max(1, int(n_client * 0.2))
            n_train = n_client - n_val

            indices = np.arange(n_client)
            np.random.RandomState(self.config.seed + client_id).shuffle(indices)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            X_train_c = X_client[train_idx]
            y_train_c = y_client[train_idx]
            X_val_c = X_client[val_idx]
            y_val_c = y_client[val_idx]

            # Train client
            model, history = train_local_client(
                client_id=client_id,
                X_train=X_train_c,
                y_train=y_train_c,
                X_val=X_val_c,
                y_val=y_val_c,
                config=self.config,
                num_channels=num_channels,
            )

            self.client_models.append(model)
            self.client_histories.append(history)
            total_train_time += history["train_time"]

        # Evaluate all clients
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)

        self.results = self._evaluate_all_clients(client_partitions)
        self.results["total_train_time"] = total_train_time
        self.results["config"] = asdict(self.config)

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _setup_logging(self):
        """Configure logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = os.path.join(self.config.output_dir, "local_only.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from directory or generate synthetic."""
        try:
            X, y = load_real_data(self.config.data_dir, self.config.data_files)
            logger.info(f"Loaded real data from {self.config.data_dir}")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load real data: {e}")
            logger.info("Generating synthetic data...")
            X, y = create_synthetic_data(
                num_samples=1000,
                seq_length=100,
                num_channels=14,
                task=self.config.task,
                num_classes=self.config.num_classes,
                seed=self.config.seed,
            )
        return X, y

    def _split_global_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split off global test set."""
        n_samples = len(X)
        n_test = max(1, int(n_samples * self.config.global_test_split))

        indices = np.arange(n_samples)
        np.random.RandomState(self.config.seed).shuffle(indices)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def _evaluate_all_clients(
        self,
        client_partitions: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Evaluate all client models.

        Returns:
            Dictionary with per-client and aggregate metrics
        """
        results = {
            "per_client": [],
            "local_metrics": {},
            "global_metrics": {},
        }

        local_metrics_list = []
        global_metrics_list = []

        for client_id, (model, history, (X_c, y_c)) in enumerate(
            zip(self.client_models, self.client_histories, client_partitions)
        ):
            rul_scale = history.get("rul_scale", 1.0)
            
            # Convert to tensors and normalize targets to match training
            X_local = torch.tensor(X_c, dtype=torch.float32)
            if self.config.task == "rul" and self.config.normalize_rul and rul_scale > 0:
                # Normalize targets to match training scale
                y_local = torch.tensor(y_c / rul_scale, dtype=torch.float32)
            else:
                y_local = torch.tensor(
                    y_c,
                    dtype=torch.float32 if self.config.task == "rul" else torch.long
                )

            # Evaluate on local data
            local_metrics = evaluate_model(
                model,
                X_local,
                y_local,
                self.config.task,
                batch_size=self.config.batch_size,
                device=self.config.device,
                rul_scale=rul_scale,
            )

            # Prepare global test data with same normalization
            if self.config.task == "rul" and self.config.normalize_rul and rul_scale > 0:
                y_global_normalized = self.global_test_data[1] / rul_scale
            else:
                y_global_normalized = self.global_test_data[1]
            
            # Evaluate on global test data
            global_metrics = evaluate_model(
                model,
                self.global_test_data[0],
                y_global_normalized,
                self.config.task,
                batch_size=self.config.batch_size,
                device=self.config.device,
                rul_scale=rul_scale,
            )

            client_result = {
                "client_id": client_id,
                "num_samples": len(X_c),
                "training": history,
                "local_eval": local_metrics,
                "global_eval": global_metrics,
            }
            results["per_client"].append(client_result)

            local_metrics_list.append(local_metrics)
            global_metrics_list.append(global_metrics)

            logger.info(f"\nClient {client_id}:")
            logger.info(f"  Local samples: {len(X_c)}")
            if self.config.task == "rul":
                logger.info(f"  Local  - MAE: {local_metrics['mae']:.2f}, RMSE: {local_metrics['rmse']:.2f}")
                logger.info(f"  Global - MAE: {global_metrics['mae']:.2f}, RMSE: {global_metrics['rmse']:.2f}")
            else:
                logger.info(f"  Local  - Accuracy: {local_metrics['accuracy']:.2%}, F1: {local_metrics['f1']:.3f}")
                logger.info(f"  Global - Accuracy: {global_metrics['accuracy']:.2%}, F1: {global_metrics['f1']:.3f}")

        # Compute aggregate statistics
        results["local_metrics"] = self._aggregate_metrics(local_metrics_list)
        results["global_metrics"] = self._aggregate_metrics(global_metrics_list)

        return results

    def _aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean and std of metrics across clients."""
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        aggregated = {}

        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return aggregated

    def _save_results(self):
        """Save results and models."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save results JSON
        results_path = os.path.join(self.config.output_dir, "results.json")
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(self.results), f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Save client models
        if self.config.save_client_models:
            for client_id, model in enumerate(self.client_models):
                model_path = os.path.join(
                    self.config.output_dir,
                    f"client_{client_id}_model.pt"
                )
                torch.save(model.state_dict(), model_path)
            logger.info(f"Client models saved to {self.config.output_dir}")

    def _print_summary(self):
        """Print summary of results."""
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY: Local-Only Baseline Results")
        logger.info("=" * 60)

        if self.config.task == "rul":
            local_mae = self.results["local_metrics"].get("mae", {})
            global_mae = self.results["global_metrics"].get("mae", {})
            local_rmse = self.results["local_metrics"].get("rmse", {})
            global_rmse = self.results["global_metrics"].get("rmse", {})

            logger.info("\nLocal Performance (on each client's own data):")
            logger.info(f"  MAE:  {local_mae.get('mean', 0):.2f} ± {local_mae.get('std', 0):.2f}")
            logger.info(f"  RMSE: {local_rmse.get('mean', 0):.2f} ± {local_rmse.get('std', 0):.2f}")

            logger.info("\nGlobal Performance (on held-out test set):")
            logger.info(f"  MAE:  {global_mae.get('mean', 0):.2f} ± {global_mae.get('std', 0):.2f}")
            logger.info(f"  RMSE: {global_rmse.get('mean', 0):.2f} ± {global_rmse.get('std', 0):.2f}")

        else:
            local_acc = self.results["local_metrics"].get("accuracy", {})
            global_acc = self.results["global_metrics"].get("accuracy", {})
            local_f1 = self.results["local_metrics"].get("f1", {})
            global_f1 = self.results["global_metrics"].get("f1", {})

            logger.info("\nLocal Performance (on each client's own data):")
            logger.info(f"  Accuracy: {local_acc.get('mean', 0):.2%} ± {local_acc.get('std', 0):.2%}")
            logger.info(f"  F1:       {local_f1.get('mean', 0):.3f} ± {local_f1.get('std', 0):.3f}")

            logger.info("\nGlobal Performance (on held-out test set):")
            logger.info(f"  Accuracy: {global_acc.get('mean', 0):.2%} ± {global_acc.get('std', 0):.2%}")
            logger.info(f"  F1:       {global_f1.get('mean', 0):.3f} ± {global_f1.get('std', 0):.3f}")

        logger.info(f"\nTotal training time: {self.results.get('total_train_time', 0):.2f}s")
        logger.info(f"Number of clients: {self.config.num_clients}")
        logger.info(f"Heterogeneity: {self.config.heterogeneity_mode}")


# --------------------- CLI ---------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local-Only Training Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file",
    )

    # Data arguments
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--window-size", type=int, help="Sliding window size")
    parser.add_argument("--hop-size", type=int, help="Hop size between windows")

    # Client arguments
    parser.add_argument("--num-clients", type=int, help="Number of clients")
    parser.add_argument(
        "--heterogeneity",
        type=str,
        choices=["uniform", "dirichlet", "extreme"],
        help="Heterogeneity mode",
    )
    parser.add_argument("--dirichlet-alpha", type=float, help="Dirichlet alpha parameter")

    # Task arguments
    parser.add_argument("--task", type=str, choices=["rul", "classification"], help="Task type")
    parser.add_argument("--num-classes", type=int, help="Number of classes")

    # Training arguments
    parser.add_argument("--local-epochs", type=int, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")

    # Output arguments
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config from file
    config = load_config(args.config)

    # Override with CLI args
    cli_overrides = {
        "data_dir": args.data_dir,
        "window_size": args.window_size,
        "hop_size": args.hop_size,
        "num_clients": args.num_clients,
        "heterogeneity_mode": args.heterogeneity,
        "dirichlet_alpha": args.dirichlet_alpha,
        "task": args.task,
        "num_classes": args.num_classes,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(config, key, value)

    # Run experiment
    experiment = LocalOnlyBaseline(config)
    results = experiment.run()

    return results


if __name__ == "__main__":
    main()
