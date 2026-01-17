"""Federated Experiments Matrix — Comprehensive FL Experimentation Framework.

This module provides a systematic framework for running federated learning
experiments across multiple dimensions:

1. **Participation Fraction**: What fraction of clients participate each round?
2. **Data Heterogeneity**: How non-IID is the data distribution across clients?
3. **Local Epochs**: How many local training epochs per client per round?
4. **Communication Rounds**: Total number of federated rounds
5. **Privacy/Compression**: Optional differential privacy and gradient compression

Usage:
    python -m experiments.federated_matrix --config configs/federated_matrix.yaml
    
    # Or with command-line args:
    python -m experiments.federated_matrix \
        --num-clients 10 \
        --participation-fractions 0.3 0.5 1.0 \
        --heterogeneity-levels uniform dirichlet \
        --local-epochs 1 5 10 \
        --num-rounds 50

Output:
- Per-experiment metrics (train/val loss, MAE, RMSE, etc.)
- Aggregated comparison across experiment grid
- Learning curves per configuration
- Final results table with best configurations

This module enables systematic comparison of FL configurations to answer:
- How does participation rate affect convergence?
- How does data heterogeneity impact global model quality?
- What is the optimal local epochs setting?
- How do FL results compare to centralized/local-only baselines?
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

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
from src.server.aggregator import FedAvgAggregator, sample_clients
from src.server.orchestrator import (
    FLOrchestrator,
    RoundLogger,
    CheckpointManager,
    RoundMetrics,
)
from src.data.segment import segment_windows


logger = logging.getLogger(__name__)


# --------------------- Configuration ---------------------


@dataclass
class FederatedExperimentConfig:
    """Configuration for a single federated experiment."""
    
    # Experiment identification
    experiment_id: str = ""
    experiment_name: str = "federated_experiment"
    
    # Data profile: "clean" for existing behavior, "non_iid_hard" for stress testing
    # NOTE: Changing data_profile ONLY affects data generation.
    # It does NOT alter: training loops, model architecture, aggregation,
    # metrics, logging, or random seed handling.
    data_profile: str = "clean"
    
    # Data settings
    data_dir: str = "data/raw"
    data_files: List[str] = field(default_factory=list)
    window_size: int = 50
    hop_size: int = 10
    normalize_windows: bool = True
    global_test_split: float = 0.15
    
    # Client settings
    num_clients: int = 10
    heterogeneity_mode: str = "uniform"  # "uniform", "dirichlet", "extreme"
    dirichlet_alpha: float = 0.5
    extreme_imbalance: List[float] = field(default_factory=list)
    
    # Task settings
    task: str = "rul"
    num_classes: int = 2
    
    # Model architecture
    num_layers: int = 4
    hidden_dim: int = 64
    kernel_size: int = 3
    dropout: float = 0.2
    fc_hidden: int = 32
    
    # Federated learning settings
    num_rounds: int = 50
    participation_fraction: float = 0.5  # Fraction of clients per round
    local_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    
    # Client training settings
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    normalize_rul: bool = True
    rul_max: Optional[float] = None
    
    # Privacy settings (optional)
    enable_dp: bool = False  # Differential privacy
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    
    # Compression settings (optional)
    enable_compression: bool = False
    compression_ratio: float = 0.1  # Keep top 10% of gradients
    
    # Evaluation settings
    eval_every: int = 5  # Evaluate global model every N rounds
    checkpoint_every: int = 10
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Output
    output_dir: str = "experiments/outputs/federated"
    save_checkpoints: bool = True
    
    # Hardware
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.experiment_id:
            self.experiment_id = f"{self.heterogeneity_mode}_pf{self.participation_fraction}_le{self.local_epochs}"


@dataclass 
class ExperimentMatrixConfig:
    """Configuration for the full experiment matrix."""
    
    # Base configuration (applied to all experiments)
    base_config: FederatedExperimentConfig = field(default_factory=FederatedExperimentConfig)
    
    # Matrix dimensions to vary
    participation_fractions: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])
    heterogeneity_modes: List[str] = field(default_factory=lambda: ["uniform", "dirichlet"])
    local_epochs_list: List[int] = field(default_factory=lambda: [1, 5, 10])
    dirichlet_alphas: List[float] = field(default_factory=lambda: [0.5])  # Only used if dirichlet in modes
    
    # Global settings
    output_dir: str = "experiments/outputs/federated_matrix"
    parallel_experiments: bool = False  # Future: run experiments in parallel
    
    def generate_experiment_configs(self) -> List[FederatedExperimentConfig]:
        """Generate all experiment configurations from the matrix."""
        configs = []
        
        for pf, het_mode, le in itertools.product(
            self.participation_fractions,
            self.heterogeneity_modes,
            self.local_epochs_list,
        ):
            # For dirichlet mode, also vary alpha
            alphas = self.dirichlet_alphas if het_mode == "dirichlet" else [0.5]
            
            for alpha in alphas:
                config = FederatedExperimentConfig(
                    # Copy base config
                    experiment_name=self.base_config.experiment_name,
                    data_dir=self.base_config.data_dir,
                    data_files=self.base_config.data_files,
                    window_size=self.base_config.window_size,
                    hop_size=self.base_config.hop_size,
                    normalize_windows=self.base_config.normalize_windows,
                    global_test_split=self.base_config.global_test_split,
                    num_clients=self.base_config.num_clients,
                    task=self.base_config.task,
                    num_classes=self.base_config.num_classes,
                    num_layers=self.base_config.num_layers,
                    hidden_dim=self.base_config.hidden_dim,
                    kernel_size=self.base_config.kernel_size,
                    dropout=self.base_config.dropout,
                    fc_hidden=self.base_config.fc_hidden,
                    num_rounds=self.base_config.num_rounds,
                    batch_size=self.base_config.batch_size,
                    lr=self.base_config.lr,
                    weight_decay=self.base_config.weight_decay,
                    optimizer=self.base_config.optimizer,
                    early_stopping_enabled=self.base_config.early_stopping_enabled,
                    early_stopping_patience=self.base_config.early_stopping_patience,
                    normalize_rul=self.base_config.normalize_rul,
                    rul_max=self.base_config.rul_max,
                    enable_dp=self.base_config.enable_dp,
                    dp_epsilon=self.base_config.dp_epsilon,
                    dp_delta=self.base_config.dp_delta,
                    dp_max_grad_norm=self.base_config.dp_max_grad_norm,
                    enable_compression=self.base_config.enable_compression,
                    compression_ratio=self.base_config.compression_ratio,
                    eval_every=self.base_config.eval_every,
                    checkpoint_every=self.base_config.checkpoint_every,
                    seed=self.base_config.seed,
                    deterministic=self.base_config.deterministic,
                    save_checkpoints=self.base_config.save_checkpoints,
                    device=self.base_config.device,
                    # Varied parameters
                    participation_fraction=pf,
                    heterogeneity_mode=het_mode,
                    local_epochs=le,
                    dirichlet_alpha=alpha,
                    output_dir=os.path.join(
                        self.output_dir,
                        f"het_{het_mode}_pf_{pf}_le_{le}" + (f"_alpha_{alpha}" if het_mode == "dirichlet" else "")
                    ),
                )
                config.experiment_id = f"het_{het_mode}_pf_{pf}_le_{le}" + (f"_alpha_{alpha}" if het_mode == "dirichlet" else "")
                configs.append(config)
        
        return configs


def load_config(config_path: Optional[str] = None) -> ExperimentMatrixConfig:
    """Load configuration from YAML/JSON file."""
    config = ExperimentMatrixConfig()
    
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        except ImportError:
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
        
        # Load base config
        if "base_config" in cfg_dict:
            for key, value in cfg_dict["base_config"].items():
                if hasattr(config.base_config, key):
                    setattr(config.base_config, key, value)
        
        # Load matrix dimensions
        for key in ["participation_fractions", "heterogeneity_modes", "local_epochs_list", "dirichlet_alphas"]:
            if key in cfg_dict:
                setattr(config, key, cfg_dict[key])
        
        if "output_dir" in cfg_dict:
            config.output_dir = cfg_dict["output_dir"]
    
    return config


# --------------------- Data Utilities ---------------------


def create_synthetic_data(
    num_samples: int = 1000,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic time-series data for testing."""
    rng = np.random.RandomState(seed)
    
    X = np.zeros((num_samples, seq_length, num_channels), dtype=np.float32)
    
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, seq_length)
        for c in range(num_channels):
            freq = 0.5 + c * 0.2
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = 1.0 + rng.uniform(-0.3, 0.3)
            noise = rng.randn(seq_length) * 0.1
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


def partition_data_for_clients(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    heterogeneity_mode: str = "uniform",
    dirichlet_alpha: float = 0.5,
    extreme_imbalance: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data across clients with configurable heterogeneity."""
    rng = np.random.RandomState(seed)
    n_samples = len(X)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    if heterogeneity_mode == "uniform":
        splits = np.array_split(indices, num_clients)
        return [(X[s], y[s]) for s in splits]
    
    elif heterogeneity_mode == "dirichlet":
        # For continuous RUL, bin into groups first
        if len(np.unique(y)) > 10:
            # Bin RUL values into 10 groups
            y_binned = np.digitize(y, np.linspace(y.min(), y.max(), 11)[:-1]) - 1
        else:
            y_binned = y.astype(int)
        
        unique_labels = np.unique(y_binned)
        label_distribution = rng.dirichlet([dirichlet_alpha] * num_clients, len(unique_labels))
        
        client_indices = [[] for _ in range(num_clients)]
        for label_idx, label in enumerate(unique_labels):
            label_mask = y_binned == label
            label_indices = indices[label_mask[indices]]
            
            n_label = len(label_indices)
            proportions = label_distribution[label_idx]
            split_sizes = (proportions * n_label).astype(int)
            split_sizes[-1] = n_label - split_sizes[:-1].sum()
            
            rng.shuffle(label_indices)
            pos = 0
            for client_id, size in enumerate(split_sizes):
                client_indices[client_id].extend(label_indices[pos:pos + size])
                pos += size
        
        return [(X[np.array(ci)], y[np.array(ci)]) for ci in client_indices if len(ci) > 0]
    
    elif heterogeneity_mode == "extreme":
        if extreme_imbalance is None:
            extreme_imbalance = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Sort by target value and partition
        sorted_idx = indices[np.argsort(y[indices])]
        proportions = rng.dirichlet([0.5] * num_clients)
        split_sizes = (proportions * n_samples).astype(int)
        split_sizes[-1] = n_samples - split_sizes[:-1].sum()
        
        splits = []
        pos = 0
        for size in split_sizes:
            splits.append(sorted_idx[pos:pos + size])
            pos += size
        
        return [(X[s], y[s]) for s in splits if len(s) > 0]
    
    else:
        raise ValueError(f"Unknown heterogeneity mode: {heterogeneity_mode}")


# --------------------- Model Creation ---------------------


def create_model(
    task: str,
    num_channels: int,
    config: FederatedExperimentConfig,
) -> nn.Module:
    """Create a fresh model instance."""
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


# --------------------- Privacy & Compression ---------------------


def clip_gradients(delta: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
    """Clip gradient deltas to max norm for differential privacy."""
    total_norm = 0.0
    for tensor in delta.values():
        total_norm += tensor.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {k: v * scale for k, v in delta.items()}
    return delta


def add_dp_noise(
    delta: Dict[str, torch.Tensor],
    noise_multiplier: float,
    max_norm: float,
) -> Dict[str, torch.Tensor]:
    """Add Gaussian noise for differential privacy."""
    noisy_delta = {}
    for k, v in delta.items():
        noise = torch.randn_like(v) * noise_multiplier * max_norm
        noisy_delta[k] = v + noise
    return noisy_delta


def compress_delta(
    delta: Dict[str, torch.Tensor],
    compression_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Top-k sparsification for gradient compression."""
    compressed = {}
    for k, v in delta.items():
        flat = v.flatten()
        k_elements = max(1, int(len(flat) * compression_ratio))
        _, indices = torch.topk(flat.abs(), k_elements)
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0
        compressed[k] = (flat * mask).reshape(v.shape)
    return compressed


# --------------------- Evaluation ---------------------


@torch.no_grad()
def evaluate_global_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    batch_size: int = 32,
    device: str = "cpu",
    rul_scale: float = 1.0,
) -> Dict[str, float]:
    """Evaluate the global model on test data."""
    model.eval()
    model.to(device)
    
    dataset = TensorDataset(X_test, y_test)
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


# --------------------- Single Experiment Runner ---------------------


class FederatedExperiment:
    """Runs a single federated learning experiment."""
    
    def __init__(self, config: FederatedExperimentConfig):
        self.config = config
        self.client_trainers: Dict[str, SimulatedClientTrainer] = {}
        self.global_model: Optional[nn.Module] = None
        self.global_test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.round_history: List[Dict] = []
        self.rul_scale: float = 1.0
    
    def run(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the federated experiment.
        
        Returns:
            Dictionary with experiment results
        """
        self._setup()
        
        # =================================================================
        # NON-IID HARD MODE DATA DISPATCH
        # NOTE: This ONLY affects data generation/partitioning.
        # Training loops, model architecture, aggregation, metrics,
        # logging, and random seed handling remain COMPLETELY UNCHANGED.
        # =================================================================
        if self.config.data_profile == "non_iid_hard":
            from src.data.non_iid_generator import generate_non_iid_hard_data
            
            logger.info("Using NON-IID HARD data profile")
            logger.info("  - Label skew: client-specific RUL distributions")
            logger.info("  - Feature skew: client-specific noise/bias")
            logger.info("  - Quantity skew: imbalanced sample counts")
            
            # Generate pre-partitioned heterogeneous data
            client_partitions = generate_non_iid_hard_data(
                num_clients=self.config.num_clients,
                seq_length=100,  # Fixed for synthetic data
                num_channels=14,  # Fixed for synthetic data
                task=self.config.task,
                num_classes=self.config.num_classes,
                seed=self.config.seed,
                round_id=0,  # No concept drift during initial data generation
            )
            
            # Create merged dataset for global operations (test split, normalization)
            all_X = np.concatenate([X_c for X_c, y_c in client_partitions], axis=0)
            all_y = np.concatenate([y_c for X_c, y_c in client_partitions], axis=0)
            
            logger.info(f"Total data shape: {all_X.shape}")
            
            # Hold out global test set (from merged data)
            X_train, y_train, X_test, y_test = self._split_test(all_X, all_y)
            
            # Auto-detect RUL scale for normalization
            if self.config.task == "rul" and self.config.normalize_rul:
                if self.config.rul_max is not None:
                    self.rul_scale = float(self.config.rul_max)
                else:
                    self.rul_scale = float(y_train.max())
            
            # Normalize test targets for evaluation
            if self.config.task == "rul" and self.config.normalize_rul:
                y_test_normalized = y_test / self.rul_scale
            else:
                y_test_normalized = y_test
            
            self.global_test_data = (
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(
                    y_test_normalized,
                    dtype=torch.float32 if self.config.task == "rul" else torch.long
                ),
            )
            
            # Re-generate client partitions excluding test indices
            # (simplified: regenerate with same seed, proportionally reduce samples)
            test_fraction = self.config.global_test_split
            train_partitions = []
            for client_id, (X_c, y_c) in enumerate(client_partitions):
                n_train = int(len(X_c) * (1 - test_fraction))
                train_partitions.append((X_c[:n_train], y_c[:n_train]))
            
            client_partitions = train_partitions
            
            # Log data distribution
            logger.info(f"Partitioned data across {len(client_partitions)} clients (non_iid_hard)")
            for i, (X_c, y_c) in enumerate(client_partitions):
                if self.config.task == "rul":
                    logger.info(f"  Client {i}: {len(X_c)} samples, RUL: [{y_c.min():.1f}, {y_c.max():.1f}]")
                else:
                    logger.info(f"  Client {i}: {len(X_c)} samples")
            
            # Initialize client trainers
            self._init_clients(client_partitions)
            
            # Initialize global model
            num_channels = client_partitions[0][0].shape[2]
            self.global_model = create_model(self.config.task, num_channels, self.config)
            self.global_model.to(self.config.device)
            
            # Run federated training (UNCHANGED)
            results = self._run_federated_training()
            
            return results
        # =================================================================
        # END NON-IID HARD MODE DATA DISPATCH
        # =================================================================
        
        # =================================================================
        # CLEAN MODE (DEFAULT) - EXISTING BEHAVIOR UNCHANGED
        # =================================================================
        # Load or generate data
        if X is None or y is None:
            X, y = self._load_data()
        
        # Ensure 3D data
        if X.ndim == 2:
            X = segment_windows(
                X,
                W=self.config.window_size,
                H=self.config.hop_size,
                normalize=self.config.normalize_windows,
            )
            num_windows = X.shape[0]
            if len(y) > num_windows:
                indices = list(range(
                    self.config.window_size - 1,
                    len(y),
                    self.config.hop_size
                ))[:num_windows]
                y = y[indices]
        
        logger.info(f"Data shape: {X.shape}")
        
        # Hold out global test set
        X_train, y_train, X_test, y_test = self._split_test(X, y)
        
        # Auto-detect RUL scale for normalization
        if self.config.task == "rul" and self.config.normalize_rul:
            if self.config.rul_max is not None:
                self.rul_scale = float(self.config.rul_max)
            else:
                self.rul_scale = float(y_train.max())
        
        # Normalize test targets for evaluation
        if self.config.task == "rul" and self.config.normalize_rul:
            y_test_normalized = y_test / self.rul_scale
        else:
            y_test_normalized = y_test
        
        self.global_test_data = (
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(
                y_test_normalized,
                dtype=torch.float32 if self.config.task == "rul" else torch.long
            ),
        )
        
        # Partition data across clients
        client_partitions = partition_data_for_clients(
            X_train, y_train,
            num_clients=self.config.num_clients,
            heterogeneity_mode=self.config.heterogeneity_mode,
            dirichlet_alpha=self.config.dirichlet_alpha,
            extreme_imbalance=self.config.extreme_imbalance,
            seed=self.config.seed,
        )
        
        # Log data distribution
        logger.info(f"Partitioned data across {len(client_partitions)} clients")
        for i, (X_c, y_c) in enumerate(client_partitions):
            if self.config.task == "rul":
                logger.info(f"  Client {i}: {len(X_c)} samples, RUL: [{y_c.min():.1f}, {y_c.max():.1f}]")
            else:
                logger.info(f"  Client {i}: {len(X_c)} samples")
        
        # Initialize client trainers
        self._init_clients(client_partitions)
        
        # Initialize global model
        num_channels = X.shape[2]
        self.global_model = create_model(self.config.task, num_channels, self.config)
        self.global_model.to(self.config.device)
        
        # Run federated training
        results = self._run_federated_training()
        
        return results
    
    def _setup(self):
        """Setup logging and directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        log_file = os.path.join(self.config.output_dir, "experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load or generate data."""
        try:
            # Try loading real data
            import pandas as pd
            data_path = Path(self.config.data_dir)
            files = list(data_path.glob("*.parquet")) + list(data_path.glob("*.csv"))
            
            if not files:
                raise FileNotFoundError("No data files found")
            
            all_X, all_y = [], []
            for fpath in sorted(files):
                if str(fpath).endswith(".parquet"):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)
                
                # Determine target column
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
                
                all_X.append(df[feature_cols].values.astype(np.float32))
                all_y.append(df[y_col].values.astype(np.float32))
            
            return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)
            
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load real data: {e}")
            logger.info("Generating synthetic data...")
            return create_synthetic_data(
                num_samples=1000,
                seq_length=100,
                num_channels=14,
                task=self.config.task,
                num_classes=self.config.num_classes,
                seed=self.config.seed,
            )
    
    def _split_test(
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
    
    def _init_clients(self, client_partitions: List[Tuple[np.ndarray, np.ndarray]]):
        """Initialize client trainers."""
        task_type = TaskType.RUL if self.config.task == "rul" else TaskType.CLASSIFICATION
        
        for client_id, (X_c, y_c) in enumerate(client_partitions):
            client_config = ClientConfig(
                task=task_type,
                num_classes=self.config.num_classes,
                val_split=0.2,
                batch_size=self.config.batch_size,
                local_epochs=self.config.local_epochs,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                optimizer=self.config.optimizer,
                early_stopping_enabled=self.config.early_stopping_enabled,
                early_stopping_patience=self.config.early_stopping_patience,
                normalize_rul=self.config.normalize_rul,
                rul_max=self.rul_scale if self.config.normalize_rul else None,
                seed=self.config.seed + client_id if self.config.seed else None,
                device=self.config.device,
            )
            
            trainer = SimulatedClientTrainer(
                client_id=client_id,
                data=(
                    torch.tensor(X_c, dtype=torch.float32),
                    torch.tensor(
                        y_c,
                        dtype=torch.float32 if self.config.task == "rul" else torch.long
                    ),
                ),
                config=client_config,
            )
            
            self.client_trainers[str(client_id)] = trainer
    
    def _run_federated_training(self) -> Dict[str, Any]:
        """Execute federated training rounds."""
        logger.info("=" * 60)
        logger.info(f"Starting Federated Training: {self.config.experiment_id}")
        logger.info(f"  Rounds: {self.config.num_rounds}")
        logger.info(f"  Clients: {self.config.num_clients}")
        logger.info(f"  Participation: {self.config.participation_fraction:.0%}")
        logger.info(f"  Local epochs: {self.config.local_epochs}")
        logger.info(f"  Heterogeneity: {self.config.heterogeneity_mode}")
        logger.info("=" * 60)
        
        aggregator = FedAvgAggregator(round_id=0)
        client_ids = list(self.client_trainers.keys())
        
        best_loss = float("inf")
        best_round = 0
        start_time = time.time()
        
        for round_id in range(1, self.config.num_rounds + 1):
            round_start = time.time()
            
            # Reset aggregator for this round
            aggregator.reset(round_id)
            
            # Select clients for this round
            num_to_select = max(1, int(len(client_ids) * self.config.participation_fraction))
            selected_clients = sample_clients(
                client_ids,
                num_clients=num_to_select,
                seed=self.config.seed + round_id if self.config.seed else None,
            )
            
            # Get global model state
            global_state = {"model": self.global_model}
            
            # Collect updates from selected clients
            client_metrics = {}
            total_samples = 0
            
            for client_id in selected_clients:
                trainer = self.client_trainers[client_id]
                result = trainer.train_local(global_state)
                
                if result["status"] == "success":
                    delta = result["delta"]
                    
                    # Apply privacy/compression if enabled
                    if self.config.enable_dp:
                        delta = clip_gradients(delta, self.config.dp_max_grad_norm)
                        noise_mult = self.config.dp_max_grad_norm * np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon
                        delta = add_dp_noise(delta, noise_mult, self.config.dp_max_grad_norm)
                    
                    if self.config.enable_compression:
                        delta = compress_delta(delta, self.config.compression_ratio)
                    
                    aggregator.add_update(
                        client_id=client_id,
                        delta=delta,
                        num_samples=result["num_samples"],
                        round_id=round_id,
                    )
                    
                    client_metrics[client_id] = result.get("val_metrics", result.get("train_metrics", {}))
                    total_samples += result["num_samples"]
            
            # Aggregate and apply
            if aggregator.num_updates > 0:
                aggregator.aggregate_and_apply(self.global_model)
            
            round_time = time.time() - round_start
            
            # Compute average client metrics
            avg_metrics = self._average_client_metrics(client_metrics)
            
            # Evaluate global model periodically
            global_metrics = None
            if round_id % self.config.eval_every == 0 or round_id == self.config.num_rounds:
                global_metrics = evaluate_global_model(
                    self.global_model,
                    self.global_test_data[0],
                    self.global_test_data[1],
                    self.config.task,
                    batch_size=self.config.batch_size,
                    device=self.config.device,
                    rul_scale=self.rul_scale,
                )
                
                # Track best
                if global_metrics["loss"] < best_loss:
                    best_loss = global_metrics["loss"]
                    best_round = round_id
                    if self.config.save_checkpoints:
                        self._save_checkpoint(round_id, is_best=True)
            
            # Log round
            round_info = {
                "round_id": round_id,
                "num_participants": len(selected_clients),
                "total_samples": total_samples,
                "avg_client_metrics": avg_metrics,
                "global_metrics": global_metrics,
                "round_time": round_time,
            }
            self.round_history.append(round_info)
            
            # Console output
            if global_metrics:
                if self.config.task == "rul":
                    logger.info(
                        f"Round {round_id}/{self.config.num_rounds}: "
                        f"{len(selected_clients)} clients, "
                        f"global MAE={global_metrics['mae']:.2f}, "
                        f"RMSE={global_metrics['rmse']:.2f}"
                    )
                else:
                    logger.info(
                        f"Round {round_id}/{self.config.num_rounds}: "
                        f"{len(selected_clients)} clients, "
                        f"global Acc={global_metrics['accuracy']:.2%}"
                    )
            else:
                logger.info(
                    f"Round {round_id}/{self.config.num_rounds}: "
                    f"{len(selected_clients)} clients, {total_samples} samples"
                )
            
            # Checkpoint periodically
            if self.config.save_checkpoints and round_id % self.config.checkpoint_every == 0:
                self._save_checkpoint(round_id)
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = evaluate_global_model(
            self.global_model,
            self.global_test_data[0],
            self.global_test_data[1],
            self.config.task,
            batch_size=self.config.batch_size,
            device=self.config.device,
            rul_scale=self.rul_scale,
        )
        
        # Compile results
        results = {
            "experiment_id": self.config.experiment_id,
            "config": asdict(self.config),
            "final_metrics": final_metrics,
            "best_loss": best_loss,
            "best_round": best_round,
            "total_time": total_time,
            "round_history": self.round_history,
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("=" * 60)
        logger.info("Experiment Complete")
        logger.info(f"  Final metrics: {final_metrics}")
        logger.info(f"  Best loss: {best_loss:.4f} at round {best_round}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info("=" * 60)
        
        return results
    
    def _average_client_metrics(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Compute average metrics across clients."""
        if not client_metrics:
            return {}
        
        all_keys = set()
        for metrics in client_metrics.values():
            all_keys.update(metrics.keys())
        
        avg = {}
        for key in all_keys:
            values = [m[key] for m in client_metrics.values() if key in m]
            if values:
                avg[key] = float(np.mean(values))
        
        return avg
    
    def _save_checkpoint(self, round_id: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "round_id": round_id,
            "model_state_dict": self.global_model.state_dict(),
            "config": asdict(self.config),
        }
        
        path = os.path.join(checkpoint_dir, f"round_{round_id:04d}.pt")
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
    
    def _save_results(self, results: Dict):
        """Save experiment results."""
        # Convert numpy/torch types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results_path = os.path.join(self.config.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(convert(results), f, indent=2)


# --------------------- Experiment Matrix Runner ---------------------


class FederatedExperimentMatrix:
    """Runs a matrix of federated learning experiments."""
    
    def __init__(self, config: ExperimentMatrixConfig):
        self.config = config
        self.experiment_results: List[Dict] = []
    
    def run(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run all experiments in the matrix.
        
        Returns:
            Dictionary with all results and comparison
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.config.output_dir, "matrix.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        
        # Generate experiment configurations
        experiment_configs = self.config.generate_experiment_configs()
        
        logger.info("=" * 60)
        logger.info("Federated Experiment Matrix")
        logger.info("=" * 60)
        logger.info(f"Total experiments: {len(experiment_configs)}")
        logger.info(f"Participation fractions: {self.config.participation_fractions}")
        logger.info(f"Heterogeneity modes: {self.config.heterogeneity_modes}")
        logger.info(f"Local epochs: {self.config.local_epochs_list}")
        logger.info("=" * 60)
        
        # Load data once if not provided
        if X is None or y is None:
            logger.info("Loading data...")
            X, y = self._load_or_generate_data()
        
        # Run each experiment
        for i, exp_config in enumerate(experiment_configs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Experiment {i + 1}/{len(experiment_configs)}: {exp_config.experiment_id}")
            logger.info(f"{'=' * 60}")
            
            experiment = FederatedExperiment(exp_config)
            results = experiment.run(X=X.copy(), y=y.copy())
            
            self.experiment_results.append(results)
        
        # Compile comparison
        comparison = self._compile_comparison()
        
        # Save all results
        all_results = {
            "matrix_config": {
                "participation_fractions": self.config.participation_fractions,
                "heterogeneity_modes": self.config.heterogeneity_modes,
                "local_epochs_list": self.config.local_epochs_list,
            },
            "experiments": self.experiment_results,
            "comparison": comparison,
        }
        
        results_path = os.path.join(self.config.output_dir, "matrix_results.json")
        with open(results_path, "w") as f:
            json.dump(self._convert_for_json(all_results), f, indent=2)
        
        # Print summary
        self._print_summary(comparison)
        
        return all_results
    
    def _load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load or generate data for experiments."""
        try:
            import pandas as pd
            data_path = Path(self.config.base_config.data_dir)
            files = list(data_path.glob("*.parquet")) + list(data_path.glob("*.csv"))
            
            if not files:
                raise FileNotFoundError("No data files")
            
            all_X, all_y = [], []
            for fpath in sorted(files):
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
                
                exclude = {"index", "timestamp", "client_id", "unit_id", y_col}
                feature_cols = [c for c in df.columns if c not in exclude]
                
                all_X.append(df[feature_cols].values.astype(np.float32))
                all_y.append(df[y_col].values.astype(np.float32))
            
            return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)
            
        except Exception as e:
            logger.warning(f"Could not load data: {e}, generating synthetic")
            return create_synthetic_data(
                num_samples=1000,
                seq_length=100,
                num_channels=14,
                task=self.config.base_config.task,
                num_classes=self.config.base_config.num_classes,
                seed=self.config.base_config.seed,
            )
    
    def _compile_comparison(self) -> Dict[str, Any]:
        """Compile comparison across all experiments."""
        comparison = {
            "by_participation": {},
            "by_heterogeneity": {},
            "by_local_epochs": {},
            "best_config": None,
            "worst_config": None,
        }
        
        # Group by dimensions
        for result in self.experiment_results:
            config = result["config"]
            final = result["final_metrics"]
            
            pf = config["participation_fraction"]
            het = config["heterogeneity_mode"]
            le = config["local_epochs"]
            
            # By participation
            if pf not in comparison["by_participation"]:
                comparison["by_participation"][pf] = []
            comparison["by_participation"][pf].append(final)
            
            # By heterogeneity
            if het not in comparison["by_heterogeneity"]:
                comparison["by_heterogeneity"][het] = []
            comparison["by_heterogeneity"][het].append(final)
            
            # By local epochs
            if le not in comparison["by_local_epochs"]:
                comparison["by_local_epochs"][le] = []
            comparison["by_local_epochs"][le].append(final)
        
        # Compute averages
        for dim, groups in [
            ("by_participation", comparison["by_participation"]),
            ("by_heterogeneity", comparison["by_heterogeneity"]),
            ("by_local_epochs", comparison["by_local_epochs"]),
        ]:
            for key, metrics_list in groups.items():
                avg = {}
                for metric_key in metrics_list[0].keys():
                    values = [m[metric_key] for m in metrics_list]
                    avg[metric_key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                comparison[dim][key] = avg
        
        # Find best and worst
        best_loss = float("inf")
        worst_loss = float("-inf")
        
        for result in self.experiment_results:
            loss = result["final_metrics"]["loss"]
            if loss < best_loss:
                best_loss = loss
                comparison["best_config"] = {
                    "experiment_id": result["experiment_id"],
                    "final_metrics": result["final_metrics"],
                    "config": {
                        "participation_fraction": result["config"]["participation_fraction"],
                        "heterogeneity_mode": result["config"]["heterogeneity_mode"],
                        "local_epochs": result["config"]["local_epochs"],
                    }
                }
            if loss > worst_loss:
                worst_loss = loss
                comparison["worst_config"] = {
                    "experiment_id": result["experiment_id"],
                    "final_metrics": result["final_metrics"],
                    "config": {
                        "participation_fraction": result["config"]["participation_fraction"],
                        "heterogeneity_mode": result["config"]["heterogeneity_mode"],
                        "local_epochs": result["config"]["local_epochs"],
                    }
                }
        
        return comparison
    
    def _print_summary(self, comparison: Dict):
        """Print summary of experiment matrix results."""
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT MATRIX SUMMARY")
        logger.info("=" * 60)
        
        # Best config
        best = comparison["best_config"]
        logger.info(f"\nBest Configuration:")
        logger.info(f"  ID: {best['experiment_id']}")
        logger.info(f"  Participation: {best['config']['participation_fraction']}")
        logger.info(f"  Heterogeneity: {best['config']['heterogeneity_mode']}")
        logger.info(f"  Local epochs: {best['config']['local_epochs']}")
        logger.info(f"  Final metrics: {best['final_metrics']}")
        
        # By participation
        logger.info(f"\nBy Participation Fraction:")
        for pf, metrics in sorted(comparison["by_participation"].items()):
            if "mae" in metrics:
                logger.info(f"  {pf}: MAE={metrics['mae']['mean']:.2f}±{metrics['mae']['std']:.2f}")
            elif "accuracy" in metrics:
                logger.info(f"  {pf}: Acc={metrics['accuracy']['mean']:.2%}±{metrics['accuracy']['std']:.2%}")
        
        # By heterogeneity
        logger.info(f"\nBy Heterogeneity Mode:")
        for het, metrics in comparison["by_heterogeneity"].items():
            if "mae" in metrics:
                logger.info(f"  {het}: MAE={metrics['mae']['mean']:.2f}±{metrics['mae']['std']:.2f}")
            elif "accuracy" in metrics:
                logger.info(f"  {het}: Acc={metrics['accuracy']['mean']:.2%}±{metrics['accuracy']['std']:.2%}")
        
        # By local epochs
        logger.info(f"\nBy Local Epochs:")
        for le, metrics in sorted(comparison["by_local_epochs"].items()):
            if "mae" in metrics:
                logger.info(f"  {le}: MAE={metrics['mae']['mean']:.2f}±{metrics['mae']['std']:.2f}")
            elif "accuracy" in metrics:
                logger.info(f"  {le}: Acc={metrics['accuracy']['mean']:.2%}±{metrics['accuracy']['std']:.2%}")
    
    def _convert_for_json(self, obj):
        """Convert numpy/torch types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(i) for i in obj]
        return obj


# --------------------- CLI ---------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Experiments Matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    # Matrix dimensions
    parser.add_argument(
        "--participation-fractions",
        type=float,
        nargs="+",
        help="Participation fractions to test",
    )
    parser.add_argument(
        "--heterogeneity-levels",
        type=str,
        nargs="+",
        choices=["uniform", "dirichlet", "extreme"],
        help="Heterogeneity modes to test",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        nargs="+",
        help="Local epochs settings to test",
    )
    
    # Base config overrides
    parser.add_argument("--num-clients", type=int, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, help="Number of FL rounds")
    parser.add_argument("--task", type=str, choices=["rul", "classification"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    
    # Single experiment mode
    parser.add_argument(
        "--single-experiment",
        action="store_true",
        help="Run single experiment instead of matrix",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.participation_fractions:
        config.participation_fractions = args.participation_fractions
    if args.heterogeneity_levels:
        config.heterogeneity_modes = args.heterogeneity_levels
    if args.local_epochs:
        config.local_epochs_list = args.local_epochs
    
    # Base config overrides
    if args.num_clients:
        config.base_config.num_clients = args.num_clients
    if args.num_rounds:
        config.base_config.num_rounds = args.num_rounds
    if args.task:
        config.base_config.task = args.task
    if args.batch_size:
        config.base_config.batch_size = args.batch_size
    if args.lr:
        config.base_config.lr = args.lr
    if args.seed:
        config.base_config.seed = args.seed
    
    # Run
    if args.single_experiment:
        # Run single experiment with base config
        experiment = FederatedExperiment(config.base_config)
        results = experiment.run()
    else:
        # Run full matrix
        matrix = FederatedExperimentMatrix(config)
        results = matrix.run()
    
    return results


if __name__ == "__main__":
    main()
