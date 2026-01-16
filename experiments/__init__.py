# Experiments package for federated learning baselines and experiments

"""
This package contains experiment scripts and baselines for comparing
federated learning approaches.

Modules:
- centralized_baseline: Centralized training pipeline (baseline)
- local_only_baseline: Local-only per-client training baseline
- federated_matrix: Comprehensive federated experiments framework
"""

from .centralized_baseline import CentralizedConfig
from .local_only_baseline import LocalOnlyBaseline, LocalOnlyConfig
from .federated_matrix import (
    FederatedExperiment,
    FederatedExperimentConfig,
    FederatedExperimentMatrix,
    ExperimentMatrixConfig,
)

__all__ = [
    "CentralizedConfig",
    "LocalOnlyBaseline",
    "LocalOnlyConfig",
    "FederatedExperiment",
    "FederatedExperimentConfig",
    "FederatedExperimentMatrix",
    "ExperimentMatrixConfig",
]
