# Server module - imports with optional FastAPI dependency
try:
    from .app import app, FLServer
    _HAS_FASTAPI = True
except ImportError:
    app = None
    FLServer = None
    _HAS_FASTAPI = False

from .aggregator import (
    fedavg_aggregate,
    apply_delta,
    sample_clients,
    FedAvgAggregator,
)
from .orchestrator import (
    FLOrchestrator,
    RoundLogger,
    CheckpointManager,
    RoundMetrics,
)

__all__ = [
    "app",
    "FLServer",
    "fedavg_aggregate",
    "apply_delta",
    "sample_clients",
    "FedAvgAggregator",
    "FLOrchestrator",
    "RoundLogger",
    "CheckpointManager",
    "RoundMetrics",
]
