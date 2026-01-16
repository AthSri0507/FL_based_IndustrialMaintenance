"""Experiment Logging Module — Comprehensive per-round metrics tracking.

This module provides advanced experiment logging capabilities:

1. **Structured Metrics Logging**: Per-round, per-client, and global metrics
2. **Multiple Output Formats**: JSONL, CSV, TensorBoard, console
3. **Real-time Aggregation**: Running statistics and summaries
4. **Experiment Comparison**: Compare metrics across experiments
5. **Visualization Helpers**: Export data for plotting

Usage:
    from src.utils.logging import ExperimentLogger, MetricsAggregator
    
    # Create logger
    logger = ExperimentLogger("experiments/logs", "my_experiment")
    
    # Log round metrics
    logger.log_round(round_id=1, metrics={
        "global_loss": 0.5,
        "global_mae": 10.2,
        "clients": {"client_0": {"loss": 0.4}, "client_1": {"loss": 0.6}},
    })
    
    # Get summary
    summary = logger.get_summary()
    
    # Export for visualization
    logger.export_csv("metrics.csv")

Formats:
- JSONL: Append-only, one JSON record per line (default)
- CSV: Tabular format for spreadsheets
- TensorBoard: For TensorBoard visualization (if tensorboard installed)
"""

import csv
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ===================== Data Classes =====================


@dataclass
class ClientRoundMetrics:
    """Metrics for a single client in a single round."""
    client_id: str
    round_id: int
    num_samples: int
    
    # Training metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # Task-specific metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    
    # Training details
    local_epochs_completed: int = 0
    early_stopped: bool = False
    
    # Timing
    training_time_seconds: Optional[float] = None
    
    # Extra metrics (flexible)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten extra into main dict
        extra = d.pop("extra", {})
        d.update(extra)
        return d


@dataclass
class RoundLogEntry:
    """Complete log entry for a federated learning round."""
    round_id: int
    timestamp: str
    
    # Participation
    num_participants: int
    total_samples: int
    client_ids: List[str]
    
    # Global model metrics (evaluated on test set)
    global_loss: Optional[float] = None
    global_mae: Optional[float] = None
    global_rmse: Optional[float] = None
    global_accuracy: Optional[float] = None
    global_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Aggregated client metrics
    avg_client_train_loss: Optional[float] = None
    avg_client_val_loss: Optional[float] = None
    
    # Per-client metrics
    client_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Timing
    round_duration_seconds: Optional[float] = None
    aggregation_time_seconds: Optional[float] = None
    
    # Status
    status: str = "completed"
    errors: List[str] = field(default_factory=list)
    
    # Configuration snapshot (optional)
    config_snapshot: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    experiment_name: str
    total_rounds: int
    total_clients: int
    total_samples_trained: int
    
    # Best metrics
    best_global_loss: Optional[float] = None
    best_global_loss_round: Optional[int] = None
    best_global_mae: Optional[float] = None
    best_global_mae_round: Optional[int] = None
    
    # Final metrics
    final_global_loss: Optional[float] = None
    final_global_mae: Optional[float] = None
    final_global_rmse: Optional[float] = None
    
    # Averages
    avg_round_duration: Optional[float] = None
    avg_participants_per_round: Optional[float] = None
    
    # Timing
    total_training_time: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================== Metrics Aggregator =====================


class MetricsAggregator:
    """Aggregates metrics across rounds for running statistics."""
    
    def __init__(self):
        self._round_metrics: List[Dict[str, Any]] = []
        self._client_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._best_metrics: Dict[str, Tuple[float, int]] = {}  # metric_name -> (value, round)
        self._metric_history: Dict[str, List[float]] = defaultdict(list)
    
    def add_round(self, round_id: int, metrics: Dict[str, Any]) -> None:
        """Add metrics for a round."""
        metrics["round_id"] = round_id
        self._round_metrics.append(metrics)
        
        # Track history of scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_id"):
                self._metric_history[key].append(value)
                
                # Track best (assuming lower is better for loss/mae/rmse)
                is_lower_better = any(x in key.lower() for x in ["loss", "mae", "rmse", "error"])
                
                if key not in self._best_metrics:
                    self._best_metrics[key] = (value, round_id)
                else:
                    current_best, _ = self._best_metrics[key]
                    if is_lower_better and value < current_best:
                        self._best_metrics[key] = (value, round_id)
                    elif not is_lower_better and value > current_best:
                        self._best_metrics[key] = (value, round_id)
    
    def add_client_round(self, client_id: str, round_id: int, metrics: Dict[str, Any]) -> None:
        """Add metrics for a client in a specific round."""
        metrics["round_id"] = round_id
        self._client_metrics[client_id].append(metrics)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a metric across rounds."""
        return self._metric_history.get(metric_name, [])
    
    def get_best(self, metric_name: str) -> Optional[Tuple[float, int]]:
        """Get best value and round for a metric."""
        return self._best_metrics.get(metric_name)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        history = self._metric_history.get(metric_name, [])
        return history[-1] if history else None
    
    def get_mean(self, metric_name: str) -> Optional[float]:
        """Get mean of a metric across all rounds."""
        history = self._metric_history.get(metric_name, [])
        return statistics.mean(history) if history else None
    
    def get_std(self, metric_name: str) -> Optional[float]:
        """Get standard deviation of a metric."""
        history = self._metric_history.get(metric_name, [])
        return statistics.stdev(history) if len(history) > 1 else None
    
    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        """Get summary for a specific client."""
        client_rounds = self._client_metrics.get(client_id, [])
        if not client_rounds:
            return {}
        
        summary = {
            "num_rounds_participated": len(client_rounds),
            "total_samples": sum(r.get("num_samples", 0) for r in client_rounds),
        }
        
        # Compute averages for numeric metrics
        for key in client_rounds[0].keys():
            if key in ["round_id", "client_id"]:
                continue
            values = [r[key] for r in client_rounds if key in r and isinstance(r[key], (int, float))]
            if values:
                summary[f"avg_{key}"] = statistics.mean(values)
        
        return summary
    
    def get_all_client_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all clients."""
        return {cid: self.get_client_summary(cid) for cid in self._client_metrics.keys()}
    
    def to_dataframe(self):
        """Convert round metrics to pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self._round_metrics)
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")


# ===================== Experiment Logger =====================


class ExperimentLogger:
    """Comprehensive experiment logger with multiple output formats."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        console_output: bool = True,
        enable_tensorboard: bool = False,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            console_output: Whether to log to console
            enable_tensorboard: Whether to enable TensorBoard logging
            log_level: Logging level for console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Create experiment subdirectory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.jsonl_file = self.experiment_dir / "rounds.jsonl"
        self.events_file = self.experiment_dir / "events.jsonl"
        self.summary_file = self.experiment_dir / "summary.json"
        
        # Metrics aggregator
        self.aggregator = MetricsAggregator()
        
        # Console logger
        self._console_output = console_output
        self._logger = logging.getLogger(f"experiment.{experiment_name}")
        self._logger.setLevel(log_level)
        
        if console_output and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(handler)
        
        # TensorBoard writer (optional)
        self._tb_writer = None
        if enable_tensorboard:
            self._init_tensorboard()
        
        # State tracking
        self._start_time = time.time()
        self._started_at = datetime.utcnow().isoformat()
        self._current_round = 0
        self._total_samples = 0
        self._all_clients: set = set()
        
        # Log experiment start
        self.log_event("experiment_start", {
            "experiment_name": experiment_name,
            "log_dir": str(self.experiment_dir),
        })
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.experiment_dir / "tensorboard"
            self._tb_writer = SummaryWriter(str(tb_dir))
            self._logger.info(f"TensorBoard logging enabled at {tb_dir}")
        except ImportError:
            self._logger.warning("TensorBoard not available, disabling TB logging")
    
    def log_round(
        self,
        round_id: int,
        num_participants: int,
        total_samples: int,
        client_ids: List[str],
        global_metrics: Optional[Dict[str, float]] = None,
        client_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        round_duration: Optional[float] = None,
        status: str = "completed",
        errors: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics for a completed round.
        
        Args:
            round_id: Round number
            num_participants: Number of participating clients
            total_samples: Total samples trained this round
            client_ids: List of participating client IDs
            global_metrics: Global model metrics (loss, mae, etc.)
            client_metrics: Per-client metrics dict
            round_duration: Round duration in seconds
            status: Round status
            errors: List of errors if any
            extra: Extra data to include
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Build log entry
        entry = RoundLogEntry(
            round_id=round_id,
            timestamp=timestamp,
            num_participants=num_participants,
            total_samples=total_samples,
            client_ids=client_ids,
            round_duration_seconds=round_duration,
            status=status,
            errors=errors or [],
        )
        
        # Add global metrics
        if global_metrics:
            entry.global_metrics = global_metrics
            entry.global_loss = global_metrics.get("loss")
            entry.global_mae = global_metrics.get("mae")
            entry.global_rmse = global_metrics.get("rmse")
            entry.global_accuracy = global_metrics.get("accuracy")
        
        # Add client metrics
        if client_metrics:
            entry.client_metrics = client_metrics
            
            # Compute averages
            train_losses = [m.get("train_loss") for m in client_metrics.values() if m.get("train_loss") is not None]
            val_losses = [m.get("val_loss") for m in client_metrics.values() if m.get("val_loss") is not None]
            
            if train_losses:
                entry.avg_client_train_loss = statistics.mean(train_losses)
            if val_losses:
                entry.avg_client_val_loss = statistics.mean(val_losses)
        
        # Write to JSONL
        entry_dict = entry.to_dict()
        if extra:
            entry_dict.update(extra)
        
        with open(self.jsonl_file, "a") as f:
            f.write(json.dumps(entry_dict) + "\n")
        
        # Update aggregator
        agg_metrics = {
            "num_participants": num_participants,
            "total_samples": total_samples,
            "round_duration": round_duration,
        }
        if global_metrics:
            agg_metrics.update({f"global_{k}": v for k, v in global_metrics.items()})
        if entry.avg_client_train_loss:
            agg_metrics["avg_client_train_loss"] = entry.avg_client_train_loss
        
        self.aggregator.add_round(round_id, agg_metrics)
        
        # Update client aggregator
        if client_metrics:
            for cid, metrics in client_metrics.items():
                self.aggregator.add_client_round(cid, round_id, metrics)
                self._all_clients.add(cid)
        
        # Update state
        self._current_round = round_id
        self._total_samples += total_samples
        
        # TensorBoard logging
        if self._tb_writer:
            if global_metrics:
                for key, value in global_metrics.items():
                    self._tb_writer.add_scalar(f"global/{key}", value, round_id)
            
            self._tb_writer.add_scalar("round/participants", num_participants, round_id)
            self._tb_writer.add_scalar("round/samples", total_samples, round_id)
            
            if round_duration:
                self._tb_writer.add_scalar("round/duration_seconds", round_duration, round_id)
        
        # Console output
        if self._console_output:
            msg = f"Round {round_id}: {num_participants} clients, {total_samples} samples"
            if global_metrics:
                if "mae" in global_metrics:
                    msg += f", MAE={global_metrics['mae']:.2f}"
                if "loss" in global_metrics:
                    msg += f", loss={global_metrics['loss']:.4f}"
                if "accuracy" in global_metrics:
                    msg += f", acc={global_metrics['accuracy']:.2%}"
            self._logger.info(msg)
    
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a generic event (start, end, error, checkpoint, etc.)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data or {},
        }
        
        with open(self.events_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        if self._console_output and event_type in ["error", "warning"]:
            self._logger.warning(f"Event [{event_type}]: {data}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        self.log_event("config_saved", {"path": str(config_file)})
    
    def get_summary(self) -> ExperimentSummary:
        """Get current experiment summary."""
        summary = ExperimentSummary(
            experiment_name=self.experiment_name,
            total_rounds=self._current_round,
            total_clients=len(self._all_clients),
            total_samples_trained=self._total_samples,
            started_at=self._started_at,
            total_training_time=time.time() - self._start_time,
        )
        
        # Best metrics
        best_loss = self.aggregator.get_best("global_loss")
        if best_loss:
            summary.best_global_loss, summary.best_global_loss_round = best_loss
        
        best_mae = self.aggregator.get_best("global_mae")
        if best_mae:
            summary.best_global_mae, summary.best_global_mae_round = best_mae
        
        # Final metrics
        summary.final_global_loss = self.aggregator.get_latest("global_loss")
        summary.final_global_mae = self.aggregator.get_latest("global_mae")
        summary.final_global_rmse = self.aggregator.get_latest("global_rmse")
        
        # Averages
        summary.avg_round_duration = self.aggregator.get_mean("round_duration")
        summary.avg_participants_per_round = self.aggregator.get_mean("num_participants")
        
        return summary
    
    def save_summary(self) -> str:
        """Save experiment summary to file."""
        summary = self.get_summary()
        summary.completed_at = datetime.utcnow().isoformat()
        
        with open(self.summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        return str(self.summary_file)
    
    def export_csv(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Export round metrics to CSV.
        
        Args:
            filepath: Output path (default: rounds.csv in experiment dir)
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.experiment_dir / "rounds.csv"
        else:
            filepath = Path(filepath)
        
        # Read JSONL
        rounds = self.load_rounds()
        
        if not rounds:
            raise ValueError("No rounds to export")
        
        # Flatten nested dicts
        flat_rounds = []
        for r in rounds:
            flat = {}
            for key, value in r.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, (int, float, str, bool, type(None))):
                            flat[f"{key}_{subkey}"] = subval
                elif isinstance(value, list):
                    flat[key] = ",".join(str(v) for v in value)
                else:
                    flat[key] = value
            flat_rounds.append(flat)
        
        # Get all columns
        all_columns = set()
        for r in flat_rounds:
            all_columns.update(r.keys())
        columns = sorted(all_columns)
        
        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat_rounds)
        
        return str(filepath)
    
    def export_client_metrics_csv(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Export per-client metrics to CSV."""
        if filepath is None:
            filepath = self.experiment_dir / "client_metrics.csv"
        else:
            filepath = Path(filepath)
        
        summaries = self.aggregator.get_all_client_summaries()
        
        if not summaries:
            raise ValueError("No client metrics to export")
        
        rows = []
        for client_id, summary in summaries.items():
            row = {"client_id": client_id}
            row.update(summary)
            rows.append(row)
        
        columns = sorted(set().union(*[r.keys() for r in rows]))
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        
        return str(filepath)
    
    def load_rounds(self) -> List[Dict[str, Any]]:
        """Load all round logs from file."""
        rounds = []
        if self.jsonl_file.exists():
            with open(self.jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        rounds.append(json.loads(line))
        return rounds
    
    def load_events(self) -> List[Dict[str, Any]]:
        """Load all events from file."""
        events = []
        if self.events_file.exists():
            with open(self.events_file, "r") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        return events
    
    def get_metric_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get history of a metric as (round_id, value) pairs."""
        rounds = self.load_rounds()
        history = []
        
        for r in rounds:
            round_id = r.get("round_id")
            
            # Check in global_metrics
            if "global_metrics" in r and metric_name in r["global_metrics"]:
                history.append((round_id, r["global_metrics"][metric_name]))
            elif metric_name in r:
                value = r[metric_name]
                if isinstance(value, (int, float)):
                    history.append((round_id, value))
        
        return history
    
    def finalize(self) -> str:
        """Finalize logging and save summary."""
        self.log_event("experiment_end", {
            "total_rounds": self._current_round,
            "total_samples": self._total_samples,
            "duration_seconds": time.time() - self._start_time,
        })
        
        summary_path = self.save_summary()
        
        if self._tb_writer:
            self._tb_writer.close()
        
        self._logger.info(f"Experiment complete. Summary saved to {summary_path}")
        
        return summary_path
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_event("error", {
                "type": str(exc_type.__name__),
                "message": str(exc_val),
            })
        self.finalize()
        return False


# ===================== Experiment Comparison =====================


class ExperimentComparison:
    """Compare metrics across multiple experiments."""
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Args:
            log_dir: Directory containing experiment logs
        """
        self.log_dir = Path(log_dir)
    
    def list_experiments(self) -> List[str]:
        """List all experiments in log directory."""
        experiments = []
        for path in self.log_dir.iterdir():
            if path.is_dir() and (path / "summary.json").exists():
                experiments.append(path.name)
        return sorted(experiments)
    
    def load_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Load summary for an experiment."""
        summary_file = self.log_dir / experiment_name / "summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary not found for {experiment_name}")
        
        with open(summary_file, "r") as f:
            return json.load(f)
    
    def compare_summaries(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare summaries across experiments.
        
        Args:
            experiment_names: List of experiments to compare (default: all)
        
        Returns:
            Comparison dict with metrics per experiment
        """
        if experiment_names is None:
            experiment_names = self.list_experiments()
        
        comparison = {
            "experiments": {},
            "best_by_metric": {},
        }
        
        for name in experiment_names:
            try:
                summary = self.load_summary(name)
                comparison["experiments"][name] = summary
            except FileNotFoundError:
                continue
        
        # Find best experiment for each metric
        metrics_to_compare = [
            ("best_global_loss", "min"),
            ("best_global_mae", "min"),
            ("final_global_loss", "min"),
            ("final_global_mae", "min"),
        ]
        
        for metric, mode in metrics_to_compare:
            values = {}
            for name, summary in comparison["experiments"].items():
                if metric in summary and summary[metric] is not None:
                    values[name] = summary[metric]
            
            if values:
                if mode == "min":
                    best_name = min(values, key=values.get)
                else:
                    best_name = max(values, key=values.get)
                
                comparison["best_by_metric"][metric] = {
                    "experiment": best_name,
                    "value": values[best_name],
                }
        
        return comparison
    
    def to_dataframe(self, experiment_names: Optional[List[str]] = None):
        """Convert comparison to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        comparison = self.compare_summaries(experiment_names)
        return pd.DataFrame(comparison["experiments"]).T


# ===================== Convenience Functions =====================


def create_experiment_logger(
    output_dir: Union[str, Path],
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    enable_tensorboard: bool = False,
) -> ExperimentLogger:
    """Create and initialize an experiment logger.
    
    Args:
        output_dir: Output directory for logs
        experiment_name: Experiment name
        config: Optional config to log
        enable_tensorboard: Enable TensorBoard
    
    Returns:
        Configured ExperimentLogger
    """
    logger = ExperimentLogger(
        log_dir=output_dir,
        experiment_name=experiment_name,
        enable_tensorboard=enable_tensorboard,
    )
    
    if config:
        logger.log_config(config)
    
    return logger


def load_experiment_metrics(log_dir: Union[str, Path], experiment_name: str) -> List[Dict[str, Any]]:
    """Load metrics from an experiment.
    
    Args:
        log_dir: Log directory
        experiment_name: Experiment folder name
    
    Returns:
        List of round metrics
    """
    jsonl_file = Path(log_dir) / experiment_name / "rounds.jsonl"
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"No metrics found for {experiment_name}")
    
    rounds = []
    with open(jsonl_file, "r") as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))
    
    return rounds


def get_metric_series(
    log_dir: Union[str, Path],
    experiment_name: str,
    metric_name: str,
) -> Tuple[List[int], List[float]]:
    """Get a metric series for plotting.
    
    Args:
        log_dir: Log directory
        experiment_name: Experiment name
        metric_name: Metric to extract
    
    Returns:
        Tuple of (round_ids, values)
    """
    rounds = load_experiment_metrics(log_dir, experiment_name)
    
    round_ids = []
    values = []
    
    for r in rounds:
        round_id = r.get("round_id")
        
        # Try different locations for the metric
        value = None
        if metric_name in r:
            value = r[metric_name]
        elif "global_metrics" in r and metric_name in r.get("global_metrics", {}):
            value = r["global_metrics"][metric_name]
        
        if isinstance(value, (int, float)):
            round_ids.append(round_id)
            values.append(value)
    
    return round_ids, values
