"""Round Orchestration & Logging — FL round execution, metrics, and checkpoints.

This module provides:
- RoundMetrics: dataclass for per-round metrics
- RoundLogger: logs metrics to file (JSON lines) and optionally console
- CheckpointManager: saves/loads model checkpoints
- FLOrchestrator: orchestrates FL rounds end-to-end

Design notes:
- Orchestrator is decoupled from transport (no HTTP here, just logic)
- Metrics are append-only (JSON lines format for easy parsing)
- Checkpoints include model state, round info, and config

Execution model:
Execution model:
- **Rounds are synchronous**: Clients within a round are executed in parallel
  using multiprocessing, but aggregation and round advancement remain synchronous. There is NO timeout enforcement at this
  layer. A hanging client will block the round indefinitely. For production use, implement
  timeouts in the transport layer (e.g., HTTP client timeout) or use async execution.
- Client selection samples from `available_client_ids` (if provided) or falls back to
  the static `client_ids` list. For dynamic client availability, provide a callback.
"""

import json
import os
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn

from .aggregator import FedAvgAggregator, sample_clients, apply_delta

def _train_client_worker(args):
    """
    Worker function for multiprocessing.
    Must be top-level to be picklable.
    """
    client_id, global_state_snapshot, config, train_client_fn = args
    result = train_client_fn(client_id, global_state_snapshot, config)
    return client_id, result



# ---------------------- Metrics ----------------------

@dataclass
class RoundMetrics:
    """Metrics collected for a single FL round."""
    round_id: int
    num_participants: int
    total_samples: int
    client_ids: List[str]
    
    # aggregated metrics from clients
    avg_client_loss: Optional[float] = None
    client_losses: Optional[Dict[str, float]] = None
    
    # global model evaluation (optional, computed after aggregation)
    global_loss: Optional[float] = None
    global_metrics: Optional[Dict[str, float]] = None
    
    # timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # status
    status: str = "completed"  # "completed", "failed", "partial"
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class RoundLogger:
    """Logs per-round metrics to JSON lines file and optionally console."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "fl_experiment",
        console_log: bool = True,
    ):
        """
        Args:
            log_dir: directory for log files
            experiment_name: prefix for log filename
            console_log: whether to also log to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        
        self.console_log = console_log
        self._logger = logging.getLogger(f"fl.{experiment_name}")
        if console_log and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_round(self, metrics: RoundMetrics) -> None:
        """Log metrics for a completed round."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "round",
            **metrics.to_dict(),
        }
        
        # append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        # console output
        if self.console_log:
            loss_str = f", avg_loss={metrics.avg_client_loss:.4f}" if metrics.avg_client_loss is not None else ""
            self._logger.info(
                f"Round {metrics.round_id}: "
                f"{metrics.num_participants} clients, "
                f"{metrics.total_samples} samples{loss_str}"
            )

    def log_event(self, event_type: str, data: Dict) -> None:
        """Log a generic event (start, end, error, etc.)."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            **data,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_history(self) -> List[Dict]:
        """Load all logged records from file."""
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


# ---------------------- Checkpointing ----------------------

class CheckpointManager:
    """Manages model checkpoints during FL training."""

    def __init__(self, checkpoint_dir: str, experiment_name: str = "fl_experiment"):
        """
        Args:
            checkpoint_dir: directory for checkpoint files
            experiment_name: prefix for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

    def save(
        self,
        model: nn.Module,
        round_id: int,
        metrics: Optional[RoundMetrics] = None,
        config: Optional[Dict] = None,
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: model to save
            round_id: current round number
            metrics: optional metrics for this round
            config: optional experiment config
            is_best: if True, also save as 'best' checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "round_id": round_id,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if metrics is not None:
            checkpoint["metrics"] = metrics.to_dict()
        if config is not None:
            checkpoint["config"] = config

        # save round checkpoint
        filename = f"{self.experiment_name}_round_{round_id:04d}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        # save as best if requested
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)

        # save as latest
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        return str(path)

    def load(self, path: Optional[str] = None, load_latest: bool = False) -> Dict:
        """Load a checkpoint.

        Args:
            path: explicit path to checkpoint file
            load_latest: if True, load the 'latest' checkpoint

        Returns:
            Checkpoint dict with model_state_dict, round_id, etc.
        """
        if load_latest:
            path = str(self.checkpoint_dir / f"{self.experiment_name}_latest.pt")
        if path is None:
            raise ValueError("Must provide path or set load_latest=True")
        return torch.load(path, weights_only=False)

    def load_model(self, model: nn.Module, path: Optional[str] = None, load_latest: bool = False) -> int:
        """Load model weights from checkpoint.

        Args:
            model: model to load weights into
            path: explicit path or None
            load_latest: load latest checkpoint

        Returns:
            Round ID from checkpoint
        """
        checkpoint = self.load(path, load_latest)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["round_id"]

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint files."""
        return sorted([str(p) for p in self.checkpoint_dir.glob(f"{self.experiment_name}_round_*.pt")])


# ---------------------- Orchestrator ----------------------

class FLOrchestrator:
    """Orchestrates federated learning rounds.

    This class ties together:
    - Client selection (from available clients)
    - Local training (via callback)
    - Aggregation (with enforced round binding)
    - Logging
    - Checkpointing

    The orchestrator is transport-agnostic: actual client communication
    is handled by a callback function.

    Round ownership:
    - The orchestrator owns `current_round` and increments it each round.
    - The aggregator is explicitly reset with `round_id` before each round.
    - Client updates must match the current round_id (enforced by aggregator).

    Client availability:
    - By default, samples from static `client_ids` list.
    - For dynamic availability, provide `available_client_ids_fn` callback.

    Execution model:
    - Synchronous: clients are called sequentially, no timeout at this layer.
    - For timeouts, implement in `train_client_fn` or transport layer.
    """

    def __init__(
        self,
        model: nn.Module,
        aggregator: FedAvgAggregator,
        logger: RoundLogger,
        checkpoint_manager: CheckpointManager,
        client_ids: List[str],
        config: Optional[Dict] = None,
        available_client_ids_fn: Optional[Callable[[], List[str]]] = None,
    ):
        """
        Args:
            model: global model to train
            aggregator: FedAvgAggregator instance (round_id will be managed by orchestrator)
            logger: RoundLogger for metrics
            checkpoint_manager: CheckpointManager for checkpoints
            client_ids: list of all registered client IDs (fallback if no availability fn)
            config: experiment configuration dict
            available_client_ids_fn: optional callback returning currently available client IDs
            config: experiment configuration dict
            available_client_ids_fn: optional callback returning currently available client IDs
        """
        self.model = model
        self.aggregator = aggregator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.client_ids = client_ids
        self.config = config or {}
        self._available_client_ids_fn = available_client_ids_fn
        
        self.current_round = 0
        self.best_loss = float("inf")
        self._history: List[RoundMetrics] = []

    def run_round(
        self,
        train_client_fn: Callable[[str, Dict[str, torch.Tensor], Dict], Dict],
        num_clients: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
        evaluate_fn: Optional[Callable[[nn.Module], Dict]] = None,
    ) -> RoundMetrics:
        """Execute a single FL round.

        Args:
            train_client_fn: callback(client_id, global_state_dict, config) -> {delta, num_samples, metrics}
                             The global_state_dict is a FROZEN SNAPSHOT (deepcopy) of model weights.
                             Clients must compute deltas relative to this snapshot.
                             This enforces FL isolation: no client can mutate global model state.
            num_clients: number of clients to sample
            fraction: fraction of clients to sample
            seed: random seed for client selection
            evaluate_fn: optional callback(model) -> {loss, ...} for global evaluation

        Returns:
            RoundMetrics for this round
        """
        self.current_round += 1
        round_id = self.current_round
        started_at = datetime.utcnow()

        # CRITICAL: reset aggregator with explicit round_id to enforce round binding
        # This ensures any stale updates from previous rounds are rejected
        self.aggregator.reset(round_id)

        # get available clients (dynamic if callback provided, else static list)
        if self._available_client_ids_fn is not None:
            available = self._available_client_ids_fn()
        else:
            available = self.client_ids

        # select clients from available pool
        selected = sample_clients(
            available,
            num_clients=num_clients,
            fraction=fraction,
            seed=seed,
        )

        # CRITICAL: Create frozen snapshot of global model state
        # This enforces FL isolation: each client receives identical, immutable state
        # No client can accidentally mutate the global model during training
        global_state_snapshot = deepcopy(self.model.state_dict())

        # collect updates from clients
        client_losses = {}
        errors = []

        max_workers = min(
            len(selected),
            max(1, os.cpu_count() // 2)
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _train_client_worker,
                    (client_id, global_state_snapshot, self.config, train_client_fn)
                )
                for client_id in selected
            ]

            for future in as_completed(futures):
                try:
                    client_id, result = future.result()

                    self.aggregator.add_update(
                        client_id=client_id,
                        delta=result["delta"],
                        num_samples=result["num_samples"],
                        round_id=round_id,
                    )

                    if "metrics" in result and "loss" in result["metrics"]:
                        client_losses[client_id] = result["metrics"]["loss"]

                except Exception as e:
                    errors.append(str(e))


        # aggregate and apply
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        if self.aggregator.num_updates == 0:
            metrics = RoundMetrics(
                round_id=round_id,
                num_participants=0,
                total_samples=0,
                client_ids=[],
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_seconds=duration,
                status="failed",
                error="No successful client updates",
            )
            self.logger.log_round(metrics)
            self._history.append(metrics)
            return metrics

        # apply aggregated delta to global model
        self.aggregator.aggregate_and_apply(self.model)

        # compute average client loss
        avg_loss = None
        if client_losses:
            avg_loss = sum(client_losses.values()) / len(client_losses)

        # optional global evaluation
        global_loss = None
        global_metrics = None
        if evaluate_fn is not None:
            try:
                eval_result = evaluate_fn(self.model)
                global_loss = eval_result.get("loss")
                global_metrics = eval_result
            except Exception as e:
                errors.append(f"eval: {str(e)}")

        # build metrics
        metrics = RoundMetrics(
            round_id=round_id,
            num_participants=self.aggregator.num_updates,
            total_samples=self.aggregator.total_samples,
            client_ids=self.aggregator.client_ids,
            avg_client_loss=avg_loss,
            client_losses=client_losses,
            global_loss=global_loss,
            global_metrics=global_metrics,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration,
            status="completed" if not errors else "partial",
            error="; ".join(errors) if errors else None,
        )

        # log metrics
        self.logger.log_round(metrics)
        self._history.append(metrics)

        # checkpoint
        # IMPORTANT: "best" is determined by global_loss (validation), NOT avg_client_loss (training)
        # Using training loss for model selection is scientifically invalid in non-iid FL
        is_best = global_loss is not None and global_loss < self.best_loss
        if is_best:
            self.best_loss = global_loss
        self.checkpoint_manager.save(
            self.model,
            round_id,
            metrics=metrics,
            config=self.config,
            is_best=is_best,
        )

        return metrics

    def run(
        self,
        num_rounds: int,
        train_client_fn: Callable[[str, Dict[str, torch.Tensor], Dict], Dict],
        num_clients: Optional[int] = None,
        fraction: Optional[float] = None,
        evaluate_fn: Optional[Callable[[nn.Module], Dict]] = None,
        seed: Optional[int] = None,
    ) -> List[RoundMetrics]:
        """Run multiple FL rounds.

        Args:
            num_rounds: number of rounds to run
            train_client_fn: callback(client_id, global_state_dict, config) -> {delta, num_samples, metrics}
            num_clients: clients per round
            fraction: fraction of clients per round
            evaluate_fn: optional callback(model) -> {loss, ...} for global evaluation
            seed: base seed for reproducibility (round seed = seed + round_id)

        Returns:
            List of RoundMetrics for all rounds
        """
        self.logger.log_event("experiment_start", {
            "num_rounds": num_rounds,
            "num_clients_total": len(self.client_ids),
            "config": self.config,
        })

        results = []
        for r in range(num_rounds):
            round_seed = (seed + r) if seed is not None else None
            metrics = self.run_round(
                train_client_fn=train_client_fn,
                num_clients=num_clients,
                fraction=fraction,
                seed=round_seed,
                evaluate_fn=evaluate_fn,
            )
            results.append(metrics)

        self.logger.log_event("experiment_end", {
            "total_rounds": len(results),
            "best_loss": self.best_loss if self.best_loss < float("inf") else None,
        })

        return results



    @property
    def history(self) -> List[RoundMetrics]:
        """Get history of all rounds."""
        return self._history.copy()
