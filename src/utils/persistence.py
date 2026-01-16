"""Configuration Persistence Module for Federated Learning Experiments.

This module provides comprehensive configuration management:

1. **Config Persistence**: Save/load experiment configs in YAML/JSON
2. **Checkpointing**: Enhanced checkpoint management with config bundling
3. **Experiment Resume**: Resume experiments from checkpoints
4. **Config Validation**: Schema validation for configs
5. **Config Versioning**: Track config changes across experiments

Usage:
    from src.utils.persistence import (
        ConfigManager,
        ExperimentCheckpoint,
        save_config,
        load_config,
    )
    
    # Save config
    config_manager = ConfigManager("experiments/configs")
    config_manager.save(my_config, "experiment_001")
    
    # Load config
    loaded_config = config_manager.load("experiment_001")
    
    # Full experiment checkpoint with config
    checkpoint = ExperimentCheckpoint("experiments/checkpoints", "my_experiment")
    checkpoint.save_full(model, config, round_id=10, metrics=metrics)
    
    # Resume experiment
    model, config, round_id = checkpoint.load_and_resume(model)

Supported formats:
- YAML (preferred for human readability)
- JSON (for programmatic access)
- Python dataclasses (automatic serialization)
"""

import copy
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ===================== Config Serialization =====================


def _serialize_value(value: Any) -> Any:
    """Serialize a value to JSON-compatible format."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    elif is_dataclass(value) and not isinstance(value, type):
        return _serialize_value(asdict(value))
    elif isinstance(value, Path):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif hasattr(value, "__dict__"):
        return _serialize_value(vars(value))
    else:
        return str(value)


def _deserialize_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Deserialize a dict to a dataclass instance."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    
    # Get field names and types
    field_names = {f.name for f in fields(cls)}
    
    # Filter to only known fields
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    
    return cls(**filtered_data)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to serializable dict."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_value(asdict(obj))
    elif isinstance(obj, dict):
        return _serialize_value(obj)
    else:
        raise TypeError(f"Expected dataclass or dict, got {type(obj)}")


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert dict to dataclass instance."""
    return _deserialize_to_dataclass(data, cls)


# ===================== YAML/JSON I/O =====================


def _ensure_yaml():
    """Check if PyYAML is available."""
    try:
        import yaml
        return yaml
    except ImportError:
        return None


def save_config(
    config: Union[Dict, Any],
    path: Union[str, Path],
    format: str = "auto",
) -> str:
    """Save configuration to file.
    
    Args:
        config: Config dict or dataclass
        path: Output path (extension determines format if auto)
        format: "yaml", "json", or "auto"
    
    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict if dataclass
    if is_dataclass(config) and not isinstance(config, type):
        config_dict = dataclass_to_dict(config)
    elif isinstance(config, dict):
        config_dict = _serialize_value(config)
    else:
        raise TypeError(f"Config must be dict or dataclass, got {type(config)}")
    
    # Add metadata
    config_dict["_metadata"] = {
        "saved_at": datetime.utcnow().isoformat(),
        "config_hash": _compute_hash(config_dict),
    }
    
    # Determine format
    if format == "auto":
        if path.suffix in [".yaml", ".yml"]:
            format = "yaml"
        else:
            format = "json"
    
    # Save
    if format == "yaml":
        yaml = _ensure_yaml()
        if yaml is None:
            logger.warning("PyYAML not installed, falling back to JSON")
            path = path.with_suffix(".json")
            format = "json"
        else:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {path}")
            return str(path)
    
    if format == "json":
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config to {path}")
        return str(path)
    
    raise ValueError(f"Unknown format: {format}")


def load_config(
    path: Union[str, Path],
    cls: Optional[Type[T]] = None,
) -> Union[Dict[str, Any], T]:
    """Load configuration from file.
    
    Args:
        path: Path to config file
        cls: Optional dataclass type to deserialize into
    
    Returns:
        Config dict or dataclass instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load based on extension
    if path.suffix in [".yaml", ".yml"]:
        yaml = _ensure_yaml()
        if yaml is None:
            raise ImportError("PyYAML required for .yaml files")
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            config_dict = json.load(f)
    
    # Remove metadata before returning
    config_dict.pop("_metadata", None)
    
    # Convert to dataclass if requested
    if cls is not None:
        return dict_to_dataclass(config_dict, cls)
    
    return config_dict


def _compute_hash(config: Dict) -> str:
    """Compute deterministic hash of config for tracking."""
    # Remove metadata for hashing
    config_copy = {k: v for k, v in config.items() if not k.startswith("_")}
    config_str = json.dumps(config_copy, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ===================== ConfigManager =====================


class ConfigManager:
    """Manages experiment configurations with versioning and validation."""
    
    def __init__(
        self,
        config_dir: Union[str, Path],
        default_format: str = "yaml",
    ):
        """
        Args:
            config_dir: Directory for config files
            default_format: Default format ("yaml" or "json")
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_format = default_format
        self._history_file = self.config_dir / "config_history.json"
    
    def save(
        self,
        config: Union[Dict, Any],
        name: str,
        format: Optional[str] = None,
        track_history: bool = True,
    ) -> str:
        """Save a named configuration.
        
        Args:
            config: Config dict or dataclass
            name: Config name (without extension)
            format: Override default format
            track_history: Whether to track in history
        
        Returns:
            Path to saved config
        """
        format = format or self.default_format
        ext = ".yaml" if format == "yaml" else ".json"
        path = self.config_dir / f"{name}{ext}"
        
        saved_path = save_config(config, path, format=format)
        
        if track_history:
            self._add_to_history(name, saved_path, config)
        
        return saved_path
    
    def load(
        self,
        name: str,
        cls: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], T]:
        """Load a named configuration.
        
        Args:
            name: Config name (with or without extension)
            cls: Optional dataclass type
        
        Returns:
            Config dict or dataclass
        """
        # Try with extension first
        path = self.config_dir / name
        if not path.exists():
            # Try adding extensions
            for ext in [".yaml", ".yml", ".json"]:
                candidate = self.config_dir / f"{name}{ext}"
                if candidate.exists():
                    path = candidate
                    break
        
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {name}")
        
        return load_config(path, cls)
    
    def list_configs(self) -> List[str]:
        """List all saved config names."""
        configs = []
        for ext in ["*.yaml", "*.yml", "*.json"]:
            for path in self.config_dir.glob(ext):
                if not path.name.startswith("_") and path.name != "config_history.json":
                    configs.append(path.stem)
        return sorted(set(configs))
    
    def delete(self, name: str) -> bool:
        """Delete a named configuration."""
        deleted = False
        for ext in [".yaml", ".yml", ".json"]:
            path = self.config_dir / f"{name}{ext}"
            if path.exists():
                path.unlink()
                deleted = True
        return deleted
    
    def copy(self, source_name: str, dest_name: str) -> str:
        """Copy a configuration to a new name."""
        config = self.load(source_name)
        return self.save(config, dest_name)
    
    def get_history(self) -> List[Dict]:
        """Get config save history."""
        if not self._history_file.exists():
            return []
        with open(self._history_file, "r") as f:
            return json.load(f)
    
    def _add_to_history(self, name: str, path: str, config: Any):
        """Add entry to history."""
        history = self.get_history()
        
        if is_dataclass(config) and not isinstance(config, type):
            config_dict = dataclass_to_dict(config)
        else:
            config_dict = _serialize_value(config)
        
        entry = {
            "name": name,
            "path": path,
            "saved_at": datetime.utcnow().isoformat(),
            "config_hash": _compute_hash(config_dict),
        }
        history.append(entry)
        
        with open(self._history_file, "w") as f:
            json.dump(history, f, indent=2)


# ===================== ExperimentCheckpoint =====================


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    round_id: int
    timestamp: str
    config_hash: str
    total_rounds: Optional[int] = None
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    total_samples_trained: int = 0
    num_clients: int = 0
    experiment_name: str = ""
    notes: str = ""


class ExperimentCheckpoint:
    """Enhanced checkpoint manager with full experiment state.
    
    Saves:
    - Model weights
    - Optimizer state (optional)
    - Full experiment config
    - Training history/metrics
    - Random states for reproducibility
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_name: str,
        max_checkpoints: int = 5,
    ):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name prefix for checkpoint files
            max_checkpoints: Max round checkpoints to keep (excluding best/latest)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        
        # Subdirectories
        self.models_dir = self.checkpoint_dir / "models"
        self.configs_dir = self.checkpoint_dir / "configs"
        self.history_dir = self.checkpoint_dir / "history"
        
        for d in [self.models_dir, self.configs_dir, self.history_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_full(
        self,
        model: nn.Module,
        config: Union[Dict, Any],
        round_id: int,
        metrics: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        history: Optional[List[Dict]] = None,
        is_best: bool = False,
        extra: Optional[Dict] = None,
    ) -> str:
        """Save complete experiment checkpoint.
        
        Args:
            model: Model to checkpoint
            config: Experiment configuration
            round_id: Current round number
            metrics: Current metrics dict
            optimizer: Optional optimizer state
            scheduler: Optional LR scheduler state
            history: Training history (list of per-round metrics)
            is_best: Mark as best checkpoint
            extra: Extra data to save
        
        Returns:
            Path to saved checkpoint
        """
        # Serialize config
        if is_dataclass(config) and not isinstance(config, type):
            config_dict = dataclass_to_dict(config)
        else:
            config_dict = _serialize_value(config)
        
        # Build checkpoint
        checkpoint = {
            "round_id": round_id,
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "metadata": {
                "experiment_name": self.experiment_name,
                "timestamp": datetime.utcnow().isoformat(),
                "config_hash": _compute_hash(config_dict),
                "pytorch_version": torch.__version__,
            },
        }
        
        if metrics is not None:
            checkpoint["metrics"] = _serialize_value(metrics)
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if history is not None:
            checkpoint["history"] = _serialize_value(history)
        
        if extra is not None:
            checkpoint["extra"] = _serialize_value(extra)
        
        # Save random states for reproducibility
        checkpoint["random_states"] = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        try:
            import numpy as np
            checkpoint["random_states"]["numpy"] = np.random.get_state()
        except ImportError:
            pass
        
        try:
            import random
            checkpoint["random_states"]["python"] = random.getstate()
        except Exception:
            pass
        
        # Save checkpoint file
        filename = f"{self.experiment_name}_round_{round_id:04d}.pt"
        path = self.models_dir / filename
        torch.save(checkpoint, path)
        
        # Save as latest
        latest_path = self.models_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save as best if requested
        if is_best:
            best_path = self.models_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
        
        # Save config separately for easy access
        config_path = self.configs_dir / f"{self.experiment_name}_config.yaml"
        save_config(config_dict, config_path)
        
        # Save history separately
        if history is not None:
            history_path = self.history_dir / f"{self.experiment_name}_history.json"
            with open(history_path, "w") as f:
                json.dump(_serialize_value(history), f, indent=2)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {path}")
        return str(path)
    
    def load_full(
        self,
        model: nn.Module,
        path: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_random_states: bool = True,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load complete experiment checkpoint.
        
        Args:
            model: Model to load weights into
            path: Explicit checkpoint path
            load_best: Load best checkpoint
            load_latest: Load latest checkpoint
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            restore_random_states: Restore RNG states
            map_location: Device mapping for torch.load
        
        Returns:
            Dict with round_id, config, metrics, history, etc.
        """
        # Determine path
        if path is not None:
            checkpoint_path = Path(path)
        elif load_best:
            checkpoint_path = self.models_dir / f"{self.experiment_name}_best.pt"
        elif load_latest:
            checkpoint_path = self.models_dir / f"{self.experiment_name}_latest.pt"
        else:
            raise ValueError("Must provide path, load_best, or load_latest")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore optimizer if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore scheduler if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore random states
        if restore_random_states and "random_states" in checkpoint:
            states = checkpoint["random_states"]
            
            if states.get("torch") is not None:
                torch.set_rng_state(states["torch"])
            
            if torch.cuda.is_available() and states.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(states["torch_cuda"])
            
            try:
                import numpy as np
                if states.get("numpy") is not None:
                    np.random.set_state(states["numpy"])
            except ImportError:
                pass
            
            try:
                import random
                if states.get("python") is not None:
                    random.setstate(states["python"])
            except Exception:
                pass
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (round {checkpoint['round_id']})")
        
        return {
            "round_id": checkpoint["round_id"],
            "config": checkpoint.get("config", {}),
            "metrics": checkpoint.get("metrics", {}),
            "history": checkpoint.get("history", []),
            "metadata": checkpoint.get("metadata", {}),
            "extra": checkpoint.get("extra", {}),
        }
    
    def load_config_only(self) -> Dict[str, Any]:
        """Load just the config without loading model weights."""
        config_path = self.configs_dir / f"{self.experiment_name}_config.yaml"
        if config_path.exists():
            return load_config(config_path)
        
        # Fall back to loading from checkpoint
        latest_path = self.models_dir / f"{self.experiment_name}_latest.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path, weights_only=False)
            return checkpoint.get("config", {})
        
        raise FileNotFoundError(f"No config found for {self.experiment_name}")
    
    def load_history(self) -> List[Dict]:
        """Load training history."""
        history_path = self.history_dir / f"{self.experiment_name}_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                return json.load(f)
        return []
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for path in sorted(self.models_dir.glob(f"{self.experiment_name}_round_*.pt")):
            try:
                # Load just metadata without full checkpoint
                checkpoint = torch.load(path, weights_only=False)
                checkpoints.append({
                    "path": str(path),
                    "round_id": checkpoint.get("round_id"),
                    "timestamp": checkpoint.get("metadata", {}).get("timestamp"),
                    "metrics": checkpoint.get("metrics", {}),
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {path}: {e}")
        
        return checkpoints
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint if exists."""
        best_path = self.models_dir / f"{self.experiment_name}_best.pt"
        return str(best_path) if best_path.exists() else None
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint if exists."""
        latest_path = self.models_dir / f"{self.experiment_name}_latest.pt"
        return str(latest_path) if latest_path.exists() else None
    
    def _cleanup_old_checkpoints(self):
        """Remove old round checkpoints, keeping only max_checkpoints."""
        checkpoints = sorted(
            self.models_dir.glob(f"{self.experiment_name}_round_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        
        # Keep best and latest, remove old round checkpoints
        to_remove = checkpoints[:-self.max_checkpoints] if len(checkpoints) > self.max_checkpoints else []
        
        for path in to_remove:
            path.unlink()
            logger.debug(f"Removed old checkpoint: {path}")
    
    def cleanup_all(self):
        """Remove all checkpoints for this experiment."""
        for pattern in [
            f"{self.experiment_name}_*.pt",
            f"{self.experiment_name}_*.yaml",
            f"{self.experiment_name}_*.json",
        ]:
            for path in self.checkpoint_dir.rglob(pattern):
                path.unlink()


# ===================== Convenience Functions =====================


def create_experiment_checkpoint(
    output_dir: Union[str, Path],
    experiment_name: str,
    config: Union[Dict, Any],
) -> ExperimentCheckpoint:
    """Create a new experiment checkpoint manager and save initial config.
    
    Args:
        output_dir: Output directory
        experiment_name: Experiment name
        config: Initial configuration
    
    Returns:
        ExperimentCheckpoint instance
    """
    checkpoint = ExperimentCheckpoint(output_dir, experiment_name)
    
    # Save config
    config_path = checkpoint.configs_dir / f"{experiment_name}_config.yaml"
    save_config(config, config_path)
    
    return checkpoint


def resume_experiment(
    checkpoint_dir: Union[str, Path],
    experiment_name: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    load_best: bool = False,
) -> Dict[str, Any]:
    """Resume an experiment from checkpoint.
    
    Args:
        checkpoint_dir: Checkpoint directory
        experiment_name: Experiment name
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        load_best: Load best checkpoint instead of latest
    
    Returns:
        Dict with round_id, config, history, etc.
    """
    checkpoint = ExperimentCheckpoint(checkpoint_dir, experiment_name)
    
    return checkpoint.load_full(
        model=model,
        load_best=load_best,
        load_latest=not load_best,
        optimizer=optimizer,
        restore_random_states=True,
    )


# ===================== Schema Validation =====================


@dataclass
class ConfigSchema:
    """Schema definition for config validation."""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_validators: Dict[str, callable] = field(default_factory=dict)


def validate_config(
    config: Dict[str, Any],
    schema: ConfigSchema,
) -> List[str]:
    """Validate config against schema.
    
    Args:
        config: Config dict to validate
        schema: Schema to validate against
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field_name in schema.required_fields:
        if field_name not in config:
            errors.append(f"Missing required field: {field_name}")
    
    # Check types
    for field_name, expected_type in schema.field_types.items():
        if field_name in config:
            value = config[field_name]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Field '{field_name}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
    
    # Run custom validators
    for field_name, validator in schema.field_validators.items():
        if field_name in config:
            try:
                if not validator(config[field_name]):
                    errors.append(f"Field '{field_name}' failed validation")
            except Exception as e:
                errors.append(f"Field '{field_name}' validation error: {e}")
    
    return errors


# ===================== Federated Learning Config Schema =====================


FL_CONFIG_SCHEMA = ConfigSchema(
    required_fields=[
        "num_clients",
        "num_rounds",
    ],
    optional_fields=[
        "participation_fraction",
        "local_epochs",
        "batch_size",
        "lr",
        "heterogeneity_mode",
        "seed",
    ],
    field_types={
        "num_clients": int,
        "num_rounds": int,
        "participation_fraction": float,
        "local_epochs": int,
        "batch_size": int,
        "lr": float,
        "seed": int,
    },
    field_validators={
        "num_clients": lambda x: x > 0,
        "num_rounds": lambda x: x > 0,
        "participation_fraction": lambda x: 0 < x <= 1,
        "local_epochs": lambda x: x > 0,
        "batch_size": lambda x: x > 0,
        "lr": lambda x: x > 0,
    },
)
