"""
Reproducibility utilities for federated learning experiments.

This module provides comprehensive tools for ensuring deterministic and
reproducible experiments:

1. Global seed management (Python, NumPy, PyTorch, CUDA)
2. Reproducibility context managers
3. Environment verification and logging
4. Reproducibility checks and validation
5. Hash-based experiment fingerprinting
"""

import hashlib
import json
import os
import platform
import random
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# ==================== Data Classes ====================


@dataclass
class RandomState:
    """Container for all random states."""
    
    python_state: Optional[tuple] = None
    numpy_state: Optional[Dict[str, Any]] = None
    torch_state: Optional[bytes] = None  # Serialized tensor
    torch_cuda_state: Optional[List[bytes]] = None  # Per-device states
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.python_state is not None:
            result["python"] = self.python_state
        if self.numpy_state is not None:
            result["numpy"] = self.numpy_state
        if self.torch_state is not None:
            result["torch"] = self.torch_state.hex() if isinstance(self.torch_state, bytes) else self.torch_state
        if self.torch_cuda_state is not None:
            result["torch_cuda"] = [s.hex() if isinstance(s, bytes) else s for s in self.torch_cuda_state]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RandomState":
        """Create from dictionary."""
        python_state = None
        if "python" in data:
            # Python random state must be tuple with nested tuple
            # Format: (version, tuple_of_ints, gauss_next)
            ps = data["python"]
            if isinstance(ps, list):
                # Convert nested lists back to tuples
                python_state = (ps[0], tuple(ps[1]), ps[2])
            else:
                python_state = ps
        
        return cls(
            python_state=python_state,
            numpy_state=data.get("numpy"),
            torch_state=bytes.fromhex(data["torch"]) if "torch" in data and isinstance(data["torch"], str) else data.get("torch"),
            torch_cuda_state=[bytes.fromhex(s) if isinstance(s, str) else s for s in data["torch_cuda"]] if "torch_cuda" in data else None,
        )


@dataclass
class EnvironmentInfo:
    """System and library environment information for reproducibility tracking."""
    
    # System info
    platform: str = ""
    platform_release: str = ""
    platform_version: str = ""
    architecture: str = ""
    hostname: str = ""
    processor: str = ""
    
    # Python info
    python_version: str = ""
    python_implementation: str = ""
    python_executable: str = ""
    
    # Library versions
    numpy_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    
    # CUDA info
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_devices: List[str] = field(default_factory=list)
    
    # Timestamp
    captured_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Capture current environment information."""
        info = cls(
            platform=platform.system(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            architecture=platform.machine(),
            hostname=platform.node(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            python_executable=sys.executable,
            numpy_version=np.__version__,
            captured_at=datetime.now().isoformat(),
        )
        
        if TORCH_AVAILABLE:
            info.torch_version = torch.__version__
            info.cuda_available = torch.cuda.is_available()
            
            if info.cuda_available:
                info.cuda_device_count = torch.cuda.device_count()
                info.cuda_devices = [
                    torch.cuda.get_device_name(i) 
                    for i in range(info.cuda_device_count)
                ]
                info.cuda_version = torch.version.cuda or ""
                info.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
        
        return info


@dataclass
class ReproducibilityReport:
    """Report on reproducibility settings and verification."""
    
    seed: Optional[int] = None
    deterministic_mode: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    environment: Optional[EnvironmentInfo] = None
    random_states_saved: bool = False
    config_hash: str = ""
    warnings: List[str] = field(default_factory=list)
    is_reproducible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "deterministic_mode": self.deterministic_mode,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "environment": self.environment.to_dict() if self.environment else None,
            "random_states_saved": self.random_states_saved,
            "config_hash": self.config_hash,
            "warnings": self.warnings,
            "is_reproducible": self.is_reproducible,
        }
    
    def add_warning(self, warning: str):
        """Add a reproducibility warning."""
        self.warnings.append(warning)
        self.is_reproducible = False


# ==================== Seed Management ====================


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all libraries for reproducibility.
    
    Args:
        seed: The seed value to use
        deterministic: If True, also enable deterministic mode for PyTorch/CUDA
    
    Example:
        >>> set_seed(42)  # All random operations now reproducible
        >>> set_seed(42, deterministic=True)  # Full deterministic mode
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            if deterministic:
                # Enable deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # PyTorch 1.8+ deterministic mode
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except RuntimeError:
                        # Some operations don't support deterministic mode
                        warnings.warn(
                            "Could not enable full deterministic algorithms. "
                            "Some operations may be non-deterministic."
                        )


def set_client_seed(base_seed: int, client_id: int) -> int:
    """
    Set seed for a specific client based on base seed.
    
    This ensures each client has a unique but reproducible seed.
    
    Args:
        base_seed: The experiment's base seed
        client_id: The client's identifier (int or will be hashed)
    
    Returns:
        The client-specific seed that was set
    """
    # Create unique seed for client
    client_seed = base_seed + client_id
    set_seed(client_seed, deterministic=False)  # Don't override CUDA determinism
    return client_seed


def set_round_seed(base_seed: int, round_id: int, client_id: Optional[int] = None) -> int:
    """
    Set seed for a specific round (and optionally client).
    
    Useful for ensuring each round has different but reproducible randomness.
    
    Args:
        base_seed: The experiment's base seed
        round_id: The current round number
        client_id: Optional client identifier
    
    Returns:
        The round-specific seed that was set
    """
    if client_id is not None:
        round_seed = base_seed + round_id * 10000 + client_id
    else:
        round_seed = base_seed + round_id * 10000
    
    set_seed(round_seed, deterministic=False)
    return round_seed


# ==================== Random State Management ====================


def get_random_states() -> RandomState:
    """
    Capture all random states for later restoration.
    
    Returns:
        RandomState object containing all captured states
    """
    state = RandomState()
    
    # Python random
    state.python_state = random.getstate()
    
    # NumPy
    np_state = np.random.get_state()
    # Convert to serializable format
    state.numpy_state = {
        "state_type": np_state[0],
        "state_array": np_state[1].tolist(),
        "pos": np_state[2],
        "has_gauss": np_state[3],
        "cached_gauss": np_state[4],
    }
    
    # PyTorch
    if TORCH_AVAILABLE:
        state.torch_state = torch.get_rng_state().numpy().tobytes()
        
        if torch.cuda.is_available():
            state.torch_cuda_state = [
                torch.cuda.get_rng_state(i).numpy().tobytes()
                for i in range(torch.cuda.device_count())
            ]
    
    return state


def set_random_states(state: RandomState) -> None:
    """
    Restore random states from a captured RandomState object.
    
    Args:
        state: RandomState object to restore from
    """
    # Python random
    if state.python_state is not None:
        random.setstate(state.python_state)
    
    # NumPy
    if state.numpy_state is not None:
        np_state = (
            state.numpy_state["state_type"],
            np.array(state.numpy_state["state_array"], dtype=np.uint32),
            state.numpy_state["pos"],
            state.numpy_state["has_gauss"],
            state.numpy_state["cached_gauss"],
        )
        np.random.set_state(np_state)
    
    # PyTorch
    if TORCH_AVAILABLE:
        if state.torch_state is not None:
            torch_state = torch.from_numpy(
                np.frombuffer(state.torch_state, dtype=np.uint8)
            )
            torch.set_rng_state(torch_state)
        
        if state.torch_cuda_state is not None and torch.cuda.is_available():
            for i, cuda_state in enumerate(state.torch_cuda_state):
                if i < torch.cuda.device_count():
                    cuda_tensor = torch.from_numpy(
                        np.frombuffer(cuda_state, dtype=np.uint8)
                    )
                    torch.cuda.set_rng_state(cuda_tensor, device=i)


def save_random_states(path: Union[str, Path]) -> None:
    """
    Save random states to a file.
    
    Args:
        path: Path to save the states to
    """
    state = get_random_states()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def load_random_states(path: Union[str, Path]) -> RandomState:
    """
    Load random states from a file.
    
    Args:
        path: Path to load states from
    
    Returns:
        The loaded RandomState object
    """
    with open(path, "r") as f:
        data = json.load(f)
    return RandomState.from_dict(data)


def restore_random_states(path: Union[str, Path]) -> None:
    """
    Load and restore random states from a file.
    
    Args:
        path: Path to load states from
    """
    state = load_random_states(path)
    set_random_states(state)


# ==================== Context Managers ====================


@contextmanager
def reproducible_context(seed: int, deterministic: bool = True):
    """
    Context manager for reproducible operations.
    
    Saves current random states, sets the seed, and restores states on exit.
    
    Args:
        seed: Seed to use within the context
        deterministic: Whether to enable deterministic mode
    
    Example:
        >>> with reproducible_context(42):
        ...     result = some_random_operation()
        >>> # Original random states restored
    """
    # Save current states
    saved_states = get_random_states()
    saved_hash_seed = os.environ.get("PYTHONHASHSEED")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        saved_cudnn_deterministic = torch.backends.cudnn.deterministic
        saved_cudnn_benchmark = torch.backends.cudnn.benchmark
    
    try:
        # Set reproducible seed
        set_seed(seed, deterministic=deterministic)
        yield
    finally:
        # Restore original states
        set_random_states(saved_states)
        
        if saved_hash_seed is not None:
            os.environ["PYTHONHASHSEED"] = saved_hash_seed
        elif "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = saved_cudnn_deterministic
            torch.backends.cudnn.benchmark = saved_cudnn_benchmark


@contextmanager
def isolated_random_state():
    """
    Context manager that isolates random operations.
    
    Any random operations within this context won't affect the global state.
    
    Example:
        >>> np.random.seed(42)
        >>> with isolated_random_state():
        ...     _ = np.random.rand(100)  # Doesn't affect global state
        >>> np.random.rand()  # Same as if isolation didn't happen
    """
    saved_states = get_random_states()
    try:
        yield
    finally:
        set_random_states(saved_states)


@contextmanager
def deterministic_mode(enabled: bool = True):
    """
    Context manager to temporarily enable/disable deterministic mode.
    
    Args:
        enabled: Whether to enable deterministic mode
    
    Example:
        >>> with deterministic_mode(True):
        ...     # All CUDA operations are deterministic here
        ...     output = model(input)
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        yield
        return
    
    # Save current settings
    saved_deterministic = torch.backends.cudnn.deterministic
    saved_benchmark = torch.backends.cudnn.benchmark
    
    try:
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(enabled)
            except RuntimeError:
                pass
        
        yield
    finally:
        torch.backends.cudnn.deterministic = saved_deterministic
        torch.backends.cudnn.benchmark = saved_benchmark


# ==================== Verification and Checks ====================


def verify_reproducibility(
    seed: int,
    func: callable,
    num_runs: int = 3,
    compare_fn: Optional[callable] = None,
) -> Tuple[bool, List[Any]]:
    """
    Verify that a function produces reproducible results.
    
    Args:
        seed: Seed to use
        func: Function to test (should take no arguments)
        num_runs: Number of times to run the function
        compare_fn: Optional comparison function (default: np.allclose or ==)
    
    Returns:
        Tuple of (is_reproducible, list_of_results)
    
    Example:
        >>> def random_op():
        ...     return np.random.rand(10)
        >>> is_repro, results = verify_reproducibility(42, random_op)
        >>> print(is_repro)  # True
    """
    results = []
    
    for _ in range(num_runs):
        with reproducible_context(seed):
            result = func()
            results.append(result)
    
    # Compare results
    if compare_fn is None:
        def compare_fn(a, b):
            if isinstance(a, np.ndarray):
                return np.allclose(a, b)
            elif TORCH_AVAILABLE and isinstance(a, torch.Tensor):
                return torch.allclose(a, b)
            else:
                return a == b
    
    is_reproducible = all(
        compare_fn(results[0], results[i]) for i in range(1, num_runs)
    )
    
    return is_reproducible, results


def check_reproducibility_settings() -> ReproducibilityReport:
    """
    Check current reproducibility settings and return a report.
    
    Returns:
        ReproducibilityReport with current settings and warnings
    """
    report = ReproducibilityReport()
    report.environment = EnvironmentInfo.capture()
    
    # Check PYTHONHASHSEED
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        report.add_warning("PYTHONHASHSEED not set - dict ordering may vary")
    else:
        try:
            report.seed = int(hash_seed)
        except ValueError:
            pass
    
    # Check PyTorch settings
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            report.cudnn_deterministic = torch.backends.cudnn.deterministic
            report.cudnn_benchmark = torch.backends.cudnn.benchmark
            
            if not report.cudnn_deterministic:
                report.add_warning("cudnn.deterministic is False - CUDA ops may vary")
            
            if report.cudnn_benchmark:
                report.add_warning("cudnn.benchmark is True - may cause non-determinism")
            
            # Check deterministic algorithms mode
            if hasattr(torch, 'are_deterministic_algorithms_enabled'):
                report.deterministic_mode = torch.are_deterministic_algorithms_enabled()
                if not report.deterministic_mode:
                    report.add_warning("Deterministic algorithms not enabled")
    
    return report


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get a dictionary of reproducibility-relevant information.
    
    Returns:
        Dictionary with environment and settings info
    """
    report = check_reproducibility_settings()
    return report.to_dict()


# ==================== Experiment Fingerprinting ====================


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of a configuration dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Hex string hash of the config
    """
    # Sort keys for deterministic ordering
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_experiment_fingerprint(
    config: Dict[str, Any],
    seed: int,
    include_environment: bool = False,
) -> str:
    """
    Compute a fingerprint for an experiment setup.
    
    Args:
        config: Experiment configuration
        seed: Random seed used
        include_environment: Whether to include environment info in fingerprint
    
    Returns:
        Hex string fingerprint
    """
    fingerprint_data = {
        "config_hash": compute_config_hash(config),
        "seed": seed,
    }
    
    if include_environment:
        env = EnvironmentInfo.capture()
        fingerprint_data["python_version"] = env.python_version
        fingerprint_data["numpy_version"] = env.numpy_version
        if TORCH_AVAILABLE:
            fingerprint_data["torch_version"] = env.torch_version
    
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


# ==================== Worker/DataLoader Seeding ====================


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.
    
    Ensures each worker has a unique but reproducible seed.
    
    Usage:
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    # Get base seed from numpy's global state
    seed = np.random.get_state()[1][0] + worker_id
    
    np.random.seed(seed)
    random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def create_worker_init_fn(base_seed: int) -> callable:
    """
    Create a worker init function with a specific base seed.
    
    Args:
        base_seed: Base seed for workers
    
    Returns:
        Worker init function for DataLoader
    
    Usage:
        >>> init_fn = create_worker_init_fn(42)
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=init_fn)
    """
    def init_fn(worker_id: int) -> None:
        seed = base_seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
    
    return init_fn


def get_generator(seed: int) -> Optional["torch.Generator"]:
    """
    Create a PyTorch Generator with a specific seed.
    
    Useful for reproducible data loading.
    
    Args:
        seed: Seed for the generator
    
    Returns:
        PyTorch Generator or None if torch unavailable
    """
    if not TORCH_AVAILABLE:
        return None
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ==================== Reproducibility Logger ====================


class ReproducibilityLogger:
    """
    Logger for tracking reproducibility information throughout an experiment.
    
    Usage:
        >>> repro_logger = ReproducibilityLogger(experiment_dir, seed=42)
        >>> repro_logger.log_initial_state()
        >>> # ... run experiment ...
        >>> repro_logger.log_checkpoint(round_id=5)
        >>> repro_logger.save_report()
    """
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        seed: Optional[int] = None,
        auto_set_seed: bool = True,
    ):
        """
        Initialize the reproducibility logger.
        
        Args:
            experiment_dir: Directory to save reproducibility info
            seed: Random seed for the experiment
            auto_set_seed: Whether to automatically set the seed
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        self.checkpoints: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        
        # Capture initial environment
        self.environment = EnvironmentInfo.capture()
        
        if auto_set_seed and seed is not None:
            set_seed(seed, deterministic=True)
        
        # Log initialization
        self._log_event("initialized", {"seed": seed, "auto_set_seed": auto_set_seed})
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event with timestamp."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        })
    
    def log_initial_state(self) -> None:
        """Log the initial random state."""
        state = get_random_states()
        state_path = self.experiment_dir / "initial_random_state.json"
        
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        
        self._log_event("initial_state_saved", {"path": str(state_path)})
    
    def log_checkpoint(
        self,
        round_id: int,
        save_state: bool = True,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a reproducibility checkpoint.
        
        Args:
            round_id: Current round number
            save_state: Whether to save the random state
            additional_info: Additional info to log
        """
        checkpoint = {
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        if save_state:
            state = get_random_states()
            state_path = self.experiment_dir / f"random_state_round_{round_id}.json"
            with open(state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            checkpoint["state_path"] = str(state_path)
        
        if additional_info:
            checkpoint["additional_info"] = additional_info
        
        self.checkpoints.append(checkpoint)
        self._log_event("checkpoint", checkpoint)
    
    def verify_state(self, round_id: int) -> bool:
        """
        Verify that the current state matches a saved checkpoint.
        
        Args:
            round_id: Round ID to verify against
        
        Returns:
            True if states match
        """
        state_path = self.experiment_dir / f"random_state_round_{round_id}.json"
        if not state_path.exists():
            return False
        
        saved_state = load_random_states(state_path)
        current_state = get_random_states()
        
        # Compare numpy states (most reliable check)
        if saved_state.numpy_state and current_state.numpy_state:
            return saved_state.numpy_state == current_state.numpy_state
        
        return False
    
    def restore_checkpoint(self, round_id: int) -> None:
        """
        Restore random state from a checkpoint.
        
        Args:
            round_id: Round ID to restore from
        """
        state_path = self.experiment_dir / f"random_state_round_{round_id}.json"
        restore_random_states(state_path)
        self._log_event("state_restored", {"round_id": round_id})
    
    def get_report(self) -> ReproducibilityReport:
        """Generate a reproducibility report."""
        report = check_reproducibility_settings()
        report.seed = self.seed
        report.environment = self.environment
        report.random_states_saved = len(self.checkpoints) > 0
        
        return report
    
    def save_report(self) -> str:
        """
        Save the reproducibility report to a file.
        
        Returns:
            Path to the saved report
        """
        report = self.get_report()
        report_dict = report.to_dict()
        report_dict["checkpoints"] = self.checkpoints
        report_dict["events"] = self.events
        
        report_path = self.experiment_dir / "reproducibility_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        return str(report_path)
    
    def save_environment(self) -> str:
        """
        Save environment information to a file.
        
        Returns:
            Path to the saved environment file
        """
        env_path = self.experiment_dir / "environment.json"
        with open(env_path, "w") as f:
            json.dump(self.environment.to_dict(), f, indent=2)
        return str(env_path)


# ==================== Convenience Functions ====================


def make_reproducible(seed: int = 42, deterministic: bool = True) -> ReproducibilityReport:
    """
    One-liner to make everything reproducible.
    
    Args:
        seed: Seed to use
        deterministic: Whether to enable deterministic mode
    
    Returns:
        ReproducibilityReport with current settings
    
    Example:
        >>> report = make_reproducible(42)
        >>> print(f"Reproducible: {report.is_reproducible}")
    """
    set_seed(seed, deterministic=deterministic)
    report = check_reproducibility_settings()
    report.seed = seed
    return report


def quick_verify(func: callable, seed: int = 42) -> bool:
    """
    Quick check if a function is reproducible.
    
    Args:
        func: Function to verify
        seed: Seed to use
    
    Returns:
        True if function is reproducible
    
    Example:
        >>> is_repro = quick_verify(lambda: np.random.rand(10))
        >>> print(is_repro)  # True
    """
    is_repro, _ = verify_reproducibility(seed, func, num_runs=2)
    return is_repro


__all__ = [
    # Data classes
    "RandomState",
    "EnvironmentInfo",
    "ReproducibilityReport",
    # Seed management
    "set_seed",
    "set_client_seed",
    "set_round_seed",
    # State management
    "get_random_states",
    "set_random_states",
    "save_random_states",
    "load_random_states",
    "restore_random_states",
    # Context managers
    "reproducible_context",
    "isolated_random_state",
    "deterministic_mode",
    # Verification
    "verify_reproducibility",
    "check_reproducibility_settings",
    "get_reproducibility_info",
    # Fingerprinting
    "compute_config_hash",
    "compute_experiment_fingerprint",
    # Worker seeding
    "worker_init_fn",
    "create_worker_init_fn",
    "get_generator",
    # Logger
    "ReproducibilityLogger",
    # Convenience
    "make_reproducible",
    "quick_verify",
    # Constants
    "TORCH_AVAILABLE",
]
