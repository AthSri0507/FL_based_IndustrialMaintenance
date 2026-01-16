# Utils package for federated learning

"""
Utility modules for configuration persistence, checkpointing, and experiment management.

Modules:
- persistence: Config persistence (YAML/JSON), checkpointing, experiment resume
- experiment_logging: Per-round metrics logging, experiment tracking, CSV/TensorBoard export
- reproducibility: Deterministic seeds, reproducibility checks, environment tracking
"""

from .persistence import (
    ConfigManager,
    ExperimentCheckpoint,
    ConfigSchema,
    save_config,
    load_config,
    dataclass_to_dict,
    dict_to_dataclass,
    validate_config,
    create_experiment_checkpoint,
    resume_experiment,
    FL_CONFIG_SCHEMA,
)

from .experiment_logging import (
    ExperimentLogger,
    MetricsAggregator,
    ExperimentComparison,
    ExperimentSummary,
    RoundLogEntry,
    ClientRoundMetrics,
    create_experiment_logger,
    load_experiment_metrics,
    get_metric_series,
)

from .reproducibility import (
    # Data classes
    RandomState,
    EnvironmentInfo,
    ReproducibilityReport,
    # Seed management
    set_seed,
    set_client_seed,
    set_round_seed,
    # State management
    get_random_states,
    set_random_states,
    save_random_states,
    load_random_states,
    restore_random_states,
    # Context managers
    reproducible_context,
    isolated_random_state,
    deterministic_mode,
    # Verification
    verify_reproducibility,
    check_reproducibility_settings,
    get_reproducibility_info,
    # Fingerprinting
    compute_config_hash,
    compute_experiment_fingerprint,
    # Worker seeding
    worker_init_fn,
    create_worker_init_fn,
    get_generator,
    # Logger
    ReproducibilityLogger,
    # Convenience
    make_reproducible,
    quick_verify,
)

__all__ = [
    # Persistence
    "ConfigManager",
    "ExperimentCheckpoint",
    "ConfigSchema",
    "save_config",
    "load_config",
    "dataclass_to_dict",
    "dict_to_dataclass",
    "validate_config",
    "create_experiment_checkpoint",
    "resume_experiment",
    "FL_CONFIG_SCHEMA",
    # Experiment Logging
    "ExperimentLogger",
    "MetricsAggregator",
    "ExperimentComparison",
    "ExperimentSummary",
    "RoundLogEntry",
    "ClientRoundMetrics",
    "create_experiment_logger",
    "load_experiment_metrics",
    "get_metric_series",
    # Reproducibility
    "RandomState",
    "EnvironmentInfo",
    "ReproducibilityReport",
    "set_seed",
    "set_client_seed",
    "set_round_seed",
    "get_random_states",
    "set_random_states",
    "save_random_states",
    "load_random_states",
    "restore_random_states",
    "reproducible_context",
    "isolated_random_state",
    "deterministic_mode",
    "verify_reproducibility",
    "check_reproducibility_settings",
    "get_reproducibility_info",
    "compute_config_hash",
    "compute_experiment_fingerprint",
    "worker_init_fn",
    "create_worker_init_fn",
    "get_generator",
    "ReproducibilityLogger",
    "make_reproducible",
    "quick_verify",
]
