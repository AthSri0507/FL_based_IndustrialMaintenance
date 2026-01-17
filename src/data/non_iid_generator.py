"""Non-IID Hard Data Generator for Federated Learning Stress Testing.

This module provides a strongly heterogeneous and non-IID data generation mode
for stress-testing federated learning algorithms. It implements four dimensions
of heterogeneity:

1. Label Skew (RUL Distribution): Each client sees a different RUL distribution
2. Feature Skew (Sensor Behavior): Client-specific sensor bias and noise
3. Quantity Skew (Data Imbalance): Different dataset sizes per client
4. Concept Drift: Temporal drift for long experiments

IMPORTANT: This module is isolated from the existing data generation code.
The existing "clean" data generator remains COMPLETELY UNCHANGED.

Usage:
    if data_profile == "clean":
        generate_clean_data()        # existing behavior (UNTOUCHED)
    elif data_profile == "non_iid_hard":
        generate_non_iid_hard_data()  # this module
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


# =============================================================================
# NON-IID HARD MODE CONFIGURATION
# =============================================================================

# -- Label Skew Configuration --
# Each client sees a different RUL distribution
# Client 0 → mostly low RUL (near failure): [0, 30]
# Client 1 → mid RUL: [30, 60]
# Client 2 → high RUL (healthy): [60, 100]
# Client 3 → bimodal: half from [0, 20], half from [80, 100]
# Client 4+ → uniform: [0, 100]

# -- Feature Skew Configuration --
# Client-specific noise levels (sensor quality differences)
CLIENT_NOISE: Dict[int, float] = {
    0: 0.1,   # Low noise (good sensor)
    1: 0.5,   # Medium-high noise
    2: 0.2,   # Low-medium noise
    3: 1.0,   # High noise (poor sensor)
    4: 0.05,  # Very low noise (excellent sensor)
}

# Client-specific bias levels (calibration drift)
CLIENT_BIAS: Dict[int, float] = {
    0: 0.0,   # No bias (well calibrated)
    1: 2.0,   # Positive bias
    2: -2.0,  # Negative bias
    3: 5.0,   # Large positive bias
    4: 0.0,   # No bias
}

# -- Quantity Skew Configuration --
# Client dataset sizes (strong participation imbalance)
CLIENT_DATA_SIZES: Dict[int, int] = {
    0: 800,   # Large dataset
    1: 200,   # Medium dataset
    2: 150,   # Small dataset
    3: 50,    # Very small dataset
    4: 300,   # Medium dataset
}


# =============================================================================
# LABEL SKEW FUNCTIONS
# =============================================================================

def sample_rul_for_client(
    client_id: int,
    n: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample RUL values for a specific client with label skew.
    
    Each client lives in a different RUL "world":
    - Client 0: Low RUL (near failure) [0, 30]
    - Client 1: Mid RUL [30, 60]
    - Client 2: High RUL (healthy) [60, 100]
    - Client 3: Bimodal (half low, half high)
    - Client 4+: Uniform [0, 100]
    
    Args:
        client_id: Client identifier (0-indexed)
        n: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of RUL values of shape (n,)
    """
    rng = np.random.RandomState(seed)
    
    if client_id == 0:
        # Low RUL: near failure
        return rng.uniform(0, 30, n).astype(np.float32)
    elif client_id == 1:
        # Mid RUL
        return rng.uniform(30, 60, n).astype(np.float32)
    elif client_id == 2:
        # High RUL: healthy
        return rng.uniform(60, 100, n).astype(np.float32)
    elif client_id == 3:
        # Bimodal: half from [0, 20], half from [80, 100]
        n_low = n // 2
        n_high = n - n_low
        low_rul = rng.uniform(0, 20, n_low)
        high_rul = rng.uniform(80, 100, n_high)
        combined = np.concatenate([low_rul, high_rul])
        rng.shuffle(combined)  # Mix the bimodal samples
        return combined.astype(np.float32)
    else:
        # Uniform for any additional clients
        return rng.uniform(0, 100, n).astype(np.float32)


def sample_labels_for_client(
    client_id: int,
    n: int,
    num_classes: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample classification labels for a specific client with label skew.
    
    Each client has a different class distribution:
    - Client 0: Mostly class 0 (90%)
    - Client 1: Mostly class 1 (90%)
    - Client 2: Balanced
    - Client 3: Only one class
    - Client 4+: Uniform random
    
    Args:
        client_id: Client identifier (0-indexed)
        n: Number of samples to generate
        num_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        Array of class labels of shape (n,)
    """
    rng = np.random.RandomState(seed)
    
    if client_id == 0:
        # Mostly class 0
        probs = [0.9] + [0.1 / (num_classes - 1)] * (num_classes - 1)
        return rng.choice(num_classes, size=n, p=probs).astype(np.int64)
    elif client_id == 1:
        # Mostly class 1 (or last class if binary)
        target_class = min(1, num_classes - 1)
        probs = [0.1 / (num_classes - 1)] * num_classes
        probs[target_class] = 0.9
        probs = [p / sum(probs) for p in probs]  # Normalize
        return rng.choice(num_classes, size=n, p=probs).astype(np.int64)
    elif client_id == 2:
        # Balanced
        return rng.choice(num_classes, size=n).astype(np.int64)
    elif client_id == 3:
        # Only one class
        single_class = client_id % num_classes
        return np.full(n, single_class, dtype=np.int64)
    else:
        # Uniform random
        return rng.choice(num_classes, size=n).astype(np.int64)


# =============================================================================
# FEATURE SKEW FUNCTIONS
# =============================================================================

def get_client_noise_level(client_id: int) -> float:
    """Get the noise level for a specific client.
    
    Args:
        client_id: Client identifier (0-indexed)
        
    Returns:
        Noise standard deviation for this client
    """
    return CLIENT_NOISE.get(client_id, 0.15)  # Default noise for unknown clients


def get_client_bias(client_id: int) -> float:
    """Get the sensor bias for a specific client.
    
    Args:
        client_id: Client identifier (0-indexed)
        
    Returns:
        Bias offset for this client
    """
    return CLIENT_BIAS.get(client_id, 0.0)  # Default no bias for unknown clients


def apply_feature_skew(
    x: np.ndarray,
    client_id: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply client-specific sensor bias and noise to features.
    
    This simulates:
    - Calibration drift between sensors at different sites
    - Sensor quality differences (noise levels)
    
    Args:
        x: Feature array of shape (N, T, C) or (N, C)
        client_id: Client identifier (0-indexed)
        seed: Random seed for reproducibility
        
    Returns:
        Modified feature array with same shape as input
    """
    rng = np.random.RandomState(seed)
    
    bias = get_client_bias(client_id)
    noise_std = get_client_noise_level(client_id)
    
    # Apply bias (calibration drift)
    x_skewed = x + bias
    
    # Apply noise (sensor quality)
    noise = rng.randn(*x.shape).astype(x.dtype) * noise_std
    x_skewed = x_skewed + noise
    
    return x_skewed.astype(np.float32)


# =============================================================================
# QUANTITY SKEW FUNCTIONS
# =============================================================================

def get_client_sample_count(client_id: int, num_clients: int = 5) -> int:
    """Get the number of samples for a specific client.
    
    Creates strong participation imbalance to make FedAvg harder.
    
    Args:
        client_id: Client identifier (0-indexed)
        num_clients: Total number of clients
        
    Returns:
        Number of samples for this client
    """
    if client_id in CLIENT_DATA_SIZES:
        return CLIENT_DATA_SIZES[client_id]
    
    # For additional clients beyond defined ones, use a pattern
    # Alternating between small and medium datasets
    if client_id % 2 == 0:
        return 100  # Small
    else:
        return 250  # Medium


# =============================================================================
# CONCEPT DRIFT FUNCTIONS
# =============================================================================

def apply_concept_drift(
    x: np.ndarray,
    client_id: int,
    round_id: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply temporal drift for long experiments.
    
    Simulates sensor aging or machine changes at specific sites.
    Currently applies drift to Client 2 after round 8.
    
    Args:
        x: Feature array of shape (N, T, C) or (N, C)
        client_id: Client identifier (0-indexed)
        round_id: Current training round (1-indexed)
        seed: Random seed for reproducibility (unused but accepted for API consistency)
        
    Returns:
        Modified feature array with same shape as input
    """
    # Only apply drift to client 2 after round 8
    if round_id > 8 and client_id == 2:
        # Simulate sensor aging: scaling + offset
        x_drifted = x * 1.2 + 3.0
        return x_drifted.astype(np.float32)
    
    return x


# =============================================================================
# SYNTHETIC DATA GENERATION (BASE)
# =============================================================================

def generate_synthetic_features(
    n_samples: int,
    seq_length: int,
    num_channels: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate base synthetic time-series sensor data.
    
    Creates realistic-looking sensor data with sinusoidal patterns,
    phase variations, and natural degradation trends.
    
    Args:
        n_samples: Number of samples to generate
        seq_length: Length of each time series
        num_channels: Number of sensor channels
        seed: Random seed for reproducibility
        
    Returns:
        Feature array of shape (n_samples, seq_length, num_channels)
    """
    rng = np.random.RandomState(seed)
    
    X = np.zeros((n_samples, seq_length, num_channels), dtype=np.float32)
    
    for i in range(n_samples):
        t = np.linspace(0, 4 * np.pi, seq_length)
        for c in range(num_channels):
            # Base sinusoidal pattern with channel-specific frequency
            freq = 0.5 + c * 0.2
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = 1.0 + rng.uniform(-0.3, 0.3)
            
            # Base signal
            signal = amplitude * np.sin(freq * t + phase)
            
            # Add small base noise
            base_noise = rng.randn(seq_length) * 0.05
            
            # Natural degradation trend for some samples
            degradation = 0.0
            if rng.rand() > 0.5:
                degradation = np.linspace(0, rng.uniform(0.3, 1.5), seq_length)
            
            X[i, :, c] = signal + base_noise + degradation
    
    return X


# =============================================================================
# MAIN NON-IID HARD DATA GENERATOR
# =============================================================================

def generate_non_iid_hard_data(
    num_clients: int = 5,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: Optional[int] = None,
    round_id: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate heterogeneous non-IID data for stress-testing federated learning.
    
    This is the main entry point for the non_iid_hard data profile.
    It applies all four dimensions of heterogeneity:
    
    1. Label Skew: Each client has a different RUL/class distribution
    2. Feature Skew: Client-specific sensor bias and noise
    3. Quantity Skew: Different dataset sizes per client
    4. Concept Drift: Temporal drift for long experiments (if round_id > 8)
    
    Args:
        num_clients: Number of clients to generate data for
        seq_length: Length of each time series
        num_channels: Number of sensor channels
        task: "rul" for regression or "classification" for classification
        num_classes: Number of classes (only used for classification task)
        seed: Random seed for reproducibility
        round_id: Current training round (for concept drift, default 0 = no drift)
        
    Returns:
        List of (X, y) tuples, one per client.
        X has shape (n_samples, seq_length, num_channels)
        y has shape (n_samples,)
    """
    client_data = []
    
    for client_id in range(num_clients):
        # Derive client-specific seed for reproducibility
        client_seed = seed + client_id if seed is not None else None
        
        # -- Quantity Skew --
        n_samples = get_client_sample_count(client_id, num_clients)
        
        # -- Generate base features --
        feature_seed = client_seed
        X = generate_synthetic_features(
            n_samples=n_samples,
            seq_length=seq_length,
            num_channels=num_channels,
            seed=feature_seed,
        )
        
        # -- Feature Skew --
        feature_skew_seed = client_seed + 1000 if client_seed is not None else None
        X = apply_feature_skew(X, client_id, seed=feature_skew_seed)
        
        # -- Concept Drift (if applicable) --
        drift_seed = client_seed + 2000 if client_seed is not None else None
        X = apply_concept_drift(X, client_id, round_id, seed=drift_seed)
        
        # -- Label Skew --
        label_seed = client_seed + 3000 if client_seed is not None else None
        if task == "rul":
            y = sample_rul_for_client(client_id, n_samples, seed=label_seed)
        else:
            y = sample_labels_for_client(
                client_id, n_samples, num_classes=num_classes, seed=label_seed
            )
        
        client_data.append((X, y))
    
    return client_data


def generate_non_iid_hard_centralized(
    num_clients: int = 5,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate centralized (merged) non-IID hard data.
    
    This produces a single dataset by merging all client data.
    Useful for centralized baseline comparisons.
    
    Args:
        num_clients: Number of clients whose data to merge
        seq_length: Length of each time series
        num_channels: Number of sensor channels
        task: "rul" for regression or "classification"
        num_classes: Number of classes (only for classification)
        seed: Random seed for reproducibility
        
    Returns:
        (X, y) tuple where:
        X has shape (total_samples, seq_length, num_channels)
        y has shape (total_samples,)
    """
    client_data = generate_non_iid_hard_data(
        num_clients=num_clients,
        seq_length=seq_length,
        num_channels=num_channels,
        task=task,
        num_classes=num_classes,
        seed=seed,
        round_id=0,  # No concept drift for centralized baseline
    )
    
    # Merge all client data
    all_X = np.concatenate([X for X, y in client_data], axis=0)
    all_y = np.concatenate([y for X, y in client_data], axis=0)
    
    return all_X, all_y


def get_data_profile_stats(client_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Compute statistics about the generated data for debugging/logging.
    
    Args:
        client_data: List of (X, y) tuples from generate_non_iid_hard_data
        
    Returns:
        Dictionary with per-client statistics
    """
    stats = {}
    
    for client_id, (X, y) in enumerate(client_data):
        client_stats = {
            "n_samples": len(X),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
            "X_mean": float(X.mean()),
            "X_std": float(X.std()),
            "feature_bias": get_client_bias(client_id),
            "feature_noise": get_client_noise_level(client_id),
        }
        stats[f"client_{client_id}"] = client_stats
    
    return stats


# =============================================================================
# NON-IID MILD MODE CONFIGURATION
# =============================================================================
# Moderate heterogeneity - between "clean" (IID) and "non_iid_hard" (extreme)
# 
# Key differences from non_iid_hard:
# - Overlapping (not disjoint) RUL ranges
# - Smaller noise/bias variations
# - Less extreme quantity imbalance
# - No concept drift

# -- Mild Label Skew Configuration --
# RUL ranges overlap significantly (unlike hard mode's disjoint ranges)
MILD_CLIENT_RUL_RANGES: Dict[int, Tuple[int, int]] = {
    0: (0, 70),    # Low-to-mid RUL
    1: (20, 90),   # Mid RUL
    2: (10, 80),   # Low-to-mid-high RUL
    3: (0, 100),   # Full range (baseline)
    4: (30, 100),  # Mid-to-high RUL
}

# -- Mild Feature Skew Configuration --
# Moderate noise levels (less extreme than hard mode)
MILD_CLIENT_NOISE: Dict[int, float] = {
    0: 0.05,   # Low noise
    1: 0.15,   # Medium noise
    2: 0.10,   # Low-medium noise
    3: 0.20,   # Medium-high noise
    4: 0.08,   # Low noise
}

# Moderate bias levels (less extreme than hard mode)
MILD_CLIENT_BIAS: Dict[int, float] = {
    0: 0.0,    # No bias
    1: 0.5,    # Small positive bias
    2: -0.5,   # Small negative bias
    3: 1.0,    # Moderate positive bias
    4: -1.0,   # Moderate negative bias
}

# -- Mild Quantity Skew Configuration --
# Moderate imbalance (less extreme than hard mode)
MILD_CLIENT_DATA_SIZES: Dict[int, int] = {
    0: 400,
    1: 300,
    2: 250,
    3: 200,
    4: 150,
}


def _sample_mild_rul_for_client(
    client_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample RUL values for a client using mild label skew.
    
    Unlike hard mode, RUL ranges overlap significantly.
    """
    rul_min, rul_max = MILD_CLIENT_RUL_RANGES.get(client_id, (0, 100))
    return rng.uniform(rul_min, rul_max, size=n_samples)


def _get_mild_client_noise(client_id: int) -> float:
    """Get noise level for mild profile."""
    return MILD_CLIENT_NOISE.get(client_id, 0.1)


def _get_mild_client_bias(client_id: int) -> float:
    """Get bias level for mild profile."""
    return MILD_CLIENT_BIAS.get(client_id, 0.0)


def _get_mild_client_sample_count(client_id: int) -> int:
    """Get sample count for mild profile."""
    return MILD_CLIENT_DATA_SIZES.get(client_id, 250)


def _apply_mild_feature_skew(
    X: np.ndarray,
    client_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply mild feature skew (noise + bias) to client data."""
    noise_level = _get_mild_client_noise(client_id)
    bias = _get_mild_client_bias(client_id)
    
    noise = rng.normal(0, noise_level, size=X.shape)
    X_skewed = X + noise + bias
    
    return X_skewed


def generate_non_iid_mild_data(
    num_clients: int = 5,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    num_classes: int = 2,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate non-IID mild heterogeneous data for FL stress testing.
    
    This creates a moderately heterogeneous dataset that sits between
    clean (IID) and non_iid_hard (extreme heterogeneity).
    
    Heterogeneity dimensions:
    - Label skew: Overlapping but shifted RUL ranges
    - Feature skew: Moderate noise/bias variations
    - Quantity skew: Moderate data size imbalance
    - NO concept drift (unlike hard mode)
    
    Args:
        num_clients: Number of clients to generate data for
        seq_length: Length of each time series sequence
        num_channels: Number of sensor channels
        task: Task type ("rul" or "classification")
        num_classes: Number of classes (only used for classification task)
        seed: Random seed for reproducibility
        
    Returns:
        List of (X, y) tuples, one per client
    """
    rng = np.random.default_rng(seed)
    client_data = []
    
    for client_id in range(num_clients):
        # Get client-specific sample count
        n_samples = _get_mild_client_sample_count(client_id)
        
        # Generate RUL values with mild label skew (overlapping ranges)
        y = _sample_mild_rul_for_client(client_id, n_samples, rng)
        
        # Generate base features (use client_id as part of seed for reproducibility)
        client_seed = seed + client_id if seed is not None else None
        X = generate_synthetic_features(
            n_samples=n_samples,
            seq_length=seq_length,
            num_channels=num_channels,
            seed=client_seed,
        )
        
        # Apply mild feature skew (smaller noise/bias)
        X = _apply_mild_feature_skew(X, client_id, rng)
        
        client_data.append((X, y))
    
    return client_data


def generate_non_iid_mild_centralized(
    num_clients: int = 5,
    seq_length: int = 100,
    num_channels: int = 14,
    task: str = "rul",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate centralized version of non-IID mild data.
    
    Generates data as if it came from multiple heterogeneous sources,
    then merges it for centralized training baseline.
    """
    client_data = generate_non_iid_mild_data(
        num_clients=num_clients,
        seq_length=seq_length,
        num_channels=num_channels,
        task=task,
        seed=seed,
    )
    
    all_X = np.concatenate([X for X, y in client_data], axis=0)
    all_y = np.concatenate([y for X, y in client_data], axis=0)
    
    return all_X, all_y


def get_mild_data_profile_stats(client_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Compute statistics for mild profile data.
    
    Args:
        client_data: List of (X, y) tuples from generate_non_iid_mild_data
        
    Returns:
        Dictionary with per-client statistics
    """
    stats = {}
    
    for client_id, (X, y) in enumerate(client_data):
        client_stats = {
            "n_samples": len(X),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
            "X_mean": float(X.mean()),
            "X_std": float(X.std()),
            "feature_bias": _get_mild_client_bias(client_id),
            "feature_noise": _get_mild_client_noise(client_id),
        }
        stats[f"client_{client_id}"] = client_stats
    
    return stats
