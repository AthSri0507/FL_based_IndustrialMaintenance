import numpy as np
from typing import Iterable, Sequence, Optional


def noise_injection(window: np.ndarray, scale: float = 0.01) -> np.ndarray:
    """Add Gaussian noise. `scale` may be scalar or per-channel array of length C."""
    T, C = window.shape
    if np.isscalar(scale):
        scale_arr = np.full((C,), float(scale))
    else:
        scale_arr = np.asarray(scale, dtype=float)
        if scale_arr.shape[0] != C:
            raise ValueError("scale must be scalar or length C array")
    noise = np.random.randn(T, C) * scale_arr[np.newaxis, :]
    return window + noise


def scaling(window: np.ndarray, sigma: float = 0.05, max_sigma: float = 0.1) -> np.ndarray:
    """Per-channel multiplicative scaling. `sigma` is clamped to `max_sigma`."""
    sigma = float(sigma)
    sigma = min(sigma, float(max_sigma))
    factors = 1.0 + np.random.randn(window.shape[1]) * sigma
    return window * factors[np.newaxis, :]


def channel_dropout(window: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Drop entire channels with probability `drop_prob` (small by default)."""
    mask = (np.random.rand(window.shape[1]) >= float(drop_prob)).astype(float)
    return window * mask[np.newaxis, :]


def time_warp(window: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Time-warp by perturbing time indices. Use with extreme caution for RUL."""
    T, C = window.shape
    orig = np.arange(T)
    random_curve = np.random.randn(T) * sigma
    cumulative = np.cumsum(random_curve)
    cum_min, cum_max = cumulative.min(), cumulative.max()
    if cum_max - cum_min == 0:
        return window.copy()
    scaled = (cumulative - cum_min) / (cum_max - cum_min) * (T - 1)
    warped_idx = orig + scaled - scaled.mean()
    warped_idx = np.clip(warped_idx, 0, T - 1)
    warped = np.zeros_like(window)
    for c in range(C):
        warped[:, c] = np.interp(orig, warped_idx, window[:, c])
    return warped


def apply_augmentations(
    windows: np.ndarray,
    augmentations: Iterable[str] = ("noise",),
    prob: float = 0.5,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
    max_per_window: int = 1,
) -> np.ndarray:
    """Apply augmentations to a batch of windows with RNG control and limited stacking.

    - `max_per_window`: maximum number of augmentations applied to each window (default 1)
    - If `seed` is provided, the global RNG is seeded for reproducibility in this call.
    """
    if params is None:
        params = {}
    if seed is not None:
        np.random.seed(int(seed))

    out = windows.copy()
    N = out.shape[0]
    aug_list = list(augmentations)
    for i in range(N):
        # decide how many augmentations to apply (0..max_per_window)
        k = 0
        for _ in range(max_per_window):
            if np.random.rand() <= prob:
                k += 1
        if k == 0:
            continue
        # pick k distinct augmentations (or allow repeats if k>len)
        choices = np.random.choice(aug_list, size=min(k, max(1, len(aug_list))), replace=False)
        for aug in choices:
            if aug == "noise":
                out[i] = noise_injection(out[i], **params.get("noise", {}))
            elif aug == "scale":
                out[i] = scaling(out[i], **params.get("scale", {}))
            elif aug == "time_warp":
                out[i] = time_warp(out[i], **params.get("time_warp", {}))
            elif aug == "dropout":
                out[i] = channel_dropout(out[i], **params.get("dropout", {}))
            else:
                raise ValueError(f"Unknown augmentation: {aug}")
    return out


def apply_task_augmentations(
    windows: np.ndarray,
    task: str = "rul",
    seed: Optional[int] = None,
    params: Optional[dict] = None,
    max_per_window: int = 1,
) -> np.ndarray:
    """Task-aware augmentation presets.

    - `task` in {"rul","fault","both"}
    - For `rul`: conservative defaults (no time_warp, light scaling/noise)
    - For `fault`: allow more aggressive augmentations including time_warp
    """
    if params is None:
        params = {}
    if task == "rul":
        augs = ("noise", "scale")
        default_params = {"noise": {"scale": params.get("noise_scale", 0.01)}, "scale": {"sigma": params.get("scale_sigma", 0.02), "max_sigma": 0.05}}
        prob = params.get("prob", 0.5)
        max_pw = max_per_window
    elif task == "fault":
        augs = ("noise", "scale", "dropout", "time_warp")
        default_params = {"noise": {"scale": params.get("noise_scale", 0.02)}, "scale": {"sigma": params.get("scale_sigma", 0.05), "max_sigma": 0.1}, "dropout": {"drop_prob": params.get("drop_prob", 0.05)}, "time_warp": {"sigma": params.get("time_warp_sigma", 0.05)}}
        prob = params.get("prob", 0.5)
        max_pw = params.get("max_per_window", max_per_window)
    else:
        # both: permissive but conservative defaults
        augs = ("noise", "scale", "dropout")
        default_params = {"noise": {"scale": params.get("noise_scale", 0.015)}, "scale": {"sigma": params.get("scale_sigma", 0.03), "max_sigma": 0.08}, "dropout": {"drop_prob": params.get("drop_prob", 0.03)}}
        prob = params.get("prob", 0.4)
        max_pw = max_per_window

    # merge provided params with defaults (defaults used when key missing)
    merged = default_params.copy()
    for k, v in (params or {}).items():
        if k in merged and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v

    return apply_augmentations(windows, augmentations=augs, prob=prob, params=merged, seed=seed, max_per_window=max_pw)
