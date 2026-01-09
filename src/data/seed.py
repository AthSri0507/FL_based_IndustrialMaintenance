import os
from typing import Union, Sequence, Optional
import numpy as np
import pandas as pd


def bootstrap_clients_from_dataframe(
    df: pd.DataFrame,
    out_dir: str,
    num_clients: int = 3,
    noise_scale: float = 0.01,
    prefix: str = "client",
    seed: Optional[int] = None,
    flip_rate: float = 0.0,
) -> Sequence[str]:
    """Create `num_clients` synthetic client files by perturbing a seed dataframe.

    Safety-first defaults:
    - deterministic when `seed` is provided
    - label flipping disabled by default (`flip_rate=0.0`)
    - per-client noise is explicit and multiplier is clamped

    This function is intended for demos and bootstrapping synthetic clients, not
    for final evaluation datasets unless you explicitly opt-in.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    os.makedirs(out_dir, exist_ok=True)
    written = []
    sensor_cols = [c for c in df.columns if c not in {"label", "timestamp", "client_id"}]

    for i in range(num_clients):
        df_copy = df.copy()
        # per-client noise multiplier to create explicit heterogeneity (bounded)
        base_multiplier = 1.0 + rng.randn() * 0.1
        multiplier = float(np.clip(base_multiplier, 0.7, 1.3))
        # optional linear client-level scaling of noise for reproducible heterogeneity
        client_scale = 1.0 + 0.05 * float(i)
        client_noise = float(noise_scale) * multiplier * client_scale

        noise = rng.randn(*df_copy[sensor_cols].shape) * client_noise
        # preserve numeric dtype strictly
        df_copy[sensor_cols] = df_copy[sensor_cols].astype("float32") + noise.astype("float32")

        # label flipping only if explicitly requested
        if flip_rate and "label" in df_copy.columns and flip_rate > 0.0:
            # flip 0 -> 1 with probability `flip_rate` (per-row)
            mask = rng.rand(len(df_copy)) < float(flip_rate)
            df_copy.loc[mask, "label"] = 1

        filename = f"{prefix}_{i}.parquet"
        path = os.path.join(out_dir, filename)
        try:
            df_copy.to_parquet(path)
        except Exception:
            # fallback to csv
            csv_path = os.path.join(out_dir, f"{prefix}_{i}.csv")
            df_copy.to_csv(csv_path, index=False)
            path = csv_path
        written.append(path)
    return written


def seed_from_dataframe(df: pd.DataFrame, out_dir: str, num_clients: int = 3, noise_scale: float = 0.01, prefix: str = "client") -> Sequence[str]:
    """Backward-compatible wrapper for `bootstrap_clients_from_dataframe`.

    Kept to avoid breaking existing tests or scripts. For new code prefer
    `bootstrap_clients_from_dataframe(..., seed=..., flip_rate=...)`.
    """
    return bootstrap_clients_from_dataframe(df, out_dir, num_clients=num_clients, noise_scale=noise_scale, prefix=prefix)
