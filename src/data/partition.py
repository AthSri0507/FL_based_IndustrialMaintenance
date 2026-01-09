import os
from typing import Dict, List
import numpy as np
import pandas as pd


def generate_client_partitions(df: pd.DataFrame, out_dir: str, num_clients: int = 3, heterogeneity: Dict[int, float] = None, prefix: str = "client", seed: int = None) -> List[str]:
    """Partition a DataFrame into `num_clients` client datasets and write to `out_dir`.

    - df: DataFrame with sensor columns and optional `label` column (0/1)
    - out_dir: directory to write client files
    - num_clients: number of client partitions to generate
    - heterogeneity: optional dict mapping client_idx -> desired fault prevalence (0..1).
      If provided, partitioning will try to produce the requested label prevalences per client.
    - prefix: filename prefix; outputs {prefix}_{i}.parquet

    Returns list of written file paths.

    Notes:
    - When `heterogeneity` is not provided, the function performs roughly equal random splits.
    - When `heterogeneity` is provided and `label` exists, the function does stratified sampling
      to approximate desired prevalences per client.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    indices = np.arange(n)
    written = []
    # deterministic RNG for reproducibility
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # pre-allocate client sizes (balanced as possible)
    base_size = n // num_clients
    sizes = [base_size] * num_clients
    for i in range(n % num_clients):
        sizes[i] += 1

    if heterogeneity and "label" in df.columns:
        # create pools for positive and negative labels (use original indices)
        pos_idx = df.index[df["label"] == 1].to_numpy().tolist()
        neg_idx = df.index[df["label"] == 0].to_numpy().tolist()
        # For each client, sample desired number of positives/negatives without reuse
        remaining_pos = pos_idx[:]
        remaining_neg = neg_idx[:]
        for i in range(num_clients):
            desired_prev = heterogeneity.get(i, None)
            size_i = sizes[i]
            if desired_prev is None:
                # default to proportional split based on remaining pool
                # we'll simply aim for size_i * global_pos_fraction
                global_pos_frac = len(pos_idx) / max(1, n)
                k_pos_target = int(round(global_pos_frac * size_i))
            else:
                # clamp desired prevalence to what is achievable given remaining pool
                max_possible = len(remaining_pos) / max(1, size_i)
                desired_prev_clamped = min(desired_prev, max_possible)
                k_pos_target = int(round(desired_prev_clamped * size_i))

            # ensure we don't request more positives than remain
            k_pos = min(k_pos_target, len(remaining_pos), size_i)
            k_neg = min(size_i - k_pos, len(remaining_neg))

            samp_pos = []
            samp_neg = []
            if k_pos > 0 and remaining_pos:
                samp_pos = list(rng.choice(remaining_pos, size=k_pos, replace=False))
                remaining_pos = [x for x in remaining_pos if x not in samp_pos]
            if k_neg > 0 and remaining_neg:
                samp_neg = list(rng.choice(remaining_neg, size=k_neg, replace=False))
                remaining_neg = [x for x in remaining_neg if x not in samp_neg]

            sel_idx = samp_pos + samp_neg

            # if still underfilled (due to exhausted pos/neg pools), take from combined remaining pool
            combined_remaining = [x for x in (remaining_pos + remaining_neg) if x not in sel_idx]
            need_more = size_i - len(sel_idx)
            if need_more > 0 and combined_remaining:
                take = min(need_more, len(combined_remaining))
                extra = list(rng.choice(combined_remaining, size=take, replace=False))
                sel_idx += extra
                # remove extras from whichever pool they were in
                remaining_pos = [x for x in remaining_pos if x not in extra]
                remaining_neg = [x for x in remaining_neg if x not in extra]

            # if sel_idx is still empty (extremely small dataset), continue with empty partition
            part = df.loc[sel_idx].reset_index()
            path = os.path.join(out_dir, f"{prefix}_{i}.parquet")
            try:
                part.to_parquet(path)
            except Exception:
                path = os.path.join(out_dir, f"{prefix}_{i}.csv")
                part.to_csv(path, index=False)
            written.append(path)
    else:
        # simple random split into equal-sized chunks (deterministic via rng)
        if seed is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
        sizes = [n // num_clients] * num_clients
        for i in range(n % num_clients):
            sizes[i] += 1
        pos = 0
        for i, sz in enumerate(sizes):
            sel = indices[pos : pos + sz]
            pos += sz
            part = df.iloc[sel].reset_index()
            path = os.path.join(out_dir, f"{prefix}_{i}.parquet")
            try:
                part.to_parquet(path)
            except Exception:
                path = os.path.join(out_dir, f"{prefix}_{i}.csv")
                part.to_csv(path, index=False)
            written.append(path)
    return written
