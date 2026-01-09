import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree
from src.data.augment import apply_augmentations


def _flatten_windows(windows: np.ndarray) -> np.ndarray:
    N, W, C = windows.shape
    return windows.reshape(N, W * C)


def _unflatten_windows(flat: np.ndarray, W: int, C: int) -> np.ndarray:
    N = flat.shape[0]
    return flat.reshape(N, W, C)


def balance_windows_smote(windows: np.ndarray, labels: np.ndarray, target_count: Optional[int] = None, k: int = 5, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Balance windows using a simple SMOTE-like algorithm.

    Args:
        windows: numpy array of shape (N, W, C)
        labels: numpy array of shape (N,) with integer class labels
        target_count: if provided, target samples per class; otherwise use the max class count
        k: number of nearest neighbors to use when synthesizing
        random_state: optional seed

    Returns:
        (windows_resampled, labels_resampled)
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    if windows.ndim != 3:
        raise ValueError("windows must be (N, W, C)")

    classes, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(classes, counts))
    if target_count is None:
        target_count = int(max(counts))

    N, W, C = windows.shape
    flat = _flatten_windows(windows)

    new_list = [flat]
    label_list = [labels]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        n_cls = len(cls_idx)
        if n_cls >= target_count:
            continue
        need = int(target_count - n_cls)
        samples = flat[cls_idx]
        # Build KD-tree on minority samples
        tree = cKDTree(samples)
        # For each synthetic example
        synths = []
        for _ in range(need):
            # pick a random sample from minority
            i = rng.randint(0, n_cls)
            x = samples[i]
            # find k nearest neighbors (exclude self)
            nn = tree.query(x, k=min(k + 1, len(samples)))[1]
            # ensure neighbor is not itself
            if np.isscalar(nn):
                nn = np.array([nn])
            nn = nn[nn != i]
            if len(nn) == 0:
                # duplicate the sample
                neighbor = x
            else:
                neighbor = samples[rng.choice(nn)]
            # interpolate
            alpha = rng.rand()
            new = x + alpha * (neighbor - x)
            synths.append(new)
        if synths:
            new_arr = np.stack(synths, axis=0)
            new_list.append(new_arr)
            label_list.append(np.full(len(synths), cls, dtype=labels.dtype))

    if len(new_list) == 1:
        return windows.copy(), labels.copy()

    all_flat = np.concatenate(new_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    all_windows = _unflatten_windows(all_flat, W, C)
    return all_windows, all_labels


def balance_windows_via_augmentation(
    windows: np.ndarray,
    labels: np.ndarray,
    target_count: Optional[int] = None,
    augmentations: Tuple[str, ...] = ("noise", "time_warp", "scale", "dropout"),
    aug_prob: float = 1.0,
    aug_params: dict = None,
    random_state: Optional[int] = None,
    task: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance windows by creating augmented copies of minority-class windows.

    This avoids interpolation between different windows (which can break temporal
    structure) by generating synthetic examples via time-preserving augmentations
    (noise, time-warp, scaling, channel dropout).

    Args:
        windows: (N, W, C)
        labels: (N,)
        target_count: desired samples per class (defaults to max class count)
        augmentations: augmentation names passed to `apply_augmentations`
        aug_prob: probability to apply each augmentation when generating a sample
        aug_params: dict mapping augmentation name -> kwargs
        random_state: seed

    Returns:
        (windows_resampled, labels_resampled)
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    if windows.ndim != 3:
        raise ValueError("windows must be (N, W, C)")

    classes, counts = np.unique(labels, return_counts=True)
    if target_count is None:
        target_count = int(max(counts))

    N, W, C = windows.shape
    new_windows = [windows]
    new_labels = [labels]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        n_cls = len(cls_idx)
        if n_cls >= target_count:
            continue
        need = int(target_count - n_cls)
        synths = []
        for _ in range(need):
            i = rng.choice(cls_idx)
            w = windows[i:i+1].copy()  # shape (1,W,C)
            # determine augmentations to use for this call
            if task is not None:
                # task-aware defaults (conservative for 'rul')
                if task == "rul":
                    allowed = ("noise", "scale")
                elif task == "fault":
                    allowed = ("noise", "scale", "dropout", "time_warp")
                else:
                    allowed = augmentations
            else:
                allowed = augmentations

            # choose at most one augmentation per synthetic window (avoid stacking)
            aug_choice = tuple([rng.choice(list(allowed))])

            # seed apply_augmentations for reproducibility
            seed = int(rng.randint(0, 2 ** 31 - 1))
            w_aug = apply_augmentations(w, augmentations=aug_choice, prob=1.0, params=aug_params, seed=seed, max_per_window=1)
            synths.append(w_aug[0])
        if synths:
            new_windows.append(np.stack(synths, axis=0))
            new_labels.append(np.full(len(synths), cls, dtype=labels.dtype))

    all_windows = np.concatenate(new_windows, axis=0)
    all_labels = np.concatenate(new_labels, axis=0)
    return all_windows, all_labels
