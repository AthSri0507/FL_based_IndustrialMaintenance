from typing import Union
import numpy as np
import pandas as pd


def segment_windows(data: Union[np.ndarray, pd.DataFrame], W: int, H: int, normalize: bool = True) -> np.ndarray:
    """Segment a multivariate time-series into sliding windows.

    Args:
        data: numpy array shape (T, C) or pandas DataFrame with shape (T, C)
        W: window length (number of timesteps)
        H: hop length (stride between windows)
        normalize: if True, normalize each window to zero mean and unit std per channel

    Returns:
        windows: numpy array of shape (N, W, C) where N = number of windows
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    T, C = arr.shape
    if W <= 0 or H <= 0:
        raise ValueError("W and H must be positive integers")

    windows = []
    for start in range(0, T - W + 1, H):
        w = arr[start : start + W].astype(float)
        if normalize:
            mean = w.mean(axis=0, keepdims=True)
            std = w.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            w = (w - mean) / std
        windows.append(w)

    if not windows:
        return np.empty((0, W, C))
    return np.stack(windows, axis=0)
