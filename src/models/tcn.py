"""Temporal Convolutional Network (TCN) for RUL prediction and fault classification.

This module provides:
- TemporalBlock: causal dilated convolution block with residual connection
- TCNEncoder: stacked temporal blocks producing fixed-dim representation
- RULPredictor: TCN encoder + regression head for Remaining Useful Life
- FaultClassifier: TCN encoder + classifier head for fault detection/classification
- Evaluation metrics: MAE, RMSE, scoring function (RUL); accuracy, precision, recall, F1 (classification)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to preserve sequence length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Single TCN block: two causal convolutions + LayerNorm + residual + dropout.

    LayerNorm is applied after convolution, before activation. This stabilizes
    training under non-IID federated settings where clients have different
    signal scales and noise levels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        # LayerNorm on channel dimension (applied after transpose)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # residual projection if channel mismatch
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        out = self.conv1(x)  # (N, out_channels, L)
        out = out.transpose(1, 2)  # (N, L, out_channels) for LayerNorm
        out = self.norm1(out).transpose(1, 2)  # back to (N, out_channels, L)
        out = self.dropout(self.relu(out))

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out).transpose(1, 2)
        out = self.dropout(self.relu(out))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""

    def __init__(self, num_channels: int, num_layers: int, kernel_size: int = 3, dropout: float = 0.2, hidden: int = 64):
        """
        Args:
            num_channels: input channels (sensor features C)
            num_layers: number of temporal blocks
            kernel_size: convolution kernel size
            dropout: dropout rate
            hidden: hidden dimension for all blocks
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = num_channels if i == 0 else hidden
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.out_channels = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L) -> (N, hidden, L)
        return self.network(x)


class RULPredictor(nn.Module):
    """TCN-based RUL predictor.

    Input: (N, W, C) windows of sensor readings (W timesteps, C channels).
    Output: (N,) predicted RUL values.
    """

    def __init__(
        self,
        num_channels: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        hidden: int = 64,
        dropout: float = 0.2,
        fc_hidden: int = 32,
    ):
        super().__init__()
        self.encoder = TCNEncoder(num_channels, num_layers, kernel_size, dropout, hidden)
        # regression head: global avg pool -> fc layers -> scalar
        self.head = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, W, C) -> transpose to (N, C, W) for conv
        x = x.transpose(1, 2)  # (N, C, W)
        enc = self.encoder(x)  # (N, hidden, W)
        pooled = enc.mean(dim=2)  # (N, hidden)
        out = self.head(pooled).squeeze(-1)  # (N,)
        return out


# ---------------------- Loss & Metrics ----------------------

def rul_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss for RUL regression."""
    return F.mse_loss(pred, target)


def rul_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    with torch.no_grad():
        return float((pred - target).abs().mean().item())


def rul_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    with torch.no_grad():
        return float(math.sqrt(F.mse_loss(pred, target).item()))


def rul_score(pred: torch.Tensor, target: torch.Tensor, a1: float = 13.0, a2: float = 10.0) -> float:
    """Asymmetric scoring function (NASA turbofan style).

    Penalizes late predictions more than early predictions.
    s = sum( exp(-d/a1) - 1 ) for d < 0 (early)
      + sum( exp(d/a2) - 1 )  for d >= 0 (late)
    """
    with torch.no_grad():
        d = pred - target  # positive = late, negative = early
        early = d < 0
        late = ~early
        score = torch.zeros_like(d)
        score[early] = torch.exp(-d[early] / a1) - 1
        score[late] = torch.exp(d[late] / a2) - 1
        return float(score.sum().item())


def compute_rul_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute MAE, RMSE, and score for RUL predictions."""
    return {
        "mae": rul_mae(pred, target),
        "rmse": rul_rmse(pred, target),
        "score": rul_score(pred, target),
    }


# ---------------------- Fault Classifier ----------------------

class FaultClassifier(nn.Module):
    """TCN-based fault detector/classifier.

    Shares the same TCN encoder trunk as RULPredictor but uses a classification head.
    Input: (N, W, C) windows of sensor readings.
    Output: (N, num_classes) logits for fault classes.

    Design notes:
    - This model assumes **single-label classification** (mutually exclusive fault classes).
      Each window is assigned exactly one class via argmax. For multi-label fault detection,
      use sigmoid + BCE loss instead of softmax + CE loss.
    - Encoder sharing: if you pass an existing `encoder`, ensure you use a **single optimizer**
      for joint training. Using separate optimizers on shared weights causes weight contamination.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        num_layers: int = 4,
        kernel_size: int = 3,
        hidden: int = 64,
        dropout: float = 0.2,
        fc_hidden: int = 32,
        encoder: TCNEncoder = None,
    ):
        """
        Args:
            num_channels: input channels (sensor features)
            num_classes: number of fault classes (2 for binary fault detection)
            num_layers: TCN encoder layers
            kernel_size: convolution kernel size
            hidden: encoder hidden dimension
            dropout: dropout rate
            fc_hidden: classifier head hidden dimension
            encoder: optionally share an existing TCNEncoder (for multi-task)
        """
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
            hidden = encoder.out_channels
        else:
            self.encoder = TCNEncoder(num_channels, num_layers, kernel_size, dropout, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, W, C) -> (N, C, W)
        x = x.transpose(1, 2)
        enc = self.encoder(x)  # (N, hidden, W)
        pooled = enc.mean(dim=2)  # (N, hidden)
        logits = self.head(pooled)  # (N, num_classes)
        return logits


# ---------------------- Classification Metrics ----------------------

def classification_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute accuracy from logits and integer targets."""
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return float((preds == target).float().mean().item())


def classification_precision_recall_f1(logits: torch.Tensor, target: torch.Tensor, pos_label: int = 1) -> dict:
    """Compute precision, recall, F1 for binary or specified positive class.

    Args:
        logits: (N, num_classes) logits
        target: (N,) integer class labels
        pos_label: positive class index for precision/recall/f1

    Returns:
        dict with precision, recall, f1 keys
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        tp = ((preds == pos_label) & (target == pos_label)).sum().float()
        fp = ((preds == pos_label) & (target != pos_label)).sum().float()
        fn = ((preds != pos_label) & (target == pos_label)).sum().float()

        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))

        return {"precision": precision, "recall": recall, "f1": f1}


def compute_classification_metrics(logits: torch.Tensor, target: torch.Tensor, pos_label: int = 1) -> dict:
    """Compute accuracy, precision, recall, F1 for classification."""
    acc = classification_accuracy(logits, target)
    prf = classification_precision_recall_f1(logits, target, pos_label)
    return {"accuracy": acc, **prf}


def fault_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for single-label fault classification.

    Args:
        logits: (N, num_classes) unnormalized scores
        target: (N,) integer class labels

    Returns:
        Scalar loss tensor
    """
    return F.cross_entropy(logits, target)
