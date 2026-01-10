"""Message structure and tensor-only payload validation for FL messages.

This module enforces a strict contract for messages exchanged between
clients and the server. Rules:
 - Only primitive metadata (str, int, float, bool), dicts, and
     PyTorch tensors (`torch.Tensor`) are allowed as numeric payloads.
 - NumPy arrays (`numpy.ndarray`) are explicitly disallowed to avoid
     accidental use of non-tensor numeric buffers.
 - Raw Python lists/tuples of numbers are disallowed; users must pass
     numeric data as tensors.

This is a structure-and-sanity checker — it does not provide cryptographic
privacy guarantees (DP or secure aggregation). It should be thought of as
"message structure & tensor-only payload validation" rather than a
full privacy verification system.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple
import torch
import numpy as np


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]


def _is_tensor_like(x: Any) -> bool:
    # Only accept PyTorch tensors as tensor-like. NumPy arrays are
    # explicitly rejected so callers do not accidentally send numeric
    # buffers that bypass tensor APIs.
    return isinstance(x, torch.Tensor)


def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))


def _validate_value(path: str, value: Any, errors: List[str]) -> None:
    """Recursively validate a value.

    Allowed types: primitives, `dict`, and `torch.Tensor`.
    Disallowed: `numpy.ndarray`, raw `list`/`tuple` numeric buffers, and any
    unexpected container types.
    """
    if _is_primitive(value):
        return

    # Accept torch tensors only
    if _is_tensor_like(value):
        return

    # Explicitly forbid NumPy arrays
    if isinstance(value, np.ndarray):
        errors.append(f"NumPy array found at {path}; convert to torch.Tensor instead")
        return

    if isinstance(value, dict):
        for k, v in value.items():
            _validate_value(f"{path}.{k}", v, errors)
        return

    if isinstance(value, (list, tuple)):
        # Disallow raw lists/tuples entirely for numeric payloads. If the
        # user has a list of metadata dicts, they must still ensure that the
        # dicts contain only primitives/tensors; however permitting lists
        # creates a wide attack surface so we disallow them here.
        errors.append(f"Raw list/tuple at {path} is not allowed; use tensors or dicts")
        return

    # Fallback: unknown/unexpected type
    errors.append(f"Unexpected type at {path}: {type(value).__name__}")


def validate_message(message: Any, allowed_root_keys: Tuple[str, ...] = None) -> ValidationResult:
    """Validate a federated message payload.

    Parameters
    - message: usually a dict representing the payload (e.g., {
        'client_id': 'c1', 'round_id': 1, 'delta': {...}, 'num_samples': 10,
        'metrics': {...}
      })
    - allowed_root_keys: optional tuple of allowed top-level keys. If provided,
      any other top-level key will be reported as an error.

    Returns: ValidationResult(valid, errors)
    """
    errors: List[str] = []

    if not isinstance(message, dict):
        return ValidationResult(False, ["Message must be a dict at root"])

    if allowed_root_keys is not None:
        for k in message.keys():
            if k not in allowed_root_keys:
                errors.append(f"Unexpected root key: {k}")

    for k, v in message.items():
        _validate_value(k, v, errors)

    return ValidationResult(valid=(len(errors) == 0), errors=errors)
