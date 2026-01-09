"""FedAvg Aggregator — weighted averaging and client selection.

This module provides:
- fedavg_aggregate: weighted averaging of model deltas by sample count
- sample_clients: client selection per round (random or stratified)
- FedAvgAggregator: stateful aggregator class for use in server

Design notes:
- This implements **delta-based FedAvg**: clients return (local_weights - global_weights),
  and we aggregate deltas then apply to the global model. This is mathematically equivalent
  to weight-based FedAvg **only when all clients initialize local training from the same
  global model checkpoint at the start of each round**. If a client trains on a stale
  checkpoint or resumes from a previous local state, the equivalence breaks.
- The aggregator is round-aware: updates must match the current round_id to be accepted.
- Duplicate submissions from the same client in a round are rejected.
- **Precision policy**: All deltas are aggregated in float32 regardless of client dtype.
  This is intentional for numerical stability and memory efficiency. Mixed-precision
  clients will have their updates coerced to float32 during aggregation.

Security limitations (acknowledged, not addressed):
- **num_samples is client-reported and trusted**: A malicious client can inflate its
  sample count to bias the global model. This is acceptable for academic experiments
  but not for adversarial settings. Mitigations (not implemented): server-side sample
  verification, contribution clipping, or Byzantine-robust aggregation.
"""

import random
from typing import Dict, List, Optional, Set
from copy import deepcopy

import torch
import torch.nn as nn


class DeltaValidationError(ValueError):
    """Raised when client delta fails validation."""
    pass


def validate_delta_compatibility(
    reference: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    client_id: str = "unknown",
) -> None:
    """Validate that a delta is compatible with the reference (first delta or model).

    Checks:
    - identical key sets
    - matching tensor shapes
    - matching dtypes

    Raises:
        DeltaValidationError: if validation fails
    """
    ref_keys = set(reference.keys())
    delta_keys = set(delta.keys())

    # check key sets match
    missing = ref_keys - delta_keys
    extra = delta_keys - ref_keys
    if missing or extra:
        raise DeltaValidationError(
            f"Client {client_id}: key mismatch. Missing: {missing}, Extra: {extra}"
        )

    # check shapes and dtypes
    for key in ref_keys:
        ref_tensor = reference[key]
        delta_tensor = delta[key]

        if ref_tensor.shape != delta_tensor.shape:
            raise DeltaValidationError(
                f"Client {client_id}: shape mismatch for '{key}'. "
                f"Expected {ref_tensor.shape}, got {delta_tensor.shape}"
            )

        # dtype check (allow float32/float64 interop but warn on int vs float)
        if ref_tensor.is_floating_point() != delta_tensor.is_floating_point():
            raise DeltaValidationError(
                f"Client {client_id}: dtype mismatch for '{key}'. "
                f"Expected floating={ref_tensor.is_floating_point()}, "
                f"got floating={delta_tensor.is_floating_point()}"
            )


def fedavg_aggregate(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    validate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Weighted averaging of model deltas (FedAvg).

    This function assumes deltas have already been validated for compatibility
    (matching keys, shapes, dtypes) when called from FedAvgAggregator. For direct
    use, set validate=True to enable validation.

    All aggregation is performed in float32 for numerical stability.

    Args:
        deltas: list of state_dict deltas from clients, each is {param_name: tensor}
        weights: list of weights (typically num_samples per client)
        validate: if True, validate delta compatibility (default False for performance)

    Returns:
        Aggregated delta as state_dict (float32)

    Raises:
        ValueError: if inputs are invalid
        DeltaValidationError: if validate=True and deltas are incompatible
    """
    if len(deltas) == 0:
        raise ValueError("No deltas to aggregate")
    if len(deltas) != len(weights):
        raise ValueError("Number of deltas must match number of weights")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive")

    total_weight = sum(weights)
    reference = deltas[0]

    # optional validation for direct callers (FedAvgAggregator validates on add_update)
    if validate:
        for i, delta in enumerate(deltas[1:], start=1):
            validate_delta_compatibility(reference, delta, client_id=f"client_{i}")

    # normalize weights
    norm_weights = [w / total_weight for w in weights]

    # initialize aggregated delta with zeros (float32 by design)
    agg_delta: Dict[str, torch.Tensor] = {}
    for key in reference.keys():
        agg_delta[key] = torch.zeros_like(reference[key], dtype=torch.float32)

    # weighted sum (coerce to float32)
    for delta, w in zip(deltas, norm_weights):
        for key in agg_delta.keys():
            agg_delta[key] = agg_delta[key] + w * delta[key].float()

    return agg_delta


def apply_delta(model: nn.Module, delta: Dict[str, torch.Tensor]) -> None:
    """Apply aggregated delta to model weights in-place.

    Args:
        model: the global model to update
        delta: aggregated delta from fedavg_aggregate
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in delta:
                param.add_(delta[name].to(param.device))


def sample_clients(
    client_ids: List[str],
    num_clients: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample a subset of clients for a training round.

    Uses a local RNG instance to avoid polluting global random state.

    Args:
        client_ids: list of all available client IDs
        num_clients: exact number of clients to sample (takes precedence)
        fraction: fraction of clients to sample (0 < fraction <= 1)
        seed: random seed for reproducibility (uses local RNG, not global)

    Returns:
        List of selected client IDs
    """
    if len(client_ids) == 0:
        return []

    # use local RNG to avoid polluting global state
    rng = random.Random(seed)

    n = len(client_ids)

    if num_clients is not None:
        k = min(num_clients, n)
    elif fraction is not None:
        k = max(1, int(fraction * n))
    else:
        k = n  # select all

    return rng.sample(client_ids, k)


class FedAvgAggregator:
    """Stateful FedAvg aggregator for server integration.

    Collects client updates during a round and produces aggregated delta.

    Safety guarantees:
    - Round-aware: updates must match current round_id
    - No duplicate submissions: each client can submit once per round
    - Positive samples required: zero-sample updates are rejected
    - Delta validation: incompatible deltas are rejected immediately
    """

    def __init__(self, round_id: int = 0):
        """Initialize aggregator for a specific round.

        Args:
            round_id: the round this aggregator is collecting updates for
        """
        self._round_id: int = round_id
        self._deltas: List[Dict[str, torch.Tensor]] = []
        self._weights: List[float] = []
        self._client_ids: List[str] = []
        self._client_set: Set[str] = set()  # for O(1) duplicate check
        self._reference_delta: Optional[Dict[str, torch.Tensor]] = None

    @property
    def round_id(self) -> int:
        """Current round ID."""
        return self._round_id

    def reset(self, round_id: Optional[int] = None) -> None:
        """Reset state for a new round.

        Args:
            round_id: new round ID (increments current if not specified)
        """
        if round_id is not None:
            self._round_id = round_id
        else:
            self._round_id += 1
        self._deltas = []
        self._weights = []
        self._client_ids = []
        self._client_set = set()
        self._reference_delta = None

    def add_update(
        self,
        client_id: str,
        delta: Dict[str, torch.Tensor],
        num_samples: int,
        round_id: Optional[int] = None,
    ) -> None:
        """Add a client update.

        Args:
            client_id: client identifier
            delta: model delta from client.train_local()
            num_samples: number of samples used for training
            round_id: round this update belongs to (must match current round)

        Raises:
            ValueError: if update is rejected (wrong round, duplicate, zero samples)
            DeltaValidationError: if delta is incompatible with previous deltas
        """
        # check round binding
        if round_id is not None and round_id != self._round_id:
            raise ValueError(
                f"Round mismatch: aggregator is on round {self._round_id}, "
                f"but update is for round {round_id}"
            )

        # check duplicate submission
        if client_id in self._client_set:
            raise ValueError(
                f"Duplicate update: client {client_id} already submitted for round {self._round_id}"
            )

        # check positive samples
        if num_samples <= 0:
            raise ValueError(
                f"Invalid num_samples={num_samples} from client {client_id}. Must be > 0"
            )

        # validate delta compatibility
        if self._reference_delta is not None:
            validate_delta_compatibility(self._reference_delta, delta, client_id)
        else:
            # clone first delta as reference to prevent aliasing bugs
            # (caller may mutate the dict after submission)
            self._reference_delta = {k: v.detach().clone() for k, v in delta.items()}

        # accept update
        self._client_ids.append(client_id)
        self._client_set.add(client_id)
        self._deltas.append(delta)
        self._weights.append(float(num_samples))

    @property
    def num_updates(self) -> int:
        """Number of updates received."""
        return len(self._deltas)

    @property
    def client_ids(self) -> List[str]:
        """IDs of clients that submitted updates."""
        return self._client_ids.copy()

    @property
    def total_samples(self) -> int:
        """Total samples across all updates."""
        return int(sum(self._weights))

    def has_client(self, client_id: str) -> bool:
        """Check if a client has already submitted."""
        return client_id in self._client_set

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """Compute and return the aggregated delta.

        Returns:
            Aggregated delta (weighted average of all submitted deltas)

        Raises:
            ValueError: if no updates have been submitted
        """
        if self.num_updates == 0:
            raise ValueError("No updates to aggregate")
        return fedavg_aggregate(self._deltas, self._weights)

    def aggregate_and_apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Aggregate deltas and apply to model in-place.

        Args:
            model: global model to update

        Returns:
            The aggregated delta that was applied
        """
        delta = self.aggregate()
        apply_delta(model, delta)
        return delta

    def get_summary(self) -> Dict:
        """Get summary of current aggregator state."""
        return {
            "round_id": self._round_id,
            "num_updates": self.num_updates,
            "client_ids": self.client_ids,
            "total_samples": self.total_samples,
            "weights": self._weights.copy(),
        }
