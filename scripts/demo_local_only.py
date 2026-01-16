#!/usr/bin/env python
"""Demo script for local-only baseline experiment.

This script demonstrates the local-only training baseline, where each client
trains independently without any federated aggregation.

Run with:
    python scripts/demo_local_only.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.local_only_baseline import (
    LocalOnlyConfig,
    LocalOnlyBaseline,
    create_synthetic_data,
)


def main():
    print("=" * 60)
    print("Local-Only Baseline Demo")
    print("=" * 60)
    
    # Create small synthetic dataset
    print("\nGenerating synthetic data...")
    X, y = create_synthetic_data(
        num_samples=300,
        seq_length=50,
        num_channels=10,
        task="rul",
        seed=42,
    )
    print(f"Data shape: {X.shape}")
    print(f"RUL range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Configure experiment with 4 clients
    config = LocalOnlyConfig(
        num_clients=4,
        heterogeneity_mode="uniform",
        local_epochs=10,
        batch_size=16,
        lr=1e-3,
        early_stopping_patience=3,
        seed=42,
        device="cpu",
        save_client_models=False,
        normalize_rul=True,
    )
    
    print(f"\nExperiment Configuration:")
    print(f"  Clients: {config.num_clients}")
    print(f"  Heterogeneity: {config.heterogeneity_mode}")
    print(f"  Local epochs: {config.local_epochs}")
    print(f"  RUL normalization: {config.normalize_rul}")
    
    # Run experiment
    print("\n" + "=" * 60)
    print("Training Independent Client Models")
    print("=" * 60)
    
    experiment = LocalOnlyBaseline(config)
    results = experiment.run(X=X, y=y)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)
    
    print("\nPer-Client Performance on Global Test Set:")
    print("-" * 50)
    print(f"{'Client':<10} {'MAE':<12} {'RMSE':<12} {'Train Samples'}")
    print("-" * 50)
    
    for client in results["per_client"]:
        cid = client["client_id"]
        mae = client["global_eval"]["mae"]
        rmse = client["global_eval"]["rmse"]
        n_samples = client["num_samples"]
        print(f"{cid:<10} {mae:<12.2f} {rmse:<12.2f} {n_samples}")
    
    print("-" * 50)
    
    # Aggregate stats
    global_metrics = results["global_metrics"]
    print(f"\nAggregate (mean ± std):")
    print(f"  MAE:  {global_metrics['mae']['mean']:.2f} ± {global_metrics['mae']['std']:.2f}")
    print(f"  RMSE: {global_metrics['rmse']['mean']:.2f} ± {global_metrics['rmse']['std']:.2f}")
    
    print(f"\nTotal training time: {results['total_train_time']:.2f}s")
    
    # Observation
    print("\n" + "=" * 60)
    print("Key Observations")
    print("=" * 60)
    print("""
    - Each client trains independently on their own data partition
    - Clients have no knowledge of other clients' data
    - Performance varies across clients based on their local data
    - Global test evaluation shows how well each client generalizes
    
    This baseline establishes a LOWER BOUND on performance:
    - Federated learning should improve upon this by combining knowledge
    - Centralized learning provides an UPPER BOUND (all data together)
    """)


if __name__ == "__main__":
    main()
