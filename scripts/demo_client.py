"""Demo script for SimulatedClientTrainer with proper time-series windows."""

import torch
from src.clients.client import SimulatedClientTrainer, ClientConfig, TaskType
from src.models.tcn import RULPredictor, FaultClassifier


def demo_rul_task():
    """Demonstrate RUL prediction with time-series windows."""
    print("=" * 60)
    print("SimulatedClientTrainer Demo - RUL Task")
    print("=" * 60)
    
    # Create proper time-series window data: (N, W, C)
    N, W, C = 100, 30, 14  # 100 samples, 30 timesteps, 14 sensor channels
    X = torch.randn(N, W, C)
    y = torch.rand(N) * 100  # RUL values 0-100
    unit_ids = torch.tensor([i // 10 for i in range(N)])  # 10 units
    
    print(f"Input shape: ({N}, {W}, {C}) - (samples, window, channels)")
    print(f"Unit IDs: 10 unique units for leakage-protected split")
    
    # Create config for RUL task
    config = ClientConfig(
        task=TaskType.RUL,
        val_split=0.2,
        normalize_per_channel=True,
        batch_size=16,
        local_epochs=3,
        lr=1e-3,
        early_stopping_enabled=True,
        early_stopping_patience=5,
        seed=42,
    )
    
    # Create model
    model = RULPredictor(num_channels=14, num_layers=3, hidden=32)
    
    # Create client with unit-aware data
    client = SimulatedClientTrainer(
        client_id=0,
        data=(X, y, unit_ids),
        config=config,
    )
    
    # Train
    result = client.train_local(global_state={"model": model})
    
    print(f"\nStatus: {result['status']}")
    print(f"Training samples: {result['num_samples']}")
    print(f"Validation samples: {result['num_val_samples']}")
    print(f"Epochs completed: {result['epochs_completed']}")
    print(f"Training time: {result['training_time_seconds']:.2f}s")
    
    print("\nTraining Metrics:")
    for k, v in result["train_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nValidation Metrics:")
    for k, v in result["val_metrics"].items():
        print(f"  {k}: {v:.4f}")


def demo_classification_task():
    """Demonstrate fault classification with time-series windows."""
    print("\n" + "=" * 60)
    print("SimulatedClientTrainer Demo - Classification Task")
    print("=" * 60)
    
    # Create proper time-series window data: (N, W, C)
    N, W, C = 100, 30, 14
    X = torch.randn(N, W, C)
    y = torch.randint(0, 2, (N,))  # Binary classification
    
    print(f"Input shape: ({N}, {W}, {C}) - (samples, window, channels)")
    print(f"Task: Binary fault classification")
    
    # Create config for classification task
    config = ClientConfig(
        task=TaskType.CLASSIFICATION,
        num_classes=2,
        val_split=0.2,
        batch_size=16,
        local_epochs=3,
        lr=1e-3,
        seed=42,
    )
    
    # Create model
    model = FaultClassifier(num_channels=14, num_classes=2, num_layers=3, hidden=32)
    
    # Create client
    client = SimulatedClientTrainer(
        client_id=0,
        data=(X, y),
        config=config,
    )
    
    # Train
    result = client.train_local(global_state={"model": model})
    
    print(f"\nStatus: {result['status']}")
    print(f"Training samples: {result['num_samples']}")
    print(f"Validation samples: {result['num_val_samples']}")
    
    print("\nTraining Metrics:")
    for k, v in result["train_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nValidation Metrics:")
    for k, v in result["val_metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    demo_rul_task()
    demo_classification_task()
