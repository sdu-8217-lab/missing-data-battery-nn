"""
Integration test for the full pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from src.data.splits import train_val_test_split
from src.missing_data.mcar import simulate_mcar
from src.missing_data.mar import simulate_mar
from src.models.model_factory import create_model


def test_mcar_pipeline():
    """Test full pipeline with MCAR"""
    print("Testing MCAR pipeline...")
    
    # Create synthetic data
    X = np.random.randn(200, 16).astype(np.float32)
    y = np.random.rand(200).astype(np.float32)
    
    cfg = OmegaConf.create({
        "data": {
            "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            "preprocessing": {"standardize": False}
        },
        "model": {
            "name": "cnn1d",
            "channels": [16, 32],
            "kernel_sizes": [3, 3],
            "dropout": 0.1,
            "use_batch_norm": True,
        },
        "experiment": {"use_mim": True},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "missing": {"mode": "mcar"},
    })
    
    # Split data
    data = train_val_test_split(X, y, cfg)
    
    # Apply MCAR
    X_train_imp, mask_train, mim_train = simulate_mcar(
        data["X_train"], missing_rate=0.3, seed=42
    )
    
    # Check shapes
    assert mim_train.shape[1] == 32  # 16 + 16
    
    # Create model
    model = create_model(cfg)
    
    # Test forward pass
    with torch.no_grad():
        output = model(mim_train[:10])
    
    assert output.shape == (10,)
    
    print("  ✓ MCAR pipeline test passed")


def test_mar_pipeline():
    """Test full pipeline with MAR"""
    print("Testing MAR pipeline...")
    
    # Create synthetic data with SOH pattern
    X = np.random.randn(200, 16).astype(np.float32)
    y = np.linspace(1.0, 0.6, 200).astype(np.float32)  # SOH degradation
    
    cfg = OmegaConf.create({
        "data": {
            "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            "preprocessing": {"standardize": False}
        },
        "model": {
            "name": "mlp",
            "hidden_dims": [32, 16],
            "dropout": 0.1,
            "use_batch_norm": True,
        },
        "experiment": {"use_mim": True},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "missing": {
            "mode": "mar",
            "alpha": None,
            "beta": 2.0,
            "gamma": 0.05,
        },
    })
    
    # Split data
    data = train_val_test_split(X, y, cfg)
    
    # Apply MAR
    X_train_imp, mask_train, mim_train = simulate_mar(
        data["X_train"],
        data["y_train"],
        missing_rate=0.3,
        alpha=None,
        beta=2.0,
        gamma=0.05,
        seed=42
    )
    
    # Check shapes
    assert mim_train.shape[1] == 32
    
    # Create model
    model = create_model(cfg)
    
    # Test forward pass
    with torch.no_grad():
        output = model(mim_train[:10])
    
    assert output.shape == (10,)
    
    print("  ✓ MAR pipeline test passed")


def test_baseline_vs_mim():
    """Test baseline (no MIM) vs MIM input dimensions"""
    print("Testing Baseline vs MIM input dimensions...")
    
    # Test Baseline
    cfg_baseline = OmegaConf.create({
        "model": {"name": "mlp", "hidden_dims": [32], "dropout": 0.1, "use_batch_norm": False},
        "experiment": {"use_mim": False},
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
    })
    
    model_baseline = create_model(cfg_baseline)
    
    # Test input with 16 features (baseline)
    X_baseline = torch.randn(10, 16)
    with torch.no_grad():
        out_baseline = model_baseline(X_baseline)
    assert out_baseline.shape == (10,)
    
    # Test MIM
    cfg_mim = OmegaConf.create({
        "model": {"name": "mlp", "hidden_dims": [32], "dropout": 0.1, "use_batch_norm": False},
        "experiment": {"use_mim": True},
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
    })
    
    model_mim = create_model(cfg_mim)
    
    # Test input with 32 features (MIM)
    X_mim = torch.randn(10, 32)
    with torch.no_grad():
        out_mim = model_mim(X_mim)
    assert out_mim.shape == (10,)
    
    print("  ✓ Baseline vs MIM test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Full Pipeline Integration Tests")
    print("=" * 60)
    
    test_mcar_pipeline()
    test_mar_pipeline()
    test_baseline_vs_mim()
    
    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
