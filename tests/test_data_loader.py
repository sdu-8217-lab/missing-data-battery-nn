"""
Test data loading and preprocessing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.preprocessing import clean_dataframe, standardize_features
from src.data.features import build_features
from src.data.splits import train_val_test_split


def test_standardize_features():
    """Test feature standardization"""
    X = np.random.randn(100, 5) * 2 + 3  # mean=3, std=2
    X_scaled, mean, std = standardize_features(X)
    
    # Check mean is approximately 0 and std is approximately 1
    assert np.abs(X_scaled.mean()) < 1e-6
    assert np.abs(X_scaled.std() - 1.0) < 1e-6
    
    # Check mean and std are correct
    assert np.allclose(mean, 3.0, atol=1e-6)
    assert np.allclose(std, 2.0, atol=1e-6)


def test_train_val_test_split():
    """Test data splitting"""
    X = np.random.randn(1000, 16)
    y = np.random.rand(1000)
    
    cfg = OmegaConf.create({
        "data": {
            "split": {
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": 42
            },
            "preprocessing": {
                "standardize": False
            }
        }
    })
    
    data = train_val_test_split(X, y, cfg)
    
    # Check all keys exist
    assert "X_train" in data
    assert "y_train" in data
    assert "X_val" in data
    assert "y_val" in data
    assert "X_test" in data
    assert "y_test" in data
    
    # Check types
    assert isinstance(data["X_train"], torch.Tensor)
    assert isinstance(data["y_train"], torch.Tensor)
    
    # Check sizes
    assert len(data["X_train"]) == 640  # 1000 * 0.8 * 0.8
    assert len(data["X_val"]) == 160    # 1000 * 0.8 * 0.2
    assert len(data["X_test"]) == 200   # 1000 * 0.2


def test_build_features():
    """Test feature building from DataFrame"""
    # Create sample DataFrame
    df = pd.DataFrame({
        "voltage_mean": np.random.randn(100),
        "voltage_std": np.random.rand(100),
        "capacity": np.random.rand(100) * 2 + 1,  # 1-3 Ah
    })
    
    cfg = OmegaConf.create({
        "data": {
            "features": ["voltage_mean", "voltage_std"],
            "target": "capacity",
        }
    })
    
    X, y = build_features(df, cfg)
    
    # Check shapes
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    
    # Check types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_clean_dataframe():
    """Test DataFrame cleaning"""
    # Create DataFrame with issues
    df = pd.DataFrame({
        "a": [1, 2, np.inf, 4, 5],
        "b": [1, 2, 3, np.nan, 5],
    })
    
    cfg = OmegaConf.create({
        "data": {
            "preprocessing": {
                "remove_outliers": False,  # Skip for this test
                "fill_na": True,
                "fill_na_method": "mean",
            }
        }
    })
    
    df_clean = clean_dataframe(df, cfg)
    
    # Check no inf values
    assert not np.isinf(df_clean.values).any()
    
    # Check no NaN values
    assert not df_clean.isna().any().any()


if __name__ == "__main__":
    print("Running data loader tests...")
    
    test_standardize_features()
    print("✓ Standardize features test passed")
    
    test_train_val_test_split()
    print("✓ Train/val/test split test passed")
    
    test_build_features()
    print("✓ Build features test passed")
    
    test_clean_dataframe()
    print("✓ Clean DataFrame test passed")
    
    print("\nAll tests passed!")
