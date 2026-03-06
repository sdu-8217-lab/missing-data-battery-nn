"""
Test missing data generators (MCAR and MAR)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.missing_data.mcar import simulate_mcar
from src.missing_data.mar import simulate_mar


def test_simulate_mcar_shape():
    """Test MCAR output shapes"""
    X = torch.randn(100, 16)
    X_imp, mask, mim_input = simulate_mcar(X, missing_rate=0.3, seed=42)
    
    assert X_imp.shape == (100, 16)
    assert mask.shape == (100, 16)
    assert mim_input.shape == (100, 32)  # 16 + 16


def test_simulate_mcar_mask_binary():
    """Test MCAR mask is binary"""
    X = torch.randn(50, 10)
    _, mask, _ = simulate_mcar(X, missing_rate=0.5, seed=42)
    
    # Mask should only contain 0 or 1
    unique_values = torch.unique(mask)
    assert torch.all((unique_values == 0) | (unique_values == 1))


def test_simulate_mcar_missing_rate():
    """Test MCAR actual missing rate is close to target"""
    X = torch.randn(1000, 20)
    target_mr = 0.3
    _, mask, _ = simulate_mcar(X, missing_rate=target_mr, seed=42)
    
    actual_mr = 1.0 - mask.mean().item()
    # Allow 5% tolerance
    assert abs(actual_mr - target_mr) < 0.05


def test_simulate_mcar_reproducibility():
    """Test MCAR is reproducible with same seed"""
    X = torch.randn(100, 16)
    
    _, mask1, _ = simulate_mcar(X, missing_rate=0.3, seed=42)
    _, mask2, _ = simulate_mcar(X, missing_rate=0.3, seed=42)
    
    assert torch.allclose(mask1, mask2)


def test_simulate_mar_shape():
    """Test MAR output shapes"""
    X = torch.randn(100, 16)
    sohs = torch.rand(100)
    
    X_imp, mask, mim_input = simulate_mar(
        X, sohs, missing_rate=0.3, alpha=None, beta=2.0, gamma=0.05, seed=42
    )
    
    assert X_imp.shape == (100, 16)
    assert mask.shape == (100, 16)
    assert mim_input.shape == (100, 32)


def test_simulate_mar_property():
    """Test MAR property: lower SOH -> higher missing rate"""
    N = 1000
    X = torch.randn(N, 16)
    # SOH from 1.0 (new) to 0.6 (old)
    sohs = torch.linspace(1.0, 0.6, N)
    
    _, mask, _ = simulate_mar(
        X, sohs, missing_rate=0.3, alpha=None, beta=2.0, gamma=0.05, seed=42
    )
    
    # Calculate missing rate for high SOH (>0.9) vs low SOH (<0.7)
    high_soh_mask = sohs > 0.9
    low_soh_mask = sohs < 0.7
    
    mr_high = 1.0 - mask[high_soh_mask].mean().item()
    mr_low = 1.0 - mask[low_soh_mask].mean().item()
    
    # Low SOH should have higher missing rate
    assert mr_low > mr_high, f"Low SOH MR ({mr_low:.3f}) should be > High SOH MR ({mr_high:.3f})"


def test_simulate_mar_imputation():
    """Test MAR mean imputation"""
    X = torch.randn(100, 16)
    sohs = torch.rand(100)
    
    X_imp, mask, _ = simulate_mar(
        X, sohs, missing_rate=0.5, seed=42
    )
    
    # For each feature, check that imputed values equal the mean of observed values
    for j in range(16):
        feature = X_imp[:, j]
        mask_j = mask[:, j]
        
        if mask_j.sum() > 0 and mask_j.sum() < len(mask_j):
            observed_mean = feature[mask_j == 1].mean()
            imputed_values = feature[mask_j == 0]
            
            # All imputed values should be close to the observed mean
            assert torch.allclose(imputed_values, observed_mean, atol=1e-5)


if __name__ == "__main__":
    print("Running missing generator tests...")
    
    test_simulate_mcar_shape()
    print("✓ MCAR shape test passed")
    
    test_simulate_mcar_mask_binary()
    print("✓ MCAR binary mask test passed")
    
    test_simulate_mcar_missing_rate()
    print("✓ MCAR missing rate test passed")
    
    test_simulate_mcar_reproducibility()
    print("✓ MCAR reproducibility test passed")
    
    test_simulate_mar_shape()
    print("✓ MAR shape test passed")
    
    test_simulate_mar_property()
    print("✓ MAR property test passed")
    
    test_simulate_mar_imputation()
    print("✓ MAR imputation test passed")
    
    print("\nAll tests passed!")
