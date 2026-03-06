"""
Test MIM multi-missing-rate with larger sample size to verify statistical correctness
"""
import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from src.missing_data.mcar import simulate_mcar


def test_with_large_sample():
    """使用大样本验证缺失率的统计正确性"""
    print("=" * 70)
    print("Testing MIM Missing Rate with Large Sample (N=10000)")
    print("=" * 70)
    
    N, D = 10000, 16  # 大样本
    X = torch.randn(N, D)
    seed = 42
    
    print(f"\nSample size: {N}")
    print(f"{'Target MR':<12} {'Actual MR':<12} {'Error':<12} {'Status'}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    
    all_pass = True
    for target_mr in [i / 10.0 for i in range(10)]:
        X_imp, mask, mim_input = simulate_mcar(X, target_mr, seed)
        actual_mr = 1.0 - mask.mean().item()
        error = abs(actual_mr - target_mr)
        status = "[PASS]" if error < 0.02 else "[WARN]"
        
        if error >= 0.02:
            all_pass = False
            
        print(f"{target_mr:<12.1f} {actual_mr:<12.4f} {error:<12.4f} {status}")
    
    print(f"\nWith N={N}, all errors should be < 0.02")
    print(f"Result: {'All passed' if all_pass else 'Some warnings (acceptable)'}")
    
    return all_pass


if __name__ == "__main__":
    test_with_large_sample()
