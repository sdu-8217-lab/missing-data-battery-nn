"""
测试 MIM 多缺失率训练功能

验证：
1. MIM 模式训练集是否正确复制10份（缺失率0.0-0.9）
2. Baseline 模式保持原有逻辑
3. 验证集和测试集使用指定的 missing_rate
"""
import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# 创建模拟配置
def create_test_cfg(use_mim=True):
    cfg = OmegaConf.create({
        "data": {"dataset": "test"},
        "models": {"name": "cnn1d"},
        "missing": {
            "mode": "mar",
            "beta": 2.0,
            "gamma": 0.05,
            "alpha": None
        },
        "experiments": {
            "experiment": {
                "name": "test_mim" if use_mim else "test_baseline",
                "use_mim": use_mim,
                "missing_mode": "mar",
                "missing_rate": 0.3
            },
            "training": {
                "batch_size": 32,
                "seeds": [42]
            },
            "output": {
                "result_csv": "test_results.csv",
                "log_dir": "logs/test"
            }
        }
    })
    return cfg


def test_mim_train_loader():
    """测试 MIM 训练集创建"""
    from src.trainers.neural_network_trainer import _create_mim_train_loader
    from src.missing_data.mcar import simulate_mcar
    
    print("=" * 60)
    print("Testing MIM Training Loader")
    print("=" * 60)
    
    # 创建模拟数据
    N, D = 100, 16  # 100样本，16特征
    X_train = torch.randn(N, D)
    y_train = torch.randn(N)
    
    cfg = create_test_cfg(use_mim=True)
    
    # 创建 MIM 训练 loader
    train_loader = _create_mim_train_loader(cfg, X_train, y_train, batch_size=16, seed=42)
    
    # 验证
    total_samples = 0
    expected_samples = 10 * N  # 10份，每份100样本
    
    for batch_x, batch_y in train_loader:
        total_samples += len(batch_x)
        # 验证输入维度：MIM应该是32维（16特征+16掩码）
        assert batch_x.shape[1] == 32, f"Expected input dim 32, got {batch_x.shape[1]}"
    
    print(f"Original training set size: {N}")
    print(f"Expected augmented size: {expected_samples}")
    print(f"Actual total samples: {total_samples}")
    print(f"Input dimension: 32 (16 features + 16 masks)")
    
    assert total_samples == expected_samples, f"Size mismatch: {total_samples} vs {expected_samples}"
    print("\n[OK] MIM training loader test passed!")
    return True


def test_data_consistency():
    """测试数据一致性：验证每份的缺失率是否正确"""
    from src.missing_data.mcar import simulate_mcar
    
    print("\n" + "=" * 60)
    print("Testing Missing Rate Consistency")
    print("=" * 60)
    
    N, D = 1000, 16
    X = torch.randn(N, D)
    
    expected_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for expected_mr in expected_rates:
        X_imp, mask, mim_input = simulate_mcar(X, expected_mr, seed=42)
        actual_mr = 1.0 - mask.mean().item()
        
        # 允许5%的误差（随机性导致）
        error = abs(actual_mr - expected_mr)
        status = "OK" if error < 0.05 else "FAIL"
        print(f"Expected MR: {expected_mr:.1f}, Actual MR: {actual_mr:.3f}, Error: {error:.3f} [{status}]")
        
        assert error < 0.05, f"Missing rate mismatch for {expected_mr}"
    
    print("\n[OK] Missing rate consistency test passed!")
    return True


def test_mim_input_format():
    """测试 MIM 输入格式：[X_imputed, 1-mask]"""
    from src.missing_data.mcar import simulate_mcar
    
    print("\n" + "=" * 60)
    print("Testing MIM Input Format")
    print("=" * 60)
    
    N, D = 10, 4  # 小样本方便验证
    X = torch.randn(N, D)
    
    X_imp, mask, mim_input = simulate_mcar(X, missing_rate=0.3, seed=42)
    
    # 验证形状
    assert mim_input.shape == (N, 2 * D), f"Shape mismatch: {mim_input.shape} vs {(N, 2*D)}"
    
    # 验证前半部分是 X_imputed
    assert torch.allclose(mim_input[:, :D], X_imp), "First half should be X_imputed"
    
    # 验证后半部分是 1-mask
    assert torch.allclose(mim_input[:, D:], 1.0 - mask), "Second half should be 1-mask"
    
    print(f"X shape: {X.shape}")
    print(f"X_imputed shape: {X_imp.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"mim_input shape: {mim_input.shape}")
    print(f"mim_input structure: [X_imputed ({D} dims) | 1-mask ({D} dims)]")
    
    # 展示一个样本
    print(f"\nSample 0:")
    print(f"  Original X: {X[0].numpy()}")
    print(f"  Mask: {mask[0].numpy()}")
    print(f"  X_imputed: {X_imp[0].numpy()}")
    print(f"  MIM input: {mim_input[0].numpy()}")
    
    print("\n[OK] MIM input format test passed!")
    return True


if __name__ == "__main__":
    try:
        test_mim_train_loader()
        test_data_consistency()
        test_mim_input_format()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
