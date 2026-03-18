#!/usr/bin/env python3
"""
关键路径测试 - 最小测试集

测试最核心的功能，确保代码正确性。
遵循奥卡姆剃刀原则：只测试最关键的部分。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import tempfile
import os

from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.cnn1d import CNN1D
from src.utils.seed_manager import set_seed


def test_seed_reproducibility():
    """测试随机种子可复现性"""
    set_seed(42)
    r1 = torch.rand(5)
    
    set_seed(42)
    r2 = torch.rand(5)
    
    assert torch.allclose(r1, r2), "Random seed not reproducible"
    print("✓ Seed reproducibility test passed")


def test_model_forward_dimensions():
    """测试模型前向传播维度正确"""
    batch_size = 10
    
    # MLP: [batch, features] -> [batch]
    mlp = MLP(input_dim=16)
    x_mlp = torch.randn(batch_size, 16)
    y_mlp = mlp(x_mlp)
    assert y_mlp.shape == (batch_size,), f"MLP output shape wrong: {y_mlp.shape}"
    
    # LSTM: [batch, seq, features] -> [batch]
    lstm = LSTM(input_dim=16)
    x_lstm = torch.randn(batch_size, 1, 16)
    y_lstm = lstm(x_lstm)
    assert y_lstm.shape == (batch_size,), f"LSTM output shape wrong: {y_lstm.shape}"
    
    # CNN: [batch, seq, features] -> [batch]
    cnn = CNN1D(input_dim=16)
    x_cnn = torch.randn(batch_size, 1, 16)
    y_cnn = cnn(x_cnn)
    assert y_cnn.shape == (batch_size,), f"CNN output shape wrong: {y_cnn.shape}"
    
    print("✓ Model forward dimension test passed")


def test_model_save_load():
    """测试模型保存和加载"""
    model = MLP(input_dim=16)
    x = torch.randn(5, 16)
    
    # 获取保存前的输出
    model.eval()
    with torch.no_grad():
        y_before = model(x)
    
    # 保存
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_param': 42
    }, temp_path)
    
    # 加载到新模型
    model_new = MLP(input_dim=16)
    checkpoint = torch.load(temp_path, weights_only=False)
    model_new.load_state_dict(checkpoint['model_state_dict'])
    
    # 验证输出一致
    model_new.eval()
    with torch.no_grad():
        y_after = model_new(x)
    
    assert torch.allclose(y_before, y_after), "Model save/load changed output"
    assert checkpoint['test_param'] == 42, "Additional params not saved"
    
    # 清理
    os.unlink(temp_path)
    
    print("✓ Model save/load test passed")


def test_standardization():
    """测试标准化逻辑"""
    X_train = np.random.randn(100, 16).astype(np.float32)
    
    # 计算标准化参数
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0
    
    # 标准化训练集
    X_train_norm = (X_train - train_mean) / train_std
    
    # 验证训练集均值接近0，标准差接近1
    assert np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-6), "Train mean not 0"
    assert np.allclose(X_train_norm.std(axis=0), 1, atol=1e-6), "Train std not 1"
    
    # 验证测试集使用相同参数
    X_test = np.random.randn(50, 16).astype(np.float32)
    X_test_norm = (X_test - train_mean) / train_std
    
    # 测试集不应有特定的均值/方差，但变换应一致
    expected = (X_test[0, 0] - train_mean[0]) / train_std[0]
    assert np.allclose(X_test_norm[0, 0], expected), "Standardization inconsistent"
    
    print("✓ Standardization test passed")


def test_data_split_no_overlap():
    """测试数据分割不重叠"""
    # 模拟电池ID
    np.random.seed(42)
    batteries = np.array([f'B{i:02d}' for i in range(10)])
    np.random.shuffle(batteries)
    
    n_train = 5
    n_val = 2
    
    train_bats = batteries[:n_train]
    val_bats = batteries[n_train:n_train + n_val]
    test_bats = batteries[n_train + n_val:]
    
    # 验证不重叠
    assert len(set(train_bats) & set(val_bats)) == 0, "Train/Val overlap"
    assert len(set(train_bats) & set(test_bats)) == 0, "Train/Test overlap"
    assert len(set(val_bats) & set(test_bats)) == 0, "Val/Test overlap"
    
    # 验证覆盖全部
    all_bats = set(train_bats) | set(val_bats) | set(test_bats)
    assert len(all_bats) == len(batteries), "Not all batteries assigned"
    
    print("✓ Data split no-overlap test passed")


def test_mim_dimension():
    """测试MIM维度拼接正确"""
    batch_size = 10
    features = 16
    
    # 模拟数据
    X = torch.randn(batch_size, features)
    mask = torch.randint(0, 2, (batch_size, features)).float()
    
    # MIM拼接
    X_mim = torch.cat([X, mask], dim=1)
    
    assert X_mim.shape == (batch_size, features * 2), f"MIM shape wrong: {X_mim.shape}"
    
    # 验证前半部分是原始数据，后半部分是mask
    assert torch.allclose(X_mim[:, :features], X), "MIM first half not X"
    assert torch.allclose(X_mim[:, features:], mask), "MIM second half not mask"
    
    print("✓ MIM dimension test passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running Critical Path Tests")
    print("=" * 60)
    
    try:
        test_seed_reproducibility()
        test_model_forward_dimensions()
        test_model_save_load()
        test_standardization()
        test_data_split_no_overlap()
        test_mim_dimension()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
