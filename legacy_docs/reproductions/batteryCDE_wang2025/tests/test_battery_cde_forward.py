"""BatteryCDE 前向传播形状测试"""
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from reproductions.batteryCDE_wang2025.src.models.battery_cde import BatteryCDEModel  # noqa: E402


def test_battery_cde_forward():
    batch_size = 4
    seq_len = 20
    input_dim = 3
    x = torch.randn(batch_size, seq_len, input_dim)

    model = BatteryCDEModel(
        input_dim=input_dim,
        hidden_dim=16,
        attention_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    out = model(x)

    assert out.shape == (batch_size,), f"Expected shape ({batch_size},), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    print(f"test_battery_cde_forward passed: output shape {out.shape}")


def test_battery_cde_with_missing():
    """验证模型能处理含 NaN 的输入（torchcde 样条插值）。"""
    batch_size = 2
    seq_len = 20
    input_dim = 3
    x = torch.randn(batch_size, seq_len, input_dim)
    x[:, 5:10, :] = float("nan")  # 制造连续缺失

    model = BatteryCDEModel(
        input_dim=input_dim,
        hidden_dim=16,
        attention_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    out = model(x)
    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all(), "Output contains non-finite values despite NaN input"
    print(f"test_battery_cde_with_missing passed: output shape {out.shape}")


def test_battery_cde_4d_input():
    """验证 4D 输入 [B, history_len, seq_len, input_dim] 可正确处理。"""
    batch_size = 2
    history_len = 5
    seq_len = 20
    input_dim = 3
    x = torch.randn(batch_size, history_len, seq_len, input_dim)
    model = BatteryCDEModel(
        input_dim=input_dim,
        hidden_dim=16,
        attention_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    out = model(x)
    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()
    print(f"test_battery_cde_4d_input passed: output shape {out.shape}")


if __name__ == "__main__":
    test_battery_cde_forward()
    test_battery_cde_with_missing()
    test_battery_cde_4d_input()
