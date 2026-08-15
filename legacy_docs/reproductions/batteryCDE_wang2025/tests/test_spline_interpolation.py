"""自然三次样条插值测试"""
import os
import sys

import torch
import torchcde

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def test_natural_cubic_spline_with_nan():
    """验证 torchcde 自然三次样条能处理含 NaN 路径并给出连续表示。"""
    batch_size = 2
    seq_len = 10
    input_dim = 3
    x = torch.randn(batch_size, seq_len, input_dim)
    x[:, 3:6, :] = float("nan")  # 连续缺失

    coeffs = torchcde.natural_cubic_coeffs(x)
    spline = torchcde.CubicSpline(coeffs)

    t_query = torch.linspace(0, seq_len - 1, seq_len)
    X_t = spline.evaluate(t_query)

    assert X_t.shape == (batch_size, seq_len, input_dim)
    # 插值后不应再出现 NaN（torchcde 会用相邻观测填充）
    assert torch.isfinite(X_t).all(), "Spline evaluation contains NaN"
    print(f"test_natural_cubic_spline_with_nan passed: interpolated shape {X_t.shape}")


def test_spline_derivative():
    """验证样条导数可用（CDE 积分需要）。"""
    x = torch.randn(2, 10, 3)
    coeffs = torchcde.natural_cubic_coeffs(x)
    spline = torchcde.CubicSpline(coeffs)

    t_query = torch.linspace(0, 9, 10)
    dX_t = spline.derivative(t_query)
    assert dX_t.shape == (2, 10, 3)
    assert torch.isfinite(dX_t).all()
    print(f"test_spline_derivative passed: derivative shape {dX_t.shape}")


if __name__ == "__main__":
    test_natural_cubic_spline_with_nan()
    test_spline_derivative()
