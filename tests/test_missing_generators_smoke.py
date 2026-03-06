import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.missing_data.mcar import simulate_mcar
from src.missing_data.mar import simulate_mar


def main():
    print("=== MCAR / MAR 缺失机制冒烟测试 ===")

    N, D = 1000, 16
    X = torch.randn(N, D)

    # 构造从 1.0 到 0.6 线性下降的 SOH
    sohs = torch.linspace(1.0, 0.6, N)

    # 1. 测试 MCAR
    mr = 0.3
    X_imp_mcar, mask_mcar, mim_mcar = simulate_mcar(X, missing_rate=mr, seed=42)
    print(f"[MCAR] X_imp: {tuple(X_imp_mcar.shape)}, mask: {tuple(mask_mcar.shape)}, mim: {tuple(mim_mcar.shape)}")
    actual_mr_mcar = 1.0 - mask_mcar.mean().item()
    print(f"[MCAR] target MR={mr}, actual MR≈{actual_mr_mcar:.3f}")

    assert X_imp_mcar.shape == (N, D)
    assert mask_mcar.shape == (N, D)
    assert mim_mcar.shape == (N, 2 * D)

    # 2. 测试 MAR
    X_imp_mar, mask_mar, mim_mar = simulate_mar(
        X, sohs, missing_rate=mr, alpha=None, beta=2.0, gamma=0.05, seed=42
    )
    print(f"[MAR] X_imp: {tuple(X_imp_mar.shape)}, mask: {tuple(mask_mar.shape)}, mim: {tuple(mim_mar.shape)}")
    actual_mr_mar = 1.0 - mask_mar.mean().item()
    print(f"[MAR] target MR={mr}, actual MR≈{actual_mr_mar:.3f}")

    # 划分高 SOH / 低 SOH
    high_idx = sohs > 0.9
    low_idx = sohs < 0.7
    mr_high = 1.0 - mask_mar[high_idx].mean().item()
    mr_low = 1.0 - mask_mar[low_idx].mean().item()
    print(f"[MAR] MR_high(SOH>0.9) ≈ {mr_high:.3f}")
    print(f"[MAR] MR_low(SOH<0.7) ≈ {mr_low:.3f}")
    print(f"[MAR] low > high ? {mr_low > mr_high}")

    assert X_imp_mar.shape == (N, D)
    assert mask_mar.shape == (N, D)
    assert mim_mar.shape == (N, 2 * D)
    assert mr_low > mr_high  # 核心性质

    print("[OK] MCAR / MAR 缺失机制冒烟测试通过")


if __name__ == "__main__":
    main()
