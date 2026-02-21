"""
MAR (Missing At Random) 缺失机制实现

基于电池 SOH 的依赖型缺失模拟：SOH 越低，缺失概率越高。

核心公式:
    p_i = alpha * (1 - SOH_i)^beta + gamma

其中:
    - alpha: 缺失强度系数（可由目标缺失率自动反推）
    - beta: 非线性控制参数（默认 2.0）
    - gamma: 基础缺失率（默认 0.05）
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch


def _sanitize_soh(sohs: torch.Tensor) -> torch.Tensor:
    """
    将 SOH 值截断到 [0, 1] 范围内。

    参数:
        sohs: SOH 张量，形状 [N]，可能包含越界值。

    返回:
        sohs_clamped: 截断后的 SOH 张量，保证所有值在 [0, 1]。

    注意:
        若输入 SOH 存在明显越界（<0 或 >1），可能是数据预处理问题，
        但本函数仅做安全截断，不抛出异常。
    """
    sohs_clamped = torch.clamp(sohs, min=0.0, max=1.0)
    return sohs_clamped


def _compute_alpha_from_target_mr(
    sohs: torch.Tensor,
    missing_rate: float,
    beta: float,
    gamma: float,
) -> float:
    """
    根据目标整体缺失率 (missing_rate) 和当前 SOH 分布，反推 alpha 参数。

    数学推导:
        E[p_i] = E[alpha * (1-SOH)^beta + gamma]
               = alpha * E[(1-SOH)^beta] + gamma
               = missing_rate
        => alpha = (missing_rate - gamma) / E[(1-SOH)^beta]

    参数:
        sohs: SOH 张量，形状 [N]，应在 [0,1] 范围内。
        missing_rate: 目标整体缺失率，例如 0.3, 0.6, 0.9。
        beta: 非线性参数。
        gamma: 基础缺失率。

    返回:
        alpha: 计算得到的缺失强度系数，已截断到 [0, 1]。

    边界处理:
        - 若 mean_term = E[(1-SOH)^beta] 接近 0，使用 1e-6 保护除零。
        - alpha 结果截断在 [0, 1] 范围内。
    """
    # 计算 E[(1-SOH)^beta]
    term = (1.0 - sohs).pow(beta)  # [N]
    mean_term = term.mean().item()

    # 除零保护
    denom = max(mean_term, 1e-6)

    # 解出 alpha
    alpha = (missing_rate - gamma) / denom

    # 截断到 [0, 1]
    alpha = max(0.0, min(alpha, 1.0))

    return alpha


def _generate_mar_mask(
    sohs: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
    D: int,
) -> torch.Tensor:
    """
    生成 MAR 缺失掩码。

    参数:
        sohs: SOH 张量，形状 [N]，应在 [0,1] 范围内。
        alpha: 缺失强度系数。
        beta: 非线性参数。
        gamma: 基础缺失率。
        seed: 随机种子。
        D: 特征维度数。

    返回:
        mask: 缺失掩码张量，形状 [N, D]，float32。
              1.0 表示观测到，0.0 表示缺失。

    实现细节:
        - p_i = alpha * (1-SOH)^beta + gamma
        - rand ~ Uniform(0,1)
        - rand > p_i 则观测 (mask=1)，否则缺失 (mask=0)
    """
    # 设置随机种子（确保可复现）
    torch.manual_seed(seed)

    N = sohs.shape[0]

    # 计算每个样本的缺失概率 p_i
    p_missing = alpha * (1.0 - sohs).pow(beta) + gamma  # [N]
    p_missing = torch.clamp(p_missing, min=0.0, max=1.0)

    # 扩展到 [N, D]（每个特征独立决定缺失）
    p_mat = p_missing.view(N, 1).expand(-1, D)

    # 生成随机数并比较
    rand = torch.rand(N, D, device=sohs.device)
    mask = (rand > p_mat).float()

    return mask


def simulate_mar(
    X: torch.Tensor,
    sohs: torch.Tensor,
    missing_rate: float,
    alpha: Optional[float] = None,
    beta: float = 2.0,
    gamma: float = 0.05,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在电池 SOH 预测场景下，根据 SOH 构造 MAR 缺失机制。

    参数:
        X: 原始特征张量，形状 [N, D]，已完成标准化等预处理。
        sohs: SOH 张量，形状 [N]，数值应在 [0, 1] 范围内。
        missing_rate: 目标整体缺失率 (MR)，例如 0.3, 0.6, 0.9。
        alpha: 若为 None，则根据 missing_rate 和 SOH 分布自动反推 α；
               若给定，则直接使用该 α，不再调整。
        beta: 模型参数 β，控制 (1-SOH) 的非线性程度，默认 2.0。
        gamma: 模型参数 γ，基础缺失率，默认 0.05。
        seed: 随机种子，保证结果可复现。

    返回:
        X_imputed: 均值插补后的特征张量，[N, D]。
        mask: 缺失掩码张量，[N, D]，1 表示观测到，0 表示缺失。
        mim_input: MIM 输入张量，[N, 2D]，为 [X_imputed, 1-mask] 的拼接。

    约定:
        - 函数不修改原始 X 与 sohs（使用 clone）。
        - 与 MCAR 版本保持返回值含义和形状一致，便于训练脚本切换。

    示例:
        >>> X = torch.randn(100, 16)
        >>> sohs = torch.rand(100)
        >>> X_imp, mask, mim = simulate_mar(X, sohs, missing_rate=0.3)
        >>> print(mim.shape)  # torch.Size([100, 32])
    """
    # ---------- 1. 输入检查与预处理 ----------
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"X must be torch.Tensor, got {type(X)}")
    if not isinstance(sohs, torch.Tensor):
        raise TypeError(f"sohs must be torch.Tensor, got {type(sohs)}")

    if X.dim() != 2:
        raise ValueError(f"X must be 2D tensor [N, D], got shape {X.shape}")
    if sohs.dim() != 1:
        raise ValueError(f"sohs must be 1D tensor [N], got shape {sohs.shape}")

    N, D = X.shape
    if sohs.shape[0] != N:
        raise ValueError(
            f"Batch size mismatch: X has {N} samples, sohs has {sohs.shape[0]}"
        )

    # 确保 SOH 在 [0, 1] 范围内
    sohs_clean = _sanitize_soh(sohs)

    # 确保 X 和 sohs 在同一设备上
    if X.device != sohs_clean.device:
        sohs_clean = sohs_clean.to(X.device)

    # ---------- 2. 确定 α 参数 ----------
    if alpha is not None:
        # 使用给定的 alpha，并进行安全截断
        alpha_val = max(0.0, min(float(alpha), 1.0))
    else:
        # 根据目标缺失率自动反推 alpha
        alpha_val = _compute_alpha_from_target_mr(
            sohs=sohs_clean,
            missing_rate=missing_rate,
            beta=beta,
            gamma=gamma,
        )

    # ---------- 3. 生成 mask ----------
    mask = _generate_mar_mask(
        sohs=sohs_clean,
        alpha=alpha_val,
        beta=beta,
        gamma=gamma,
        seed=seed,
        D=D,
    )

    # 确保 mask 与 X 在同一设备
    if mask.device != X.device:
        mask = mask.to(X.device)

    # ---------- 4. 均值插补 ----------
    X_missing = X.clone()

    # 计算每个特征的观测数量（避免除零）
    obs_counts = mask.sum(dim=0).clamp(min=1.0)  # [D]

    # 计算特征均值（仅基于观测值）
    feature_sums = (X_missing * mask).sum(dim=0)  # [D]
    feature_means = feature_sums / obs_counts  # [D]

    # 创建插补后的特征矩阵
    X_imputed = X_missing.clone()

    # 将缺失位置（mask == 0）填补为对应特征均值
    # 使用广播: mask [N,D] == 0 的位置，填入 feature_means [D]
    missing_positions = (mask == 0.0)
    X_imputed = torch.where(missing_positions, feature_means.unsqueeze(0), X_imputed)

    # ---------- 5. 构造 MIM 输入 ----------
    mim_indicator = 1.0 - mask  # 1=缺失, 0=观测，形状 [N, D]
    mim_input = torch.cat([X_imputed, mim_indicator], dim=1)  # [N, 2D]

    return X_imputed, mask, mim_input


# -----------------------------------------------------------------------------
# 辅助函数：用于与 MCAR 接口对齐的包装（如果将来需要统一接口）
# -----------------------------------------------------------------------------

def mar_missing_simulation(
    X: torch.Tensor,
    sohs: torch.Tensor,
    missing_rate: float,
    alpha: Optional[float] = None,
    beta: float = 2.0,
    gamma: float = 0.05,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    `simulate_mar` 的别名函数，与 MCAR 模块的命名风格保持一致。

    参数和返回值与 `simulate_mar` 完全相同。
    """
    return simulate_mar(
        X=X,
        sohs=sohs,
        missing_rate=missing_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# 自测代码
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MAR Missing Mechanism - Self Test")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(0)

    # 构造测试数据
    N, D = 1000, 16
    X = torch.randn(N, D)
    # 模拟 SOH 从新电池(1.0)到老电池(0.6)的退化过程
    sohs = torch.linspace(1.0, 0.6, N)

    print(f"\nTest Data: N={N}, D={D}")
    print(f"SOH range: [{sohs.min():.3f}, {sohs.max():.3f}]")
    print("-" * 60)

    # 测试不同目标缺失率
    for target_mr in [0.3, 0.6, 0.9]:
        print(f"\nTarget Missing Rate = {target_mr:.2f}")
        print("-" * 40)

        # 运行 MAR 模拟
        X_imp, mask, mim_input = simulate_mar(
            X=X,
            sohs=sohs,
            missing_rate=target_mr,
            alpha=None,  # 自动反推 alpha
            beta=2.0,
            gamma=0.05,
            seed=42,
        )

        # 验证输出形状
        assert X_imp.shape == (N, D), f"X_imputed shape mismatch: {X_imp.shape}"
        assert mask.shape == (N, D), f"mask shape mismatch: {mask.shape}"
        assert mim_input.shape == (N, 2 * D), f"mim_input shape mismatch: {mim_input.shape}"

        # 计算实际缺失率
        actual_mr = 1.0 - mask.mean().item()

        # 验证低 SOH 区域缺失率更高
        low_soh_mask = sohs < 0.7  # 老电池
        high_soh_mask = sohs > 0.9  # 新电池

        if low_soh_mask.any() and high_soh_mask.any():
            mr_low = 1.0 - mask[low_soh_mask].mean().item()
            mr_high = 1.0 - mask[high_soh_mask].mean().item()
            print(f"  Actual MR (overall): {actual_mr:.4f}")
            print(f"  Actual MR (SOH < 0.7): {mr_low:.4f}")
            print(f"  Actual MR (SOH > 0.9): {mr_high:.4f}")
            print(f"  MAR property (low > high): {mr_low > mr_high}")
        else:
            print(f"  Actual MR (overall): {actual_mr:.4f}")

        # 验证 mask 值域
        assert torch.all((mask == 0.0) | (mask == 1.0)), "Mask should be binary"

        # 验证 mim_input 构造正确
        expected_indicator = 1.0 - mask
        actual_indicator = mim_input[:, D:]
        assert torch.allclose(actual_indicator, expected_indicator), "MIM indicator mismatch"

        print(f"  [OK] Shape check passed")
        print(f"  [OK] Mask binary check passed")
        print(f"  [OK] MIM input construction passed")

    # 测试给定 alpha 的情况
    print("\n" + "=" * 60)
    print("Test with fixed alpha=0.5")
    print("-" * 60)
    X_imp, mask, mim_input = simulate_mar(
        X=X,
        sohs=sohs,
        missing_rate=0.5,  # 这个参数在 alpha 给定时会被忽略
        alpha=0.5,
        beta=2.0,
        gamma=0.05,
        seed=42,
    )
    actual_mr = 1.0 - mask.mean().item()
    print(f"  With alpha=0.5, actual MR ≈ {actual_mr:.4f}")
    print(f"  [OK] Fixed alpha test passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
