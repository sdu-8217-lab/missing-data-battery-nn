"""SSL 冒烟：Masked Feature Reconstruction on XJTU 3C

目标：验证在 XJTU 3C 这么小的数据集 (~2000 循环) 上，
用 GraphMIM 骨架做 masked reconstruction 预训练，
val recon loss 能否显著下降。

判断标准：
- 若 val_recon_loss 在 20 epoch 内下降到初始值的 <30%（说明学到了）→ SSL 方向可行
- 若下降 <50% 或不下降 → SSL 在小数据上没用，应该改走物理约束路径

预训练目标：
    输入 x_full ∈ R^{N x 16}, 随机 mask p=0.5 得到 x_masked
    encoder(x_masked, mask) → 完整表示
    decoder(表示) → 重建 x_full
    loss = MSE(recon, x_full) [only on masked positions]

encoder: 复用 GraphMIM 的 GNN + 组池化
decoder: 一个 MLP 头（H → 16）
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.dataset_registry import get_default_feature_cols, get_target_col
from src.models.gnn_missing import FeatureGraphImputer


class MaskedReconstructionModel(nn.Module):
    """GraphMIM encoder + reconstruction head."""

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 num_layers: int = 2, group_ids=None, dropout: float = 0.1):
        super().__init__()
        self.encoder = FeatureGraphImputer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            group_ids=group_ids,
            dropout=dropout,
        )
        # 用 encoder 自带的 imputation_head 就是重建头
        # 我们直接调用 encoder.forward 即可获得 completed 特征

    def forward(self, x_val, mask):
        # completed: 观测位置保留、缺失位置由 GNN 预测
        completed = self.encoder(x_val, mask)
        return completed


def make_mask(shape, p, generator=None):
    """1 = missing, 0 = observed。"""
    if generator is None:
        return (torch.rand(shape) < p).float()
    return (torch.rand(shape, generator=generator) < p).float()


def masked_mse(pred, target, mask, eps=1e-8):
    """只在 mask=1 位置计算 MSE。"""
    diff = (pred - target) ** 2
    weighted = diff * mask
    return weighted.sum() / (mask.sum() + eps)


def main():
    device = 'cpu'   # 不抢 GPU
    torch.manual_seed(42)

    # 加载 XJTU 3C
    feature_cols = get_default_feature_cols('XJTU')
    target_col = get_target_col('XJTU')
    loader = DatasetLoaderFactory.create_loader('./data/XJTU data', 'XJTU', '3C')
    data = loader.prepare_data(feature_cols, target_col, random_seed=42)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)

    print(f"[DATA] X_train: {X_train.shape}, X_val: {X_val.shape}")

    D = X_train.shape[1]
    # 与 experiment_config 一致的 group_ids
    group_ids = [
        1 if ('current' in col.lower() or 'cv ' in col.lower()) else 0
        for col in feature_cols
    ]
    print(f"[DATA] group_ids: {group_ids}")

    model = MaskedReconstructionModel(
        input_dim=D, hidden_dim=32, num_layers=2,
        group_ids=group_ids, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] params: {n_params:,}")

    # DataLoader
    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    mask_p = 0.5

    # 记录初始 val loss（未训练前）
    model.eval()
    with torch.no_grad():
        init_losses = []
        for (xb,) in val_loader:
            xb = xb.to(device)
            mask = make_mask(xb.shape, mask_p)
            x_masked = xb * (1 - mask)
            pred = model(x_masked, mask)
            init_losses.append(masked_mse(pred, xb, mask).item())
        init_val_loss = float(np.mean(init_losses))
    print(f"\n[INIT] val_recon_loss = {init_val_loss:.4f}\n")

    print("epoch | train_loss | val_loss  | val/init")
    print("-" * 55)
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            mask = make_mask(xb.shape, mask_p)
            x_masked = xb * (1 - mask)
            pred = model(x_masked, mask)
            loss = masked_mse(pred, xb, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for (xb,) in val_loader:
                xb = xb.to(device)
                mask = make_mask(xb.shape, mask_p)
                x_masked = xb * (1 - mask)
                pred = model(x_masked, mask)
                val_losses.append(masked_mse(pred, xb, mask).item())
            val_loss = float(np.mean(val_losses))
        best_val = min(best_val, val_loss)
        ratio = val_loss / init_val_loss

        print(f"{epoch:5d} | {train_loss:.4f}    | {val_loss:.4f}   | {ratio:.3f}")

    # 判断
    final_ratio = best_val / init_val_loss
    print("\n" + "=" * 55)
    print(f"[FINAL] best_val_loss / init_val_loss = {final_ratio:.3f}")
    if final_ratio < 0.3:
        print("[DECISION] 强下降 (<30%)：SSL 方向可行 ✅")
    elif final_ratio < 0.5:
        print("[DECISION] 中等下降 (30-50%)：SSL 有效果但收益有限 ⚠️")
    elif final_ratio < 0.8:
        print("[DECISION] 弱下降 (50-80%)：SSL 收益不足以支撑独立论文 ⚠️")
    else:
        print("[DECISION] 几乎不下降 (>80%)：SSL 在此数据上无效 ❌")


if __name__ == '__main__':
    main()
