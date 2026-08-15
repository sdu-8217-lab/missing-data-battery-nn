"""GRINModel 烟雾测试"""
import sys
import os

# 将项目根目录加入 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from src.models.grin_model import GRINModel


def run_test(batch_size=4, seq_len=5, original_dim=16, use_mim=False, device='cpu'):
    input_dim = 2 * original_dim if use_mim else original_dim
    model = GRINModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        use_mim=use_mim,
        gnn_hidden=32,
        gnn_layers=2,
        gnn_num_heads=1
    ).to(device)

    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    if use_mim:
        # 后一半是缺失指示器 (1=缺失)
        x[:, :, original_dim:] = (torch.rand(batch_size, seq_len, original_dim, device=device) > 0.7).float()

    out = model(x)
    assert out.shape == (batch_size,), f"期望输出形状 {(batch_size,)}，得到 {out.shape}"
    assert not torch.isnan(out).any(), "输出包含 NaN"
    print(f"use_mim={use_mim}, input={tuple(x.shape)}, output={tuple(out.shape)}, device={device} OK")
    return True


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ok = True
    ok &= run_test(use_mim=False, device=device)
    ok &= run_test(use_mim=True, device=device)

    if ok:
        print("\n所有 GRIN smoke tests 通过！")
    else:
        print("\n存在失败的 smoke test")
        sys.exit(1)


if __name__ == '__main__':
    main()
