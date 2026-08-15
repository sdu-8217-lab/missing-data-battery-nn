"""SAITSModel 冒烟测试"""
import importlib.util
import sys
from pathlib import Path

import torch

# 直接加载 saits_model.py，避免触发 models 包的相对导入链
project_root = Path(__file__).resolve().parents[1]
saits_path = project_root / "src" / "models" / "saits_model.py"

spec = importlib.util.spec_from_file_location("saits_model", saits_path)
saits_module = importlib.util.module_from_spec(spec)
sys.modules["saits_model"] = saits_module
spec.loader.exec_module(saits_module)

SAITSModel = saits_module.SAITSModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SAITSModel(
        input_dim=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    x = torch.randn(4, 5, 16, device=device)
    out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")

    assert out.shape == torch.Size([4]), f"Expected output shape [4], got {out.shape}"
    print("SAITSModel smoke test passed!")


if __name__ == "__main__":
    main()
