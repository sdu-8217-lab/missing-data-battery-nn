import os
import sys
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

# 保证能 import src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.loader import load_dataset


def main():
    print("=== 数据加载冒烟测试（XJTU） ===")

    # 初始化 Hydra（只用 configs/config.yaml）
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="config", overrides=["data=xjtu"])

    data = load_dataset(cfg)
    for k, v in data.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    assert isinstance(data["X_train"], torch.Tensor)
    assert isinstance(data["y_train"], torch.Tensor)
    assert data["X_train"].ndim == 2
    assert data["y_train"].ndim == 1

    print("[OK] 数据加载模块冒烟测试通过")


if __name__ == "__main__":
    main()
