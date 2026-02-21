"""数据加载主入口

提供根据配置加载数据集的统一接口。
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from .features import build_features
from .preprocessing import clean_dataframe
from .splits import train_val_test_split


def load_dataset(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """主入口：根据配置加载数据集

    支持的数据集:
        - xjtu: XJTU 电池数据集
        - hust: HUST 电池数据集 (占位)
        - mit: MIT 电池数据集 (占位)

    Args:
        cfg: 配置对象，必须包含 data.dataset, data.data_dir 等配置

    Returns:
        字典包含划分后的数据集（见 train_val_test_split 返回说明）

    Raises:
        NotImplementedError: 如果请求的数据集尚未实现
        FileNotFoundError: 如果数据文件不存在
        ValueError: 如果配置无效或数据为空
    """
    dataset = cfg.data.get("dataset", None)
    if dataset is None:
        raise ValueError("cfg.data.dataset must be specified")

    if dataset == "xjtu":
        return _load_xjtu(cfg)
    elif dataset == "hust":
        return _load_hust(cfg)
    elif dataset == "mit":
        return _load_mit(cfg)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported yet")


def _load_xjtu(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 XJTU 数据集

    从指定目录加载匹配模式的 CSV 文件，合并所有电池数据后划分。

    Args:
        cfg: 配置对象，需要包含：
            - data.data_dir: 数据目录
            - data.batch: 批次名称（如 3C, 2C 等）

    Returns:
        划分后的数据集字典

    Raises:
        FileNotFoundError: 如果找不到匹配的数据文件
        ValueError: 如果所有文件处理后数据为空
    """
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch", None)

    if batch is None:
        raise ValueError("cfg.data.batch must be specified for XJTU dataset")

    pattern = f"{batch}_battery-*.csv"
    files = sorted(data_dir.glob(pattern))

    if not files:
        available_files = list(data_dir.glob("*.csv"))[:10]  # 显示前10个文件
        available_str = "\n  - ".join([f.name for f in available_files])
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' in {data_dir}\n"
            f"Available CSV files:\n  - {available_str}"
        )

    X_list, y_list = [], []

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = clean_dataframe(df, cfg)

            if df.empty:
                continue

            X, y_soh = build_features(df, cfg)
            X_list.append(X)
            y_list.append(y_soh)
        except Exception as e:
            # 记录错误但继续处理其他文件
            print(f"Warning: Failed to process {fp.name}: {e}")
            continue

    if not X_list:
        raise ValueError(
            f"No valid data loaded from {len(files)} files. "
            "Please check data format and configuration."
        )

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return train_val_test_split(X_all, y_all, cfg)


def _load_hust(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 HUST 数据集 (占位实现)

    Args:
        cfg: 配置对象

    Raises:
        NotImplementedError: 始终抛出此异常
    """
    raise NotImplementedError(
        "HUST dataset loader not implemented yet. "
        "Please use 'xjtu' dataset or implement HUST loader."
    )


def _load_mit(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 MIT 数据集 (占位实现)

    Args:
        cfg: 配置对象

    Raises:
        NotImplementedError: 始终抛出此异常
    """
    raise NotImplementedError(
        "MIT dataset loader not implemented yet. "
        "Please use 'xjtu' dataset or implement MIT loader."
    )
