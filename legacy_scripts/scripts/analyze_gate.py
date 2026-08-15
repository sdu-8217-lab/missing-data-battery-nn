"""分析已训练 FMG 模型的门控输出

用法示例：
    python scripts/analyze_gate.py \
        --model_path results/uniform_smoke_group/2026.../models/best_model_LSTM-FMG-Uniform_seed100042_....pth \
        --model_name LSTM-FMG-Uniform \
        --missing_pattern group \
        --missing_rate 0.5
"""
import sys
import re
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig
from src.data.dataset_loader import DatasetLoaderFactory
from src.data.datasets import SequenceDataset
from src.models.model_factory import ModelFactory


def parse_seed_from_path(path: str) -> int:
    m = re.search(r'_seed(\d+)_', path)
    if not m:
        raise ValueError(f'无法从路径解析 seed: {path}')
    return int(m.group(1))


def get_model_config(model_name: str):
    """从全局配置中找到对应模型配置"""
    cfg = ExperimentConfig(include_fmg=True, model_filter=[model_name])
    configs = cfg.get_model_configs()
    if not configs:
        raise ValueError(f'找不到模型配置: {model_name}')
    return configs[0]


def main():
    parser = argparse.ArgumentParser(description='FMG gate introspection')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--missing_pattern', type=str, default='group')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--dataset_name', type=str, default='XJTU')
    parser.add_argument('--batch', type=str, default='3C')
    parser.add_argument('--data_dir', type=str, default='./data/XJTU data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = parse_seed_from_path(args.model_path)
    model_config = get_model_config(args.model_name)

    # 加载数据
    loader = DatasetLoaderFactory.create_loader(args.data_dir, args.dataset_name, args.batch)
    data = loader.prepare_data(
        ExperimentConfig().feature_cols,
        ExperimentConfig().target_col,
        test_size=0.25,
        val_size=0.25,
        random_seed=seed,
    )

    # 构建测试集
    test_dataset = SequenceDataset(
        data['X_test'], data['y_test'],
        seq_len=model_config.seq_len,
        missing_rate=args.missing_rate,
        use_mim=False,
        use_fmg=True,
        seed=seed,
        missing_pattern=args.missing_pattern,
        boundaries=data.get('test_boundaries')
    )
    test_loader = DataLoader(test_dataset, batch_size=256)

    # 构建模型
    input_dim = data['X_test'].shape[1]
    model_kwargs = {k: v for k, v in model_config.__dict__.items()
                    if k not in ['name', 'model_type', 'use_mim', 'use_fmg', 'use_curriculum', 'strategy']}
    model = ModelFactory.create_model(
        model_config.model_type,
        input_dim=input_dim,
        use_mim=False,
        use_fmg=True,
        device=device,
        **model_kwargs
    )
    model.load(args.model_path)
    raw_net = model.model  # MissingGateModel

    # 跑一遍 forward 以延迟构建 gate
    with torch.no_grad():
        for x, _ in test_loader:
            _ = raw_net(x.to(device))
            break

    gate_module = raw_net.gate  # FeatureMissingGate
    print(f'加载模型: {args.model_path}')
    print(f'测试 pattern={args.missing_pattern}, missing_rate={args.missing_rate}, seed={seed}')
    print(f'Gate 结构: {gate_module.gate}\n')

    all_g = []
    all_m = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            original_dim = x.shape[-1] // 2
            features = x[..., :original_dim]
            missing_mask = x[..., original_dim:]
            g = gate_module.gate(torch.cat([features, missing_mask], dim=-1))
            all_g.append(g.cpu().numpy())
            all_m.append(missing_mask.cpu().numpy())

    all_g = np.concatenate(all_g, axis=0)  # [N, seq_len, original_dim] 或 [N, original_dim]
    all_m = np.concatenate(all_m, axis=0)

    # 按是否缺失分别统计
    g_missing = all_g[all_m > 0.5]
    g_observed = all_g[all_m < 0.5]

    print(f'样本数: {all_g.shape[0]}')
    print(f'缺失位置 gate 均值: {g_missing.mean():.4f}, 标准差: {g_missing.std():.4f}')
    print(f'观测位置 gate 均值: {g_observed.mean():.4f}, 标准差: {g_observed.std():.4f}')

    # 每个特征的平均 gate（仅缺失位置）
    if all_g.ndim == 3:
        # 序列模型：对 seq_len 平均
        feat_g_missing = np.where(all_m > 0.5, all_g, np.nan)
        feat_mean = np.nanmean(feat_g_missing, axis=(0, 1))
        feat_std = np.nanstd(feat_g_missing, axis=(0, 1))
    else:
        feat_g_missing = np.where(all_m > 0.5, all_g, np.nan)
        feat_mean = np.nanmean(feat_g_missing, axis=0)
        feat_std = np.nanstd(feat_g_missing, axis=0)

    print('\n各特征在缺失位置的 gate 均值 ± 标准差:')
    for i, (m, s) in enumerate(zip(feat_mean, feat_std)):
        print(f'  feature {i:2d}: {m:.4f} ± {s:.4f}')


if __name__ == '__main__':
    main()
