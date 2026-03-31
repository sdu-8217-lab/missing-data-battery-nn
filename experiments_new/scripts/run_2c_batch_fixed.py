#!/usr/bin/env python3
"""
2C Batch 完整实验 - Phase 1: seed 0 (修复版)
24 个模型: 3 架构 × 4 Level × 2 config (baseline/mim)
"""
import sys
import argparse
from pathlib import Path
import subprocess
import yaml
import time

# 实验配置
BATCH = '2C'
SEEDS = [0]
ARCHITECTURES = ['mlp', 'cnn', 'lstm']
LEVELS = ['level_1', 'level_2', 'level_3', 'level_4']
CONFIGS = ['baseline', 'mim']

BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = BASE_DIR.parent  # 项目根目录
MODEL_CONFIG_DIR = BASE_DIR / "configs" / "models"
CONFIG_DIR = BASE_DIR / "configs" / "experiments" / "batch_2c_complete"
RESULTS_DIR = BASE_DIR / "results" / "2c_complete"
DATA_DIR = PROJECT_DIR / "data" / "XJTU data"  # 修正的数据路径

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_experiment_config(arch, level, config_type, seed):
    """创建实验配置文件"""
    use_mim = (config_type == 'mim')
    
    config = {
        'experiment_name': f"2c_{arch}_{level}_{config_type}_seed{seed}",
        'description': f"2C batch {arch.upper()} {level} {config_type} seed={seed}",
        'seed': seed,
        'data': {
            'batch': BATCH,
            'data_dir': str(DATA_DIR)  # 使用修正的数据路径
        },
        'model': {
            'model_type': arch,
            'level': level
        },
        'training': {
            'epochs': 200,
            'lr': 0.001,
            'batch_size': 32,
            'early_stopping_patience': 30,
            'weight_decay': 1.0e-5,
            'use_mim': use_mim,
            'training_missing_rates': [i * 0.05 for i in range(20)],
            'imputation_method': 'zero'
        },
        'testing': {
            'modes': ['MCAR', 'MAR', 'MNAR'],
            'test_missing_rates': [i * 0.05 for i in range(20)],
            'imputations': ['zero', 'mean', 'iterative']
        },
        'paths': {
            'model_dir': str(RESULTS_DIR / "models"),
            'results_dir': str(RESULTS_DIR),
            'logs_dir': str(RESULTS_DIR / "logs"),
            'data_dir': str(DATA_DIR)
        }
    }
    
    config_file = CONFIG_DIR / f"2c_{arch}_{level}_{config_type}_seed{seed}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file


def check_model_exists(arch, level, config_type, seed):
    """检查模型是否已存在"""
    model_filename = f"model_2C_{arch}_{level}_{config_type}_seed{seed}.pt"
    # 搜索任何匹配的模型文件
    for model_dir in (RESULTS_DIR / "models").rglob(f"*{model_filename}*"):
        if model_dir.exists():
            return True
    return False


def train_model(arch, level, config_type, seed):
    """训练单个模型"""
    config_file = create_experiment_config(arch, level, config_type, seed)
    model_config = MODEL_CONFIG_DIR / f"{arch}_{level}.yaml"
    
    if not model_config.exists():
        print(f"❌ 模型配置不存在: {model_config}")
        return False
    
    if check_model_exists(arch, level, config_type, seed):
        print(f"⏭️  模型已存在，跳过: {arch} {level} {config_type} seed={seed}")
        return True
    
    print(f"🚀 开始训练: {arch.upper()} {level} {config_type} seed={seed}")
    start_time = time.time()
    
    cmd = [
        sys.executable,
        "scripts/train_models.py",
        "--config", str(config_file),
        "--model-config", str(model_config),
        "--seeds", str(seed),
        "--output-dir", str(RESULTS_DIR)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ 训练完成: {arch} {level} {config_type} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ 训练失败: {arch} {level} {config_type} ({elapsed:.1f}s)")
        if "未找到" in e.stderr:
            print(f"   错误: 数据路径问题 - 请检查 DATA_DIR 配置")
        else:
            print(f"   错误: {e.stderr[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--arch', type=str, default=None)
    args = parser.parse_args()
    
    archs = [args.arch] if args.arch else ARCHITECTURES
    
    print("=" * 60)
    print("2C Batch 完整实验 - Phase 1 (seed 0)")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"数据存在: {DATA_DIR.exists()}")
    print(f"架构: {archs}")
    print(f"总计: {len(archs) * len(LEVELS) * len(CONFIGS)} 个模型")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print("❌ 错误: 数据目录不存在！")
        return
    
    total = len(archs) * len(LEVELS) * len(CONFIGS)
    completed = skipped = failed = 0
    start_time = time.time()
    
    for seed in SEEDS:
        for arch in archs:
            for level in LEVELS:
                for config in CONFIGS:
                    if check_model_exists(arch, level, config, seed):
                        skipped += 1
                        continue
                    
                    success = train_model(arch, level, config, seed)
                    if success:
                        completed += 1
                    else:
                        failed += 1
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成统计")
    print("=" * 60)
    print(f"总计: {total}, 成功: {completed}, 跳过: {skipped}, 失败: {failed}")
    print(f"总耗时: {total_time/60:.1f} 分钟")


if __name__ == '__main__':
    main()
