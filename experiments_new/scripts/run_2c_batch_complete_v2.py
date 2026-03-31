#!/usr/bin/env python3
"""
2C Batch 完整实验 - Phase 1: seed 0
24 个模型: 3 架构 × 4 Level × 2 config (baseline/mim)
"""
import sys
import argparse
from pathlib import Path
import subprocess
import yaml
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 实验配置
BATCH = '2C'
SEEDS = [0]  # Phase 1: 只跑 seed 0
ARCHITECTURES = ['mlp', 'cnn', 'lstm']
LEVELS = ['level_1', 'level_2', 'level_3', 'level_4']
CONFIGS = ['baseline', 'mim']  # baseline: use_mim=false, mim: use_mim=true

BASE_DIR = Path(__file__).parent.parent
MODEL_CONFIG_DIR = BASE_DIR / "configs" / "models"
CONFIG_DIR = BASE_DIR / "configs" / "experiments" / "batch_2c_complete"
RESULTS_DIR = BASE_DIR / "results" / "2c_complete"
LOGS_DIR = BASE_DIR / "logs"

# 确保目录存在
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def create_experiment_config(arch, level, config_type, seed):
    """创建实验配置文件"""
    use_mim = (config_type == 'mim')
    
    config = {
        'experiment_name': f"2c_{arch}_{level}_{config_type}_seed{seed}",
        'description': f"2C batch {arch.upper()} {level} {config_type} seed={seed}",
        'seed': seed,
        'data': {
            'batch': BATCH,
            'data_dir': 'data/XJTU data'
        },
        'model': {
            'model_type': arch,
            'level': level
        },
        'training': {
            'epochs': 200,
            'lr': 0.001,
            'batch_size': 32,
            'early_stopping_patience': 30,  # 正确的参数名
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
            'data_dir': 'data/XJTU data'
        }
    }
    
    # 保存配置文件
    config_file = CONFIG_DIR / f"2c_{arch}_{level}_{config_type}_seed{seed}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file


def check_model_exists(arch, level, config_type, seed):
    """检查模型是否已存在"""
    model_dir = RESULTS_DIR / "models" / f"2c_{arch}_{level}_{config_type}_seed{seed}"
    model_file = model_dir / "best_model.pt"
    return model_file.exists()


def train_model(arch, level, config_type, seed, dry_run=False):
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
    
    if dry_run:
        print(f"   [Dry Run] 配置: {config_file}")
        print(f"   [Dry Run] 模型配置: {model_config}")
        return True
    
    # 构建训练命令
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
        print(f"✅ 训练完成: {arch} {level} {config_type} seed={seed} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ 训练失败: {arch} {level} {config_type} seed={seed} ({elapsed:.1f}s)")
        print(f"   错误: {e.stderr[:300]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='2C Batch 完整实验')
    parser.add_argument('--dry-run', action='store_true', help='只生成配置，不实际训练')
    parser.add_argument('--arch', type=str, default=None, help='指定架构 (mlp/cnn/lstm)')
    parser.add_argument('--level', type=str, default=None, help='指定层级 (level_1-4)')
    args = parser.parse_args()
    
    # 确定训练范围
    archs = [args.arch] if args.arch else ARCHITECTURES
    levels = [args.level] if args.level else LEVELS
    
    print("=" * 60)
    print("2C Batch 完整实验 - Phase 1 (seed 0)")
    print("=" * 60)
    print(f"架构: {archs}")
    print(f"层级: {levels}")
    print(f"配置: {CONFIGS}")
    print(f"Seeds: {SEEDS}")
    print(f"总计: {len(archs) * len(levels) * len(CONFIGS) * len(SEEDS)} 个模型")
    print("=" * 60)
    
    # 统计
    total = len(archs) * len(levels) * len(CONFIGS) * len(SEEDS)
    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    for seed in SEEDS:
        for arch in archs:
            for level in levels:
                for config in CONFIGS:
                    if check_model_exists(arch, level, config, seed):
                        skipped += 1
                        print(f"⏭️  跳过: {arch} {level} {config} seed={seed}")
                        continue
                    
                    success = train_model(arch, level, config, seed, args.dry_run)
                    if success:
                        completed += 1
                    else:
                        failed += 1
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成统计")
    print("=" * 60)
    print(f"总计: {total}")
    print(f"成功: {completed}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print("=" * 60)


if __name__ == '__main__':
    main()
