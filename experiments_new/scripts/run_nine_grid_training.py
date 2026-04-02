#!/usr/bin/env python3
"""
九宫格完整训练脚本
配置：Level 2, 4插补方法 × 3缺失模式 × 3架构 × 2 configs × 3 seeds
108 个模型：3 架构 × 3 模式 × 4 插补 × 2 configs × 3 seeds
支持分批执行：--mode MCAR/MAR/MNAR
"""
import sys
import argparse
from pathlib import Path
import subprocess
import yaml
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 九宫格配置
BATCH = '2C'
SEEDS = [0, 1, 2]  # 3 seeds for statistical reliability
ARCHITECTURES = ['mlp', 'cnn', 'lstm']
LEVEL = 'level_2'  # 固定 Level 2（性能最佳）
IMPUTATION_METHODS = ['zero', 'mean', 'iterative', 'knn']  # 4种插补
MISSING_MODES = ['MCAR', 'MAR', 'MNAR']  # 3种缺失模式
CONFIGS = ['baseline', 'mim']

BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "nine_grid_complete"
DATA_DIR = PROJECT_DIR / "data" / "XJTU data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 参数字典
PARAM_COUNTS = {
    ('mlp', 'level_1'): 5953, ('mlp', 'level_2'): 11801, 
    ('mlp', 'level_3'): 20537, ('mlp', 'level_4'): 44529,
    ('cnn', 'level_1'): 5377, ('cnn', 'level_2'): 12249,
    ('cnn', 'level_3'): 20945, ('cnn', 'level_4'): 36185,
    ('lstm', 'level_1'): 5181, ('lstm', 'level_2'): 10957,
    ('lstm', 'level_3'): 26797, ('lstm', 'level_4'): 54337,
}


def create_model_config(arch, level):
    """创建模型配置文件"""
    config_dir = BASE_DIR / "configs" / "experiments" / "l1_architecture"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_type': arch,
        'level': level,
        'param_count': PARAM_COUNTS[(arch, level)]
    }
    
    # 架构特定配置
    if arch == 'mlp':
        if level == 'level_1':
            config['hidden_dims'] = [80, 56]
        elif level == 'level_2':
            config['hidden_dims'] = [120, 80]
        elif level == 'level_3':
            config['hidden_dims'] = [168, 104]
        elif level == 'level_4':
            config['hidden_dims'] = [256, 128, 56]
        config['dropout'] = 0.15
    elif arch == 'cnn':
        if level == 'level_1':
            config['channels'] = [40, 28]
        elif level == 'level_2':
            config['channels'] = [72, 40]
        elif level == 'level_3':
            config['channels'] = [96, 56]
        elif level == 'level_4':
            config['channels'] = [136, 72]
        config['kernel_size'] = 3
        config['dropout'] = 0.1
    elif arch == 'lstm':
        if level == 'level_1':
            config['hidden_size'] = 28
            config['num_layers'] = 1
        elif level == 'level_2':
            config['hidden_size'] = 44
            config['num_layers'] = 1
        elif level == 'level_3':
            config['hidden_size'] = 44
            config['num_layers'] = 2
        elif level == 'level_4':
            config['hidden_size'] = 64
            config['num_layers'] = 2
        config['dropout'] = 0.2
    
    config_file = config_dir / f"{arch}_{level}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file


def create_experiment_config(arch, config_type, seed, imputation, pattern):
    """创建实验配置"""
    use_mim = (config_type == 'mim')
    
    config = {
        'experiment_name': f"nine_{arch}_{LEVEL}_{pattern}_{imputation}_{config_type}_seed{seed}",
        'seed': seed,
        'data': {
            'batch': BATCH,
            'data_dir': str(DATA_DIR)
        },
        'model': {
            'model_type': arch,
            'level': LEVEL
        },
        'training': {
            'epochs': 200,
            'lr': 0.001,
            'batch_size': 32,
            'early_stopping_patience': 30,
            'weight_decay': 1.0e-5,
            'use_mim': use_mim,
            'imputation_method': imputation,
            'training_missing_rates': [i * 0.05 for i in range(20)],
            'missing_pattern': pattern,
        },
        'testing': {
            'modes': [pattern],
            'test_missing_rates': [i * 0.05 for i in range(20)],
            'imputations': [imputation]
        },
        'paths': {
            'model_dir': str(RESULTS_DIR / "models"),
            'results_dir': str(RESULTS_DIR),
            'logs_dir': str(RESULTS_DIR / "logs"),
            'data_dir': str(DATA_DIR)
        }
    }
    
    config_dir = BASE_DIR / "configs" / "experiments" / "nine_grid"
    config_file = config_dir / f"exp_{arch}_{pattern}_{imputation}_{config_type}_seed{seed}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file


def train_model(arch, config_type, seed, imputation, pattern):
    """训练单个模型"""
    model_config = create_model_config(arch, LEVEL)
    exp_config = create_experiment_config(arch, config_type, seed, imputation, pattern)
    
    print(f"🚀 训练: {arch.upper()} × {pattern} × {imputation} × {config_type} seed={seed}")
    start = time.time()
    
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "train_l1_models_fixed.py"),
        "--config", str(exp_config),
        "--model-config", str(model_config),
        "--seed", str(seed),
        "--output-dir", str(RESULTS_DIR)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start
        print(f"✅ 完成: {arch} {config_type} seed={seed} ({elapsed:.1f}s)")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"❌ 失败: {arch} {config_type} seed={seed} ({elapsed:.1f}s)")
        print(f"   错误: {e.stderr[:200]}")
        return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只打印计划，不执行训练')
    parser.add_argument('--arch', type=str, choices=ARCHITECTURES, help='指定单个架构')
    parser.add_argument('--mode', type=str, choices=MISSING_MODES, help='指定缺失模式(MCAR/MAR/MNAR)，默认全部')
    parser.add_argument('--imp', type=str, choices=IMPUTATION_METHODS, help='指定插补方法，默认全部')
    args = parser.parse_args()
    
    archs = [args.arch] if args.arch else ARCHITECTURES
    modes = [args.mode] if args.mode else MISSING_MODES
    imps = [args.imp] if args.imp else IMPUTATION_METHODS
    
    total_models = len(archs) * len(modes) * len(imps) * len(CONFIGS) * len(SEEDS)
    
    print("=" * 70)
    print("九宫格完整训练")
    print("=" * 70)
    print(f"架构: {archs}")
    print(f"缺失模式: {modes}")
    print(f"插补方法: {imps}")
    print(f"级别: {LEVEL}")
    print(f"Seeds: {SEEDS}")
    print(f"总模型数: {total_models}")
    print("=" * 70)
    
    if args.dry_run:
        count = 0
        for arch in archs:
            for mode in modes:
                for imp in imps:
                    for config in CONFIGS:
                        for seed in SEEDS:
                            count += 1
                            print(f"[{count:3d}/{total_models}] {arch} × {mode} × {imp} × {config} seed={seed}")
        return
    
    results = {}
    total_start = time.time()
    
    for arch in archs:
        results[arch] = {}
        for mode in modes:
            results[arch][mode] = {}
            for imp in imps:
                results[arch][mode][imp] = {}
                for config in CONFIGS:
                    results[arch][mode][imp][config] = {}
                    for seed in SEEDS:
                        success, elapsed = train_model(arch, config, seed, imp, mode)
                        results[arch][mode][imp][config][seed] = {
                            'success': success,
                            'time': elapsed
                        }
    
    total_time = time.time() - total_start
    
    # 统计成功/失败
    success_count = sum(1 for a in results.values() for m in a.values() 
                       for i in m.values() for c in i.values() 
                       for s in c.values() if s['success'])
    
    # 保存结果摘要
    summary = {
        'total_time': total_time,
        'total_models': total_models,
        'success_count': success_count,
        'failed_count': total_models - success_count,
        'results': results,
        'config': {
            'architectures': archs,
            'missing_modes': modes,
            'imputation_methods': imps,
            'level': LEVEL,
            'seeds': SEEDS
        }
    }
    
    summary_file = RESULTS_DIR / "nine_grid_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("九宫格训练完成")
    print("=" * 70)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"成功: {success_count}/{total_models}")
    print(f"失败: {total_models - success_count}/{total_models}")
    print(f"结果保存: {summary_file}")


if __name__ == '__main__':
    main()
