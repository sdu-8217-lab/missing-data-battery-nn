#!/usr/bin/env python3
"""
批量实验运行器 - 基于meta.md的9层架构

完整实验矩阵：
- 分界线以上（L1-L6）：训练36个模型（1 seed × 6 batches × 3 models × 2 use_mim）
- 分界线以下（L7-L9）：每个模型测试120组合（3 modes × 10 test MRs × 4 imputations）
- 总结果：4,320行

Usage:
    # 运行完整实验（训练和测试）
    python run_batch_experiments.py --full-matrix --epochs 50
    
    # 仅训练阶段
    python run_batch_experiments.py --phase train --epochs 50
    
    # 仅测试阶段（需要已有训练好的模型）
    python run_batch_experiments.py --phase test
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# 实验配置（根据meta.md）
SEEDS = [42]
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
MODELS = ["mlp", "lstm", "cnn"]
USE_MIMS = ["false", "true"]  # L5: use_mim
MODES = ["MCAR", "MAR", "MNAR"]  # L7: Mode
TEST_MRS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # L8: 测试MR
IMPUTATIONS = ["mean", "knn", "iterative", "zero"]  # L9: Imputation


def parse_args():
    parser = argparse.ArgumentParser(description='Run full batch experiments')
    
    parser.add_argument('--phase', type=str, default='full',
                       choices=['full', 'train', 'test'],
                       help='Run phase: full (train+test), train only, or test only')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save/load models')
    
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Random seeds (default: [42])')
    
    parser.add_argument('--batches', type=str, nargs='+', default=None,
                       help='Batches to run (default: all 6 batches)')
    
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (default: mlp, lstm, cnn)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    
    return parser.parse_args()


def run_training_phase(args) -> List[Dict]:
    """
    分界线以上：训练阶段（L1-L6）
    训练36个模型（单种子）：1 seed × 6 batches × 3 models × 2 use_mim
    """
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    total = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    completed = 0
    failed = 0
    
    print(f"\n{'='*70}")
    print(f"Training Phase (L1-L6)")
    print(f"Total models to train: {total}")
    print(f"  Seeds: {seeds}")
    print(f"  Batches: {batches}")
    print(f"  Models: {models}")
    print(f"  Use MIM: {USE_MIMS}")
    print(f"{'='*70}\n")
    
    results = []
    
    for seed in seeds:
        for batch in batches:
            for model in models:
                for use_mim in USE_MIMS:
                    completed += 1
                    
                    print(f"\n[{completed}/{total}] Training: "
                          f"seed={seed}, batch={batch}, model={model}, use_mim={use_mim}")
                    
                    cmd = [
                        'python', 'experiments/run_experiment.py',
                        '--phase', 'train',
                        '--seed', str(seed),
                        '--batch', batch,
                        '--model', model,
                        '--use-mim', use_mim,
                        '--epochs', str(args.epochs),
                        '--save-model',
                        '--model-dir', args.model_dir
                    ]
                    
                    if args.dry_run:
                        print(f"Command: {' '.join(cmd)}")
                        continue
                    
                    try:
                        result = subprocess.run(
                            cmd, 
                            capture_output=True, 
                            text=True, 
                            timeout=600  # 10分钟超时
                        )
                        
                        if result.returncode == 0:
                            print(f"  ✓ Success")
                            results.append({
                                'seed': seed,
                                'batch': batch,
                                'model': model,
                                'use_mim': use_mim,
                                'status': 'success'
                            })
                        else:
                            print(f"  ✗ Failed: {result.stderr[:200]}")
                            failed += 1
                            results.append({
                                'seed': seed,
                                'batch': batch,
                                'model': model,
                                'use_mim': use_mim,
                                'status': 'failed',
                                'error': result.stderr[:500]
                            })
                    except subprocess.TimeoutExpired:
                        print(f"  ✗ Timeout")
                        failed += 1
                        results.append({
                            'seed': seed,
                            'batch': batch,
                            'model': model,
                            'use_mim': use_mim,
                            'status': 'timeout'
                        })
                    except Exception as e:
                        print(f"  ✗ Error: {str(e)}")
                        failed += 1
                        results.append({
                            'seed': seed,
                            'batch': batch,
                            'model': model,
                            'use_mim': use_mim,
                            'status': 'error',
                            'error': str(e)
                        })
    
    print(f"\n{'='*70}")
    print(f"Training Complete: {completed - failed}/{completed} successful")
    print(f"{'='*70}\n")
    
    return results


def run_testing_phase(args) -> List[Dict]:
    """
    分界线以下：测试阶段（L7-L9）
    每个模型测试120组合：3 modes × 10 test MRs × 4 imputations
    """
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    # 计算总数
    n_models = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    n_tests_per_model = len(MODES) * len(TEST_MRS) * len(IMPUTATIONS)
    total = n_models * n_tests_per_model
    
    completed = 0
    failed = 0
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"Testing Phase (L7-L9)")
    print(f"Models: {n_models}, Tests per model: {n_tests_per_model}")
    print(f"Total tests: {total}")
    print(f"{'='*70}\n")
    
    for seed in seeds:
        for batch in batches:
            for model in models:
                for use_mim in USE_MIMS:
                    print(f"\nTesting model: seed={seed}, batch={batch}, model={model}, use_mim={use_mim}")
                    
                    for mode in MODES:
                        for test_mr in TEST_MRS:
                            for imputation in IMPUTATIONS:
                                completed += 1
                                
                                # 生成结果文件名
                                result_file = os.path.join(
                                    args.results_dir,
                                    f"seed{seed}_batch{batch}_model{model}_"
                                    f"mim{use_mim}_mode{mode}_mr{test_mr}_imp{imputation}.json"
                                )
                                
                                # 检查是否已存在（支持断点续传）
                                if os.path.exists(result_file) and not args.dry_run:
                                    with open(result_file) as f:
                                        result = json.load(f)
                                    all_results.append(result)
                                    if completed % 100 == 0:
                                        print(f"  [{completed}/{total}] Already exists, skipped")
                                    continue
                                
                                cmd = [
                                    'python', 'experiments/run_experiment.py',
                                    '--phase', 'test',
                                    '--seed', str(seed),
                                    '--batch', batch,
                                    '--model', model,
                                    '--use-mim', use_mim,
                                    '--mode', mode,
                                    '--test-mr', str(test_mr),
                                    '--imputation', imputation,
                                    '--model-dir', args.model_dir,
                                    '--output', result_file
                                ]
                                
                                if args.dry_run:
                                    if completed <= 5 or completed == total:
                                        print(f"Command {completed}: {' '.join(cmd)}")
                                    elif completed == 6:
                                        print("  ... (more commands)")
                                    continue
                                
                                try:
                                    result = subprocess.run(
                                        cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=60  # 1分钟超时
                                    )
                                    
                                    if result.returncode == 0:
                                        # 读取结果
                                        if os.path.exists(result_file):
                                            with open(result_file) as f:
                                                result_data = json.load(f)
                                            all_results.append(result_data)
                                            if completed % 100 == 0:
                                                print(f"  [{completed}/{total}] MAE={result_data.get('test_mae', 'N/A'):.4f}")
                                    else:
                                        if completed % 100 == 0 or completed == total:
                                            print(f"  [{completed}/{total}] Failed")
                                        failed += 1
                                        all_results.append({
                                            'seed': seed, 'batch': batch, 'model': model,
                                            'use_mim': use_mim, 'mode': mode,
                                            'test_mr': test_mr, 'imputation': imputation,
                                            'status': 'failed'
                                        })
                                except Exception as e:
                                    if completed % 100 == 0 or completed == total:
                                        print(f"  [{completed}/{total}] Error: {str(e)[:50]}")
                                    failed += 1
                                    all_results.append({
                                        'seed': seed, 'batch': batch, 'model': model,
                                        'use_mim': use_mim, 'mode': mode,
                                        'test_mr': test_mr, 'imputation': imputation,
                                        'status': 'error', 'error': str(e)
                                    })
    
    print(f"\n{'='*70}")
    print(f"Testing Complete: {completed - failed}/{completed} successful")
    print(f"{'='*70}\n")
    
    return all_results


def aggregate_results(results_dir: str) -> pd.DataFrame:
    """聚合所有测试结果到CSV"""
    
    result_files = list(Path(results_dir).glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return pd.DataFrame()
    
    records = []
    for f in result_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                if data.get('status') != 'failed':
                    records.append(data)
        except:
            pass
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # 保存聚合结果
    output_file = os.path.join(results_dir, 'aggregated_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
    print(f"Total records: {len(df)}")
    
    # 打印汇总统计
    if 'test_mae' in df.columns:
        print("\nMAE Statistics:")
        print(df.groupby(['use_mim', 'imputation'])['test_mae'].mean().unstack())
    
    return df


def main():
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# Full Matrix Experiment Run")
    print(f"# Phase: {args.phase}")
    print(f"# Epochs: {args.epochs}")
    print(f"# Model Dir: {args.model_dir}")
    print(f"# Results Dir: {args.results_dir}")
    print(f"# {'Dry Run' if args.dry_run else 'Actual Run'}")
    print(f"{'#'*70}\n")
    
    # 训练阶段
    if args.phase in ['full', 'train']:
        train_results = run_training_phase(args)
        
        # 保存训练结果摘要
        train_summary_file = os.path.join(args.results_dir, 'train_summary.json')
        with open(train_summary_file, 'w') as f:
            json.dump(train_results, f, indent=2)
        print(f"Training summary saved to {train_summary_file}")
    
    # 测试阶段
    if args.phase in ['full', 'test']:
        test_results = run_testing_phase(args)
        
        # 聚合结果
        if not args.dry_run:
            df = aggregate_results(args.results_dir)
    
    print(f"\n{'#'*70}")
    print(f"# Experiment Run Complete")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
