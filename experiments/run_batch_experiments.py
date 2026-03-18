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
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

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


def run_subprocess_with_progress(
    cmd: List[str],
    config_str: str,
    pbar: tqdm,
    timeout: int = 600
) -> tuple[bool, str, float]:
    """
    运行子进程并更新进度条
    
    Returns:
        (success, error_msg, elapsed_time)
    """
    start = time.time()
    pbar.set_postfix_str(f"Running: {config_str}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            pbar.set_postfix_str(f"✓ {config_str} ({elapsed:.1f}s)")
            return True, "", elapsed
        else:
            error = result.stderr[:50] if result.stderr else "Unknown error"
            pbar.set_postfix_str(f"✗ {config_str}: {error}")
            return False, result.stderr[:500], elapsed
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        pbar.set_postfix_str(f"✗ {config_str}: Timeout")
        return False, "Timeout", elapsed
        
    except Exception as e:
        elapsed = time.time() - start
        pbar.set_postfix_str(f"✗ {config_str}: {str(e)[:50]}")
        return False, str(e), elapsed


def run_training_phase(args) -> List[Dict]:
    """分界线以上：训练阶段（L1-L6）"""
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    total = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    completed = failed = 0
    results = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Training Phase (L1-L6): {total} models")
    print(f"{'='*70}\n")
    
    with tqdm(total=total, desc="Training", ncols=100,
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for seed in seeds:
            for batch in batches:
                for model in models:
                    for use_mim in USE_MIMS:
                        config = f"{batch}-{model}-{'MIM' if use_mim=='true' else 'Std'}"
                        
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
                            pbar.set_postfix_str(f"Dry-run: {config}")
                            completed += 1
                        else:
                            success, error, elapsed = run_subprocess_with_progress(cmd, config, pbar, timeout=600)
                            completed += 1
                            
                            results.append({
                                'seed': seed, 'batch': batch, 'model': model,
                                'use_mim': use_mim, 'status': 'success' if success else 'failed',
                                'error': error, 'elapsed': elapsed
                            })
                            if not success:
                                failed += 1
                        
                        pbar.update(1)
    
    print(f"\n{'='*70}")
    print(f"Training Complete: {completed - failed}/{completed} successful")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"{'='*70}\n")
    
    return results


def run_testing_phase(args) -> List[Dict]:
    """分界线以下：测试阶段（L7-L9）"""
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    n_models = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    n_tests_per_model = len(MODES) * len(TEST_MRS) * len(IMPUTATIONS)
    total = n_models * n_tests_per_model
    completed = failed = 0
    all_results = []
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Testing Phase (L7-L9): {total} tests")
    print(f"{'='*70}\n")
    
    with tqdm(total=total, desc="Testing", ncols=100,
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for seed in seeds:
            for batch in batches:
                for model in models:
                    for use_mim in USE_MIMS:
                        model_str = f"{batch}-{model}-{'MIM' if use_mim=='true' else 'Std'}"
                        
                        for mode in MODES:
                            for test_mr in TEST_MRS:
                                for imp in IMPUTATIONS:
                                    config = f"{model_str}-{mode}-{test_mr}-{imp[:4]}"
                                    
                                    result_file = os.path.join(
                                        args.results_dir,
                                        f"seed{seed}_batch{batch}_model{model}_"
                                        f"mim{use_mim}_mode{mode}_mr{test_mr}_imp{imp}.json"
                                    )
                                    
                                    # 断点续传
                                    if os.path.exists(result_file) and not args.dry_run:
                                        with open(result_file) as f:
                                            result = json.load(f)
                                        all_results.append(result)
                                        pbar.update(1)
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
                                        '--imputation', imp,
                                        '--model-dir', args.model_dir,
                                        '--output', result_file
                                    ]
                                    
                                    if args.dry_run:
                                        pbar.set_postfix_str(f"Dry-run: {config}")
                                        completed += 1
                                    else:
                                        success, error, elapsed = run_subprocess_with_progress(cmd, config, pbar, timeout=60)
                                        completed += 1
                                        
                                        if success and os.path.exists(result_file):
                                            with open(result_file) as f:
                                                result_data = json.load(f)
                                            result_data['elapsed'] = elapsed
                                            all_results.append(result_data)
                                            mae = result_data.get('test_mae')
                                            if mae:
                                                pbar.set_postfix_str(f"✓ {config} MAE={mae:.4f}")
                                        else:
                                            failed += 1
                                            all_results.append({
                                                'seed': seed, 'batch': batch, 'model': model,
                                                'use_mim': use_mim, 'mode': mode,
                                                'test_mr': test_mr, 'imputation': imp,
                                                'status': 'failed', 'error': error, 'elapsed': elapsed
                                            })
                                    
                                    pbar.update(1)
    
    print(f"\n{'='*70}")
    print(f"Testing Complete: {completed - failed}/{completed} successful")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
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
    output_file = os.path.join(results_dir, 'aggregated_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
    print(f"Total records: {len(df)}")
    
    if 'test_mae' in df.columns:
        print("\nMAE Statistics:")
        print(df.groupby(['use_mim', 'imputation'])['test_mae'].mean().unstack())
    
    return df


def main():
    args = parse_args()
    
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
    
    if args.phase in ['full', 'train']:
        train_results = run_training_phase(args)
        with open(os.path.join(args.results_dir, 'train_summary.json'), 'w') as f:
            json.dump(train_results, f, indent=2)
    
    if args.phase in ['full', 'test']:
        test_results = run_testing_phase(args)
        if not args.dry_run:
            aggregate_results(args.results_dir)
    
    print(f"\n{'#'*70}")
    print(f"# Experiment Run Complete")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
