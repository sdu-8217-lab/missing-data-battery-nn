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
) -> tuple[bool, str, float, str]:
    """
    运行子进程并更新进度条
    
    Returns:
        (success, error_msg, elapsed_time, stdout)
    """
    start = time.time()
    pbar.set_postfix_str(f"Running: {config_str}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            pbar.set_postfix_str(f"✓ {config_str} ({elapsed:.1f}s)")
            return True, "", elapsed, result.stdout
        else:
            error = result.stderr[:50] if result.stderr else "Unknown error"
            pbar.set_postfix_str(f"✗ {config_str}: {error}")
            return False, result.stderr[:500], elapsed, result.stdout
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        pbar.set_postfix_str(f"✗ {config_str}: Timeout")
        return False, "Timeout", elapsed, ""
        
    except Exception as e:
        elapsed = time.time() - start
        pbar.set_postfix_str(f"✗ {config_str}: {str(e)[:50]}")
        return False, str(e), elapsed, ""


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


def load_existing_results(csv_file: str) -> set:
    """加载已完成的实验记录，返回标识符集合用于断点续传"""
    if not os.path.exists(csv_file):
        return set()
    
    try:
        df = pd.read_csv(csv_file)
        # 创建唯一标识符: seed_batch_model_use_mim_mode_test_mr_imputation
        identifiers = set()
        for _, row in df.iterrows():
            key = f"{row['seed']}_{row['batch']}_{row['model']}_{str(row['use_mim']).lower()}_{row['mode']}_{row['test_mr']}_{row['imputation']}"
            identifiers.add(key)
        return identifiers
    except Exception as e:
        print(f"Warning: Failed to load existing results: {e}")
        return set()


def run_testing_phase(args) -> List[Dict]:
    """分界线以下：测试阶段（L7-L9）
    
    优化版本：使用 batch-test 模式，每个模型只启动一次进程，测试120个配置
    结果直接写入 CSV 文件，无需中间 JSON 文件
    3 modes × 10 MRs × 4 imputations = 120 tests per model
    """
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    n_models = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    n_tests_per_model = len(MODES) * len(TEST_MRS) * len(IMPUTATIONS)
    total_tests = n_models * n_tests_per_model
    total_models = n_models
    
    completed_models = 0
    completed_tests = 0
    failed_tests = 0
    all_results = []
    start_time = time.time()
    
    # CSV 文件路径
    csv_file = os.path.join(args.results_dir, 'test_results.csv')
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 加载已完成的记录（断点续传）
    existing_keys = load_existing_results(csv_file) if not args.dry_run else set()
    if existing_keys:
        print(f"Found {len(existing_keys)} existing test results in {csv_file}")
    
    # 计算实际需要测试的模型数
    models_to_test = []
    for seed in seeds:
        for batch in batches:
            for model in models:
                for use_mim in USE_MIMS:
                    # 检查该模型的120个测试是否都已完成
                    missing_tests = 0
                    for mode in MODES:
                        for test_mr in TEST_MRS:
                            for imp in IMPUTATIONS:
                                key = f"{seed}_{batch}_{model}_{str(use_mim).lower()}_{mode}_{test_mr}_{imp}"
                                if key not in existing_keys:
                                    missing_tests += 1
                    
                    # 只有当有测试缺失时才添加到待测试列表
                    if missing_tests > 0:
                        models_to_test.append((seed, batch, model, use_mim))
    
    skipped_models = total_models - len(models_to_test)
    
    print(f"\n{'='*70}")
    print(f"Testing Phase (L7-L9): {total_models} models × {n_tests_per_model} tests = {total_tests} total")
    print(f"Using batch-test mode: 1 process per model")
    if skipped_models > 0:
        print(f"Skipped {skipped_models} already completed models")
    print(f"Results will be saved to: {csv_file}")
    print(f"{'='*70}\n")
    
    # 初始化 CSV 文件（如果不存在）
    if not os.path.exists(csv_file) and not args.dry_run:
        pd.DataFrame(columns=[
            'seed', 'batch', 'model', 'use_mim', 'mode', 'test_mr', 'imputation',
            'test_mae', 'test_r2', 'n_samples', 'status'
        ]).to_csv(csv_file, index=False)
    
    # 外层循环：每个模型启动一次batch-test进程
    with tqdm(total=len(models_to_test), desc="Batch Testing", ncols=100,
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for seed, batch, model, use_mim in models_to_test:
            model_str = f"seed{seed}-{batch}-{model}-{'MIM' if use_mim=='true' else 'Std'}"
            
            # 构建batch-test命令（不再使用 --output）
            cmd = [
                'python', 'experiments/run_experiment.py',
                '--phase', 'batch-test',
                '--seed', str(seed),
                '--batch', batch,
                '--model', model,
                '--use-mim', use_mim,
                '--model-dir', args.model_dir
            ]
            
            if args.dry_run:
                pbar.set_postfix_str(f"Dry-run: {model_str}")
                completed_models += 1
                completed_tests += n_tests_per_model
            else:
                # 使用更长的超时时间，因为一次要测试120个配置
                success, error, elapsed, stdout = run_subprocess_with_progress(
                    cmd, model_str, pbar, timeout=600
                )
                completed_models += 1
                
                if success:
                    # 从 stdout 解析 JSON 结果
                    try:
                        # 查找 JSON_RESULTS_START 和 JSON_RESULTS_END 之间的内容
                        start_idx = stdout.find("JSON_RESULTS_START")
                        end_idx = stdout.find("JSON_RESULTS_END")
                        
                        if start_idx != -1 and end_idx != -1:
                            json_str = stdout[start_idx + len("JSON_RESULTS_START"):end_idx]
                            batch_results = json.loads(json_str)
                            
                            if isinstance(batch_results, list):
                                # 过滤掉已存在的记录（双重保险）
                                new_results = []
                                for r in batch_results:
                                    key = f"{r['seed']}_{r['batch']}_{r['model']}_{str(r['use_mim']).lower()}_{r['mode']}_{r['test_mr']}_{r['imputation']}"
                                    if key not in existing_keys:
                                        new_results.append(r)
                                        existing_keys.add(key)  # 添加到已存在集合
                                
                                if new_results:
                                    # 实时追加写入 CSV
                                    df_new = pd.DataFrame(new_results)
                                    df_new.to_csv(csv_file, mode='a', header=False, index=False)
                                
                                all_results.extend(batch_results)
                                n_success = sum(1 for r in batch_results if r.get('status') == 'success')
                                n_failed = len(batch_results) - n_success
                                completed_tests += n_success
                                failed_tests += n_failed
                                
                                # 计算平均MAE
                                maes = [r.get('test_mae') for r in batch_results if r.get('test_mae') is not None]
                                avg_mae = sum(maes) / len(maes) if maes else 0
                                pbar.set_postfix_str(f"✓ {model_str} ({n_success} tests, avg MAE={avg_mae:.4f}, {elapsed:.1f}s)")
                        else:
                            failed_tests += n_tests_per_model
                            pbar.set_postfix_str(f"✗ {model_str}: JSON parse error")
                    except Exception as e:
                        failed_tests += n_tests_per_model
                        pbar.set_postfix_str(f"✗ {model_str}: {str(e)[:30]}")
                else:
                    failed_tests += n_tests_per_model
                    pbar.set_postfix_str(f"✗ {model_str}: {error[:30]}")
            
            pbar.update(1)
    
    print(f"\n{'='*70}")
    print(f"Testing Complete: {completed_tests}/{total_tests} tests successful")
    print(f"Models: {completed_models}/{total_models}, Failed tests: {failed_tests}")
    print(f"Results saved to: {csv_file}")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"{'='*70}\n")
    
    return all_results


def aggregate_results(results_dir: str) -> pd.DataFrame:
    """读取测试结果CSV并显示统计信息"""
    csv_file = os.path.join(results_dir, 'test_results.csv')
    
    if not os.path.exists(csv_file):
        print(f"No results file found at {csv_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        
        # 过滤掉失败的记录
        df_success = df[df['status'] == 'success']
        print(f"Successful tests: {len(df_success)}/{len(df)}")
        
        if len(df_success) > 0 and 'test_mae' in df_success.columns:
            print("\nMAE Statistics by use_mim and imputation:")
            print(df_success.groupby(['use_mim', 'imputation'])['test_mae'].mean().unstack())
        
        return df_success
    except Exception as e:
        print(f"Error reading results: {e}")
        return pd.DataFrame()


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
