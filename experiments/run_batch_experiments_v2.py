#!/usr/bin/env python3
"""
批量实验运行器 V2 - 多进程并行优化版

优化点：
1. 使用 multiprocessing 并行测试多个模型（CPU核数-1）
2. 移除 subprocess 开销，直接函数调用
3. 进程安全的 CSV 写入

Usage:
    python run_batch_experiments_v2.py --phase test --seeds 0 1 2 ...
"""

import os
import sys
import argparse
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# 实验配置
SEEDS = list(range(100))
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
MODELS = ["mlp", "lstm", "cnn"]
USE_MIMS = ["false", "true"]
MODES = ["MCAR", "MAR", "MNAR"]
TEST_MRS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
IMPUTATIONS = ["mean", "knn", "iterative", "zero"]


def parse_args():
    parser = argparse.ArgumentParser(description='Run batch experiments with parallel testing')
    parser.add_argument('--phase', type=str, default='test',
                       choices=['test'],
                       help='Run phase (only test supported in v2)')
    parser.add_argument('--model-dir', type=str, default='models/100seeds',
                       help='Directory to load models')
    parser.add_argument('--results-dir', type=str, default='results/100seeds',
                       help='Directory to save results')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Random seeds')
    parser.add_argument('--batches', type=str, nargs='+', default=None,
                       help='Batches to run')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    return parser.parse_args()


def get_model_path(model_dir: str, seed: int, batch: str, model: str, use_mim: str) -> str:
    """生成模型文件路径"""
    mim_str = "mim" if use_mim == 'true' else "no_mim"
    return os.path.join(model_dir, f"seed{seed}_batch{batch}_model{model}_{mim_str}.pt")


def batch_test_single_model(args_tuple) -> Tuple[List[Dict], float]:
    """
    测试单个模型的所有配置（120个测试）
    
    这个函数在单独的进程中运行，避免GIL限制
    """
    seed, batch, model_type, use_mim_str, model_dir = args_tuple
    
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.xjtu_loader import XJTUDataLoader
    from src.models.mlp import MLP
    from src.models.lstm import LSTM
    from src.models.cnn1d import CNN1D
    from src.utils.seed_manager import set_seed
    
    start_time = time.time()
    use_mim = use_mim_str == 'true'
    input_dim = 32 if use_mim else 16
    
    try:
        # 加载模型
        model_path = get_model_path(model_dir, seed, batch, model_type, use_mim_str)
        if not os.path.exists(model_path):
            return [], 0.0
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 从 state_dict 推断模型结构
        state_dict = checkpoint['model_state_dict']
        
        # 创建模型（根据 state_dict 的形状）
        if model_type == 'mlp':
            # 推断隐藏层维度：network.0.weight 是 [hidden0, input]
            # network.3.weight 是 [hidden1, hidden0], 等等
            hidden_dims = []
            layer_idx = 0
            while f'network.{layer_idx}.weight' in state_dict:
                out_dim = state_dict[f'network.{layer_idx}.weight'].shape[0]
                # 最后一层是输出层（大小为1），不加入 hidden_dims
                if out_dim > 1:
                    hidden_dims.append(out_dim)
                layer_idx += 3  # 每个线性层间隔3（weight, bias, activation/dropout）
            
            model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
        
        elif model_type == 'lstm':
            # 从 weight_hh_l0 推断 hidden_size: shape = [4*hidden_size, hidden_size]
            if 'lstm.weight_hh_l0' in state_dict:
                hidden_size = state_dict['lstm.weight_hh_l0'].shape[1]
                # 计算层数：统计 weight_ih_l{layer} 的数量
                num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
                model = LSTM(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers)
            else:
                model = LSTM(input_dim=input_dim)
        
        else:  # CNN1D
            # 推断 channels 和 kernel_size
            # conv_layers.0.weight shape: [out_channels, in_channels, kernel_size]
            if 'conv_layers.0.weight' in state_dict:
                conv0_shape = state_dict['conv_layers.0.weight'].shape
                kernel_size = conv0_shape[2]
                
                # 收集所有卷积层的 channels
                channels = []
                layer_idx = 0
                while f'conv_layers.{layer_idx}.weight' in state_dict:
                    out_ch = state_dict[f'conv_layers.{layer_idx}.weight'].shape[0]
                    channels.append(out_ch)
                    layer_idx += 3  # weight, bias, activation
                
                model = CNN1D(input_dim=input_dim, channels=channels, kernel_size=kernel_size)
            else:
                model = CNN1D(input_dim=input_dim)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        train_mean = checkpoint.get('train_mean')
        train_std = checkpoint.get('train_std')
        
        # 加载测试数据
        loader = XJTUDataLoader(batch_id=batch, data_dir="data/XJTU data")
        df = loader.load_data()
        
        # 设置随机种子
        np.random.seed(seed)
        batteries = df['battery_id'].unique().tolist()  # 转换为 list 避免 StringArray 警告
        np.random.shuffle(batteries)
        
        n_train = max(4, len(batteries) // 2)
        n_val = max(2, len(batteries) // 4)
        test_batteries = batteries[n_train + n_val:]
        
        test_df = df[df['battery_id'].isin(test_batteries)]
        feature_cols = [c for c in df.columns 
                       if c not in ['battery_id', 'cycle', 'capacity', 'soh']]
        
        X_test_base = test_df[feature_cols].values.astype(np.float32)
        label_col = 'capacity' if 'capacity' in test_df.columns else 'soh'
        y_test = test_df[label_col].values.astype(np.float32)
        
        # 清理数据
        X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        if train_mean is not None and train_std is not None:
            X_test_base = (X_test_base - train_mean) / train_std
        
        # 运行所有测试配置
        results = []
        
        for mode in MODES:
            for test_mr in TEST_MRS:
                # 生成缺失数据
                np.random.seed(seed + 1000 + int(test_mr * 100))
                X_test = X_test_base.copy()
                
                if test_mr > 0:
                    if mode == 'MCAR':
                        mask = np.random.rand(*X_test.shape) < test_mr
                    elif mode == 'MAR':
                        voltage_col = None
                        for i, col in enumerate(feature_cols):
                            if 'voltage' in col.lower():
                                voltage_col = i
                                break
                        if voltage_col is not None:
                            voltage = X_test[:, voltage_col]
                            voltage_norm = (voltage - voltage.min()) / (voltage.max() - voltage.min() + 1e-8)
                            missing_prob = test_mr * (0.5 + 0.5 * voltage_norm)
                            mask = np.random.rand(*X_test.shape) < missing_prob[:, np.newaxis]
                        else:
                            mask = np.random.rand(*X_test.shape) < test_mr
                    else:  # MNAR
                        soh = y_test
                        soh_norm = (soh - soh.min()) / (soh.max() - soh.min() + 1e-8)
                        missing_prob = test_mr * (0.5 + 0.5 * soh_norm)
                        mask = np.random.rand(*X_test.shape) < missing_prob[:, np.newaxis]
                    
                    X_test_missing = X_test.copy()
                    X_test_missing[mask] = np.nan
                else:
                    X_test_missing = X_test.copy()
                    mask = np.zeros_like(X_test, dtype=bool)
                
                # 4种插补方法
                for imp in IMPUTATIONS:
                    try:
                        if imp == 'mean':
                            imputer = SimpleImputer(strategy='mean')
                            X_test_imputed = imputer.fit_transform(X_test_missing)
                        elif imp == 'knn':
                            imputer = KNNImputer(n_neighbors=5)
                            X_test_imputed = imputer.fit_transform(X_test_missing)
                        elif imp == 'iterative':
                            # 平衡速度和精度：max_iter=10
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                imputer = IterativeImputer(max_iter=10, random_state=seed, tol=1e-2)
                                X_test_imputed = imputer.fit_transform(X_test_missing)
                        else:  # zero
                            X_test_imputed = np.nan_to_num(X_test_missing, nan=0.0)
                        
                        # 准备输入
                        X_test_tensor = torch.FloatTensor(X_test_imputed).to(device)
                        
                        if use_mim:
                            mask_tensor = torch.FloatTensor(mask.astype(float)).to(device)
                            X_test_tensor = torch.cat([X_test_tensor, mask_tensor], dim=1)
                        
                        if model_type in ['cnn', 'lstm']:
                            X_test_tensor = X_test_tensor.unsqueeze(1)
                        
                        # 预测
                        with torch.no_grad():
                            predictions = model(X_test_tensor).squeeze().cpu().numpy()
                        
                        # 计算指标
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        
                        results.append({
                            'seed': seed,
                            'batch': batch,
                            'model': model_type,
                            'use_mim': use_mim_str,
                            'mode': mode,
                            'test_mr': test_mr,
                            'imputation': imp,
                            'test_mae': mae,
                            'test_r2': r2,
                            'n_samples': len(y_test),
                            'status': 'success'
                        })
                    except Exception as e:
                        results.append({
                            'seed': seed,
                            'batch': batch,
                            'model': model_type,
                            'use_mim': use_mim_str,
                            'mode': mode,
                            'test_mr': test_mr,
                            'imputation': imp,
                            'status': 'failed',
                            'error': str(e)[:200]
                        })
        
        elapsed = time.time() - start_time
        return results, elapsed
        
    except Exception as e:
        import traceback
        print(f"Error in batch_test_single_model: {e}")
        traceback.print_exc()
        return [], 0.0


def run_testing_phase_parallel(args) -> None:
    """并行测试阶段"""
    seeds = args.seeds if args.seeds else SEEDS
    batches = args.batches if args.batches else BATCHES
    models = args.models if args.models else MODELS
    
    n_models = len(seeds) * len(batches) * len(models) * len(USE_MIMS)
    n_tests_per_model = len(MODES) * len(TEST_MRS) * len(IMPUTATIONS)
    
    csv_file = os.path.join(args.results_dir, 'test_results.csv')
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 加载已完成的记录
    existing_keys = set()
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                key = f"{row['seed']}_{row['batch']}_{row['model']}_{str(row['use_mim']).lower()}"
                existing_keys.add(key)
            print(f"Found {len(df)} existing test results")
        except:
            pass
    
    # 构建待测试列表
    tasks = []
    for seed in seeds:
        for batch in batches:
            for model in models:
                for use_mim in USE_MIMS:
                    key = f"{seed}_{batch}_{model}_{str(use_mim).lower()}"
                    if key not in existing_keys:
                        tasks.append((seed, batch, model, use_mim, args.model_dir))
    
    print(f"\n{'='*70}")
    print(f"Parallel Testing Phase: {n_models} models × {n_tests_per_model} tests")
    print(f"Workers: {args.workers or (mp.cpu_count() - 1)}")
    print(f"Tasks: {len(tasks)} models to test")
    print(f"Results: {csv_file}")
    print(f"{'='*70}\n")
    
    # 初始化 CSV
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=[
            'seed', 'batch', 'model', 'use_mim', 'mode', 'test_mr', 'imputation',
            'test_mae', 'test_r2', 'n_samples', 'status'
        ]).to_csv(csv_file, index=False)
    
    # 使用进程池并行执行
    workers = args.workers or max(1, mp.cpu_count() - 1)
    
    completed = 0
    failed = 0
    start_time = time.time()
    
    with mp.Pool(processes=workers) as pool:
        with tqdm(total=len(tasks), desc="Testing", ncols=100) as pbar:
            # imap_unordered 返回结果时不保证顺序
            for results, elapsed in pool.imap_unordered(batch_test_single_model, tasks):
                if results:
                    # 写入 CSV
                    df_new = pd.DataFrame(results)
                    df_new.to_csv(csv_file, mode='a', header=False, index=False)
                    
                    n_success = sum(1 for r in results if r.get('status') == 'success')
                    n_failed = len(results) - n_success
                    completed += n_success
                    failed += n_failed
                    
                    # 计算平均 MAE
                    maes = [r.get('test_mae') for r in results if r.get('test_mae') is not None]
                    avg_mae = sum(maes) / len(maes) if maes else 0
                    
                    pbar.set_postfix_str(f"MAE={avg_mae:.4f}, {elapsed:.1f}s")
                else:
                    failed += 120
                
                pbar.update(1)
    
    print(f"\n{'='*70}")
    print(f"Complete: {completed} tests successful, {failed} failed")
    print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"{'='*70}\n")


def main():
    args = parse_args()
    
    if args.phase == 'test':
        run_testing_phase_parallel(args)
    
    print("Experiment Complete!")


if __name__ == '__main__':
    # 多进程必须在 if __name__ == '__main__' 中运行
    mp.set_start_method('spawn', force=True)
    main()
