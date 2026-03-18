#!/usr/bin/env python3
"""
重构后的实验运行器 - 基于meta.md的9层架构

分界线以上（L1-L6）：训练阶段
- L1 Seed, L3 Batch, L4 Model, L5 use_mim, L6 训练MR
- 每个组合独立训练模型

分界线以下（L7-L9）：测试阶段  
- L7 Mode, L8 测试MR, L9 Imputation
- 复用已训练模型测试所有组合

Usage:
    # 训练阶段（分界线以上）
    python run_experiment.py --mode train 
        --seed 42 --batch 2C --model mlp --use-mim false
        --epochs 50 --save-model
    
    # 测试阶段（分界线以下）
    python run_experiment.py --mode test
        --seed 42 --batch 2C --model mlp --use-mim false
        --mode MCAR --test-mr 0.3 --imputation mean
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xjtu_loader import XJTUDataLoader
from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.cnn1d import CNN1D
from src.utils.seed_manager import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run battery experiment with new architecture')
    
    # 运行模式
    parser.add_argument('--phase', type=str, required=True, 
                       choices=['train', 'test'],
                       help='Experiment phase: train (L1-L6) or test (L7-L9)')
    
    # 分界线以上参数（训练和测试都需要）
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (L1)')
    parser.add_argument('--batch', type=str, required=True,
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='Battery batch (L3)')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'lstm', 'cnn'],
                       help='Model architecture (L4)')
    parser.add_argument('--use-mim', type=str, default='false',
                       choices=['true', 'false'],
                       help='Use MIM indicators (L5)')
    
    # 训练阶段特定参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save/load models')
    
    # 测试阶段特定参数（分界线以下）
    parser.add_argument('--mode', type=str, default='MCAR',
                       choices=['MCAR', 'MAR', 'MNAR'],
                       help='Missingness mode (L7)')
    parser.add_argument('--test-mr', type=float, default=0.3,
                       help='Testing missing rate (L8), range: 0.0-0.9')
    parser.add_argument('--imputation', type=str, default='mean',
                       choices=['mean', 'knn', 'iterative', 'zero'],
                       help='Imputation method (L9)')
    
    # 输出
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    
    return parser.parse_args()


def get_model_path(model_dir: str, seed: int, batch: str, model_type: str, use_mim: bool) -> str:
    """生成模型保存/加载路径"""
    mim_str = "mim" if use_mim else "no_mim"
    return os.path.join(model_dir, f"seed{seed}_batch{batch}_model{model_type}_{mim_str}.pt")


def create_model(model_type: str, input_dim: int, use_mim: bool) -> nn.Module:
    """创建模型"""
    if model_type == 'mlp':
        # MLP: 使用默认配置，适配输入维度
        return MLP(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            dropout=0.1
        )
    elif model_type == 'lstm':
        # LSTM: 使用默认配置
        return LSTM(
            input_dim=input_dim,
            hidden_size=48,
            num_layers=2,
            dropout=0.2
        )
    elif model_type == 'cnn':
        # CNN1D: 需要适配输入维度
        return CNN1D(
            input_dim=input_dim,
            channels=[16, 32],
            kernel_size=3,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def prepare_training_data(batch_id: str, seed: int, use_mim: bool, model_type: str = 'mlp') -> Tuple:
    """
    准备训练数据
    
    Args:
        batch_id: 电池批次
        seed: 随机种子
        use_mim: 是否使用MIM
        model_type: 模型类型 ('mlp', 'lstm', 'cnn')
        
    Returns:
        如果use_mim=False: (X_train, y_train, X_val, y_val, train_mean, train_std)
        如果use_mim=True: (multi_mr_data, X_val, y_val, train_mean, train_std)
    """
    # 加载数据
    loader = XJTUDataLoader(batch_id=batch_id, data_dir="data/XJTU data")
    df = loader.load_data()
    
    # 特征工程：直接使用现有特征
    feature_df = df.copy()
    
    # 分割电池（使用seed控制）
    np.random.seed(seed)
    batteries = feature_df['battery_id'].unique()
    np.random.shuffle(batteries)
    
    n_train = max(4, len(batteries) // 2)
    n_val = max(2, len(batteries) // 4)
    
    train_batteries = batteries[:n_train]
    val_batteries = batteries[n_train:n_train + n_val]
    
    # 提取特征和标签
    feature_cols = [c for c in feature_df.columns 
                   if c not in ['battery_id', 'cycle', 'capacity', 'capacity']]
    
    train_df = feature_df[feature_df['battery_id'].isin(train_batteries)]
    val_df = feature_df[feature_df['battery_id'].isin(val_batteries)]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['capacity'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['capacity'].values.astype(np.float32)
    
    # 处理无效值（NaN和Inf）
    def clean_data(X, name):
        if np.isnan(X).any():
            print(f"Warning: {np.isnan(X).sum()} NaN values in {name}, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        if np.isinf(X).any():
            print(f"Warning: {np.isinf(X).sum()} Inf values in {name}, filling with 0")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        return X
    
    X_train = clean_data(X_train, "training data")
    X_val = clean_data(X_val, "validation data")
    
    # 标准化（使用训练集统计量）
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    
    # 处理标准差为0的情况（特征在所有样本中相同）
    train_std[train_std == 0] = 1.0
    train_std = train_std + 1e-8
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    
    if not use_mim:
        # use_mim=False: 使用完整数据训练
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        # 对于CNN/LSTM模型，需要添加序列维度 [batch, seq=1, features]
        if model_type in ['cnn', 'lstm']:
            X_train_tensor = X_train_tensor.unsqueeze(1)  # [batch, 1, features]
            X_val_tensor = X_val_tensor.unsqueeze(1)
        
        return (
            X_train_tensor, torch.FloatTensor(y_train),
            X_val_tensor, torch.FloatTensor(y_val),
            train_mean, train_std
        )
    else:
        # use_mim=True: 生成多MR训练数据（MCAR）
        multi_mr_data = []
        train_mr_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for mr in train_mr_list:
            if mr == 0.0:
                X_mr = X_train.copy()
                mask = np.zeros_like(X_train)  # MR=0时没有缺失
            else:
                # MCAR生成缺失
                np.random.seed(seed + int(mr * 100))
                mask = np.random.rand(*X_train.shape) < mr
                X_mr = X_train.copy()
                X_mr[mask] = 0  # 零值填充
            
            # 拼接掩码
            mask_tensor = torch.FloatTensor(mask.astype(float))
            X_mr_tensor = torch.FloatTensor(X_mr)
            X_combined = torch.cat([X_mr_tensor, mask_tensor], dim=1)
            
            # 对于CNN/LSTM模型，添加序列维度
            if model_type in ['cnn', 'lstm']:
                X_combined = X_combined.unsqueeze(1)  # [batch, 1, features*2]
            
            multi_mr_data.append((X_combined, torch.FloatTensor(y_train)))
        
        # 验证集也添加序列维度
        X_val_tensor = torch.FloatTensor(X_val)
        if model_type in ['cnn', 'lstm']:
            X_val_tensor = X_val_tensor.unsqueeze(1)
        
        return multi_mr_data, X_val_tensor, torch.FloatTensor(y_val), train_mean, train_std


def train_model(args) -> Dict:
    """
    训练模型（分界线以上：L1-L6）
    """
    use_mim = args.use_mim == 'true'
    input_dim = 32 if use_mim else 16
    
    print(f"\n{'='*60}")
    print(f"Training Phase (L1-L6)")
    print(f"Seed: {args.seed}, Batch: {args.batch}, Model: {args.model}")
    print(f"Use MIM: {args.use_mim} (Input dim: {input_dim})")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 准备数据
    if use_mim:
        multi_mr_data, X_val, y_val, train_mean, train_std = prepare_training_data(
            args.batch, args.seed, use_mim=True, model_type=args.model
        )
        # 验证集也生成多MR用于早停
        val_data_list = []
        for mr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if mr == 0.0:
                X_val_mr = X_val.clone()
                mask_val = torch.zeros_like(X_val)
            else:
                np.random.seed(args.seed + 999 + int(mr * 100))
                # 生成与X_val相同形状的mask
                if args.model in ['cnn', 'lstm']:
                    # X_val shape: [batch, 1, features], 去掉序列维度得到 [batch, features]
                    mask_shape = (X_val.shape[0], X_val.shape[-1])
                else:
                    mask_shape = X_val.shape
                mask = np.random.rand(*mask_shape) < mr
                X_val_mr = X_val.clone()
                # 将mask扩展到与X_val_mr相同的维度
                if args.model in ['cnn', 'lstm']:
                    mask_expanded = np.broadcast_to(mask[:, np.newaxis, :], X_val_mr.shape)
                else:
                    mask_expanded = mask
                X_val_mr[mask_expanded] = 0
                mask_val = torch.FloatTensor(mask.astype(float))
                if args.model in ['cnn', 'lstm']:
                    # mask shape: [batch, features] -> [batch, 1, features] 匹配 X_val_mr
                    mask_val = mask_val.unsqueeze(1)
            
            # 拼接掩码
            if args.model in ['cnn', 'lstm']:
                # 对于CNN/LSTM，在特征维度上拼接
                X_val_combined = torch.cat([X_val_mr, mask_val], dim=-1)
            else:
                X_val_combined = torch.cat([X_val_mr, mask_val], dim=1)
            val_data_list.append((X_val_combined, y_val))
    else:
        X_train, y_train, X_val, y_val, train_mean, train_std = prepare_training_data(
            args.batch, args.seed, use_mim=False, model_type=args.model
        )
    
    # 创建模型
    model = create_model(args.model, input_dim, use_mim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        if use_mim:
            # MIM: 多MR混合训练
            epoch_loss = 0
            for X_batch, y_batch in multi_mr_data:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(multi_mr_data)
            
            # 验证：多MR平均
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_data_list:
                    X_val_batch = X_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    outputs = model(X_val_batch).squeeze()
                    val_loss += criterion(outputs, y_val_batch).item()
            val_loss = val_loss / len(val_data_list)
        else:
            # 非MIM: 完整数据训练
            train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
            # 使用固定生成器确保可复现性
            generator = torch.Generator().manual_seed(args.seed + epoch)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
            
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            
            # 验证
            model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(device)
                y_val_device = y_val.to(device)
                outputs = model(X_val_device).squeeze()
                val_loss = criterion(outputs, y_val_device).item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            if args.save_model:
                os.makedirs(args.model_dir, exist_ok=True)
                model_path = get_model_path(
                    args.model_dir, args.seed, args.batch, args.model, use_mim
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'best_val_loss': best_val_loss
                }, model_path)
                print(f"  -> Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        'train_complete': True,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1
    }


def test_model(args) -> Dict:
    """
    测试模型（分界线以下：L7-L9）
    """
    use_mim = args.use_mim == 'true'
    input_dim = 32 if use_mim else 16
    
    print(f"\n{'='*60}")
    print(f"Testing Phase (L7-L9)")
    print(f"Mode: {args.mode}, Test MR: {args.test_mr}, Imputation: {args.imputation}")
    print(f"Use MIM: {args.use_mim} (Input dim: {input_dim})")
    print(f"{'='*60}\n")
    
    # 加载模型
    model_path = get_model_path(
        args.model_dir, args.seed, args.batch, args.model, use_mim
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train first.")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 创建模型并加载权重
    model = create_model(args.model, input_dim, use_mim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 获取标准化参数
    train_mean = checkpoint.get('train_mean')
    train_std = checkpoint.get('train_std')
    
    # 加载测试数据
    loader = XJTUDataLoader(batch_id=args.batch, data_dir="data/XJTU data")
    df = loader.load_data()
    
    # 直接使用加载的数据
    feature_df = df.copy()
    
    # 获取测试电池（使用相同的seed分割）
    np.random.seed(args.seed)
    batteries = feature_df['battery_id'].unique()
    np.random.shuffle(batteries)
    
    n_train = max(4, len(batteries) // 2)
    n_val = max(2, len(batteries) // 4)
    test_batteries = batteries[n_train + n_val:]
    
    test_df = feature_df[feature_df['battery_id'].isin(test_batteries)]
    feature_cols = [c for c in feature_df.columns 
                   if c not in ['battery_id', 'cycle', 'capacity', 'capacity']]
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    # 确定标签列
    label_col = 'capacity' if 'capacity' in test_df.columns else 'soh' if 'soh' in test_df.columns else None
    if label_col is None:
        raise ValueError(f"No label column found. Available columns: {list(test_df.columns)}")
    
    y_test = test_df[label_col].values.astype(np.float32)
    
    # 处理无效值
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    if np.isinf(X_test).any():
        X_test = np.nan_to_num(X_test, posinf=0.0, neginf=0.0)
    
    # 标准化
    if train_mean is not None and train_std is not None:
        X_test = (X_test - train_mean) / train_std
    
    # 生成缺失数据（根据mode和test_mr）
    np.random.seed(args.seed + 1000 + int(args.test_mr * 100))
    
    if args.mode == 'MCAR':
        # MCAR: 完全随机缺失，与任何变量无关
        mask = np.random.rand(*X_test.shape) < args.test_mr
        
    elif args.mode == 'MAR':
        # MAR: 缺失与观测到的数据相关
        # 实现：缺失概率与电压特征相关（电压越高，缺失概率越大）
        voltage_col = None
        for i, col in enumerate(feature_cols):
            if 'voltage' in col.lower():
                voltage_col = i
                break
        
        if voltage_col is not None:
            # 归一化电压到[0,1]
            voltage = X_test[:, voltage_col]
            voltage_norm = (voltage - voltage.min()) / (voltage.max() - voltage.min() + 1e-8)
            # 缺失概率与电压成正比
            missing_prob = args.test_mr * (0.5 + 0.5 * voltage_norm)  # 范围 [0.5*mr, 1.0*mr]
            mask = np.random.rand(*X_test.shape) < missing_prob[:, np.newaxis]
        else:
            # 如果没有电压特征，回退到MCAR
            mask = np.random.rand(*X_test.shape) < args.test_mr
            
    else:  # MNAR
        # MNAR: 缺失与缺失值本身或目标值相关
        # 实现：缺失概率与SOH（目标值）相关（SOH越高，缺失概率越大）
        soh = y_test
        soh_norm = (soh - soh.min()) / (soh.max() - soh.min() + 1e-8)
        # 缺失概率与SOH成正比
        missing_prob = args.test_mr * (0.5 + 0.5 * soh_norm)  # 范围 [0.5*mr, 1.0*mr]
        # 对每一行应用相同的缺失概率（每一样本的所有特征共享相同的缺失概率）
        mask = np.random.rand(*X_test.shape) < missing_prob[:, np.newaxis]
    
    X_test_missing = X_test.copy()
    X_test_missing[mask] = np.nan
    
    # 应用插补方法
    from sklearn.impute import SimpleImputer, KNNImputer
    
    if args.imputation == 'mean':
        imputer = SimpleImputer(strategy='mean')
        X_test_imputed = imputer.fit_transform(X_test_missing)
    elif args.imputation == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        X_test_imputed = imputer.fit_transform(X_test_missing)
    elif args.imputation == 'iterative':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=args.seed)
        X_test_imputed = imputer.fit_transform(X_test_missing)
    else:  # zero
        X_test_imputed = np.nan_to_num(X_test_missing, nan=0.0)
    
    # 准备输入
    X_test_tensor = torch.FloatTensor(X_test_imputed).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    if use_mim:
        # 拼接掩码
        mask_tensor = torch.FloatTensor(mask.astype(float)).to(device)
        X_test_tensor = torch.cat([X_test_tensor, mask_tensor], dim=1)
    
    # 对于CNN/LSTM模型，添加序列维度 [batch, 1, features]
    if args.model in ['cnn', 'lstm']:
        X_test_tensor = X_test_tensor.unsqueeze(1)
    
    # 预测
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze().cpu().numpy()
    
    # 计算指标
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test MAE: {mae:.6f}, R2: {r2:.6f}")
    
    result = {
        'seed': args.seed,
        'batch': args.batch,
        'model': args.model,
        'use_mim': args.use_mim,
        'mode': args.mode,
        'test_mr': args.test_mr,
        'imputation': args.imputation,
        'test_mae': mae,
        'test_r2': r2,
        'n_samples': len(y_test)
    }
    
    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return result


def main():
    args = parse_args()
    
    if args.phase == 'train':
        result = train_model(args)
    else:  # test
        result = test_model(args)
    
    return result


if __name__ == '__main__':
    main()
