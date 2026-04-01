#!/usr/bin/env python3
"""
从 .pt 模型文件重新计算 MAE, RMSE, R² 指标
并更新 train_results.json

使用方式:
    cd paper/plotting_scripts
    python3 recalculate_metrics.py
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.factory import create_model
from src.data.dataset_loader import XJTUDatasetLoader

# 默认特征列 (XJTU数据)
DEFAULT_FEATURE_COLS = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness',
    'CV Q', 'CV charge time', 'current slope', 'current entropy'
]


def load_battery_data(batch='2C', seed=42, use_mim=False):
    """加载电池数据，返回测试集
    
    Args:
        batch: 批次名称
        seed: 随机种子
        use_mim: 是否使用MIM模式（影响输入维度）
    """
    data_dir = PROJECT_ROOT / 'data' / 'XJTU data'
    
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")
    
    # 使用 XJTUDatasetLoader 加载数据
    loader = XJTUDatasetLoader(str(data_dir), batch)
    
    if use_mim:
        # MIM模式：加载两次特征（原始 + 缺失标记）
        # 简化处理：使用原始特征拼接一个掩码特征
        batteries = loader.load_all_batteries(DEFAULT_FEATURE_COLS, 'capacity')
        
        # 为每个电池数据添加缺失特征掩码（16维掩码，初始为1）
        for bid in batteries:
            X, y = batteries[bid]
            # 拼接原始特征和掩码特征（总32维）
            mask = np.ones_like(X)  # 缺失掩码，1表示有值
            X_mim = np.concatenate([X, mask], axis=1)
            batteries[bid] = (X_mim, y)
    else:
        # 非MIM模式：只加载原始特征
        batteries = loader.load_all_batteries(DEFAULT_FEATURE_COLS, 'capacity')
    
    # 划分数据集
    battery_ids = list(batteries.keys())
    train_ids, val_ids, test_ids = loader.split_batteries(
        battery_ids, test_size=0.25, val_size=0.25, random_seed=seed
    )
    
    # 只返回测试集
    test_data = {bid: batteries[bid] for bid in test_ids}
    return test_data


def create_test_loader(test_data, batch_size=32):
    """创建测试 DataLoader"""
    X_test = np.concatenate([data[0] for data in test_data.values()], axis=0)
    y_test = np.concatenate([data[1] for data in test_data.values()], axis=0)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader


def evaluate_model(model, test_loader, device, model_type='mlp'):
    """评估模型，计算 MAE, RMSE, R², MSE"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            # CNN/LSTM 需要 3D 输入 [batch, 1, features]
            if model_type in ['cnn', 'lstm']:
                X = X.unsqueeze(1)
            
            output = model(X)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # 计算指标
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }


def load_model_from_checkpoint(pt_path, device):
    """从 checkpoint 加载模型（使用保存的配置）"""
    checkpoint = torch.load(pt_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'mlp')
    use_mim = checkpoint.get('config', {}).get('training', {}).get('use_mim', False)
    input_dim = 32 if use_mim else 16
    
    # 根据模型类型创建模型（使用保存的配置）
    if model_type == 'mlp':
        from src.models.mlp import MLP
        model = MLP(
            input_dim=input_dim,
            hidden_dims=model_config.get('hidden_dims', [120, 80]),
            dropout=model_config.get('dropout', 0.15)
        )
    elif model_type == 'cnn':
        from src.models.cnn1d import CNN1D
        model = CNN1D(
            input_dim=input_dim,
            channels=model_config.get('channels', [72, 40]),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'lstm':
        from src.models.lstm import LSTM
        model = LSTM(
            input_dim=input_dim,
            hidden_size=model_config.get('hidden_size', 44),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, model_type, checkpoint


def find_train_results_json(exp_dir):
    """查找 train_results.json 文件"""
    # 直接查找
    direct_path = exp_dir / "train_results.json"
    if direct_path.exists():
        return direct_path
    
    # 在子目录中查找
    for subdir in exp_dir.iterdir():
        if subdir.is_dir():
            path = subdir / "train_results.json"
            if path.exists():
                return path
    
    return None


def update_train_results(exp_dir, metrics, checkpoint=None):
    """更新或创建 train_results.json"""
    results_file = find_train_results_json(exp_dir)
    
    if results_file is None:
        # 创建新的 results 文件
        results_file = exp_dir / "train_results.json"
        results = {}
        
        # 从 checkpoint 中提取基本信息
        if checkpoint:
            results['experiment_name'] = checkpoint.get('config', {}).get('experiment_name', '')
            results['model_type'] = checkpoint.get('model_config', {}).get('model_type', '')
            results['level'] = checkpoint.get('model_config', {}).get('level', '')
            results['use_mim'] = checkpoint.get('config', {}).get('training', {}).get('use_mim', False)
            results['seed'] = checkpoint.get('seed', 0)
            results['param_count'] = checkpoint.get('param_count', 0)
            results['best_val_loss'] = checkpoint.get('best_val_loss', 0)
            results['test_loss'] = checkpoint.get('test_loss', 0)
    else:
        # 更新现有文件
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 备份原文件
        backup_file = results_file.parent / "train_results_backup.json"
        if not backup_file.exists():
            import shutil
            shutil.copy(results_file, backup_file)
    
    # 添加/更新指标
    results['mae'] = metrics['mae']
    results['mse'] = metrics['mse']
    results['rmse'] = metrics['rmse']
    results['r2'] = metrics['r2']
    results['mape'] = metrics['mape']
    results['metrics_calculated'] = True
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True


def process_all_models():
    """处理所有模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 实验数据目录
    exp_data_dir = Path(__file__).parent.parent / "experiment_data"
    
    # 查找所有 .pt 文件
    pt_files = list(exp_data_dir.rglob("*.pt"))
    print(f"找到 {len(pt_files)} 个模型文件")
    print(f"{'='*60}\n")
    
    success_count = 0
    error_count = 0
    
    for i, pt_file in enumerate(pt_files, 1):
        exp_name = pt_file.parent.parent.name
        print(f"[{i}/{len(pt_files)}] {exp_name}")
        
        try:
            # 找到对应的实验目录
            exp_dir = pt_file.parent.parent
            
            # 加载模型
            model, model_type, checkpoint = load_model_from_checkpoint(pt_file, device)
            
            # 获取批次信息
            batch = '2C'  # 默认批次
            if '3C' in str(pt_file):
                batch = '3C'
            
            # 加载数据（只加载测试集）
            seed = checkpoint.get('seed', 42)
            use_mim = checkpoint.get('config', {}).get('training', {}).get('use_mim', False)
            test_data = load_battery_data(batch, seed=seed, use_mim=use_mim)
            test_loader = create_test_loader(test_data)
            
            # 评估
            metrics = evaluate_model(model, test_loader, device, model_type)
            
            print(f"    MAE: {metrics['mae']:.6f}  RMSE: {metrics['rmse']:.6f}  R²: {metrics['r2']:.4f}  MAPE: {metrics['mape']:.2f}%")
            
            # 更新结果文件
            if update_train_results(exp_dir, metrics, checkpoint):
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"    ❌ 错误: {e}")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"处理完成: 成功 {success_count}/{len(pt_files)}, 失败 {error_count}/{len(pt_files)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    process_all_models()
