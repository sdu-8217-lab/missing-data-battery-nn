#!/usr/bin/env python3
"""
九宫格实验后评估脚本
为所有216个模型生成测试指标
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.mlp import MLP
from src.models.cnn1d import CNN1D
from src.models.lstm import LSTM
from src.data.dataset_loader import XJTUDatasetLoader


def load_model_from_checkpoint(pt_path, device='cpu'):
    """从checkpoint动态加载模型"""
    checkpoint = torch.load(pt_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'mlp')
    use_mim = checkpoint.get('config', {}).get('training', {}).get('use_mim', False)
    input_dim = 32 if use_mim else 16
    
    if model_type == 'mlp':
        model = MLP(
            input_dim=input_dim,
            hidden_dims=model_config.get('hidden_dims', [120, 80]),
            dropout=model_config.get('dropout', 0.15)
        )
    elif model_type == 'cnn':
        model = CNN1D(
            input_dim=input_dim,
            channels=model_config.get('channels', [72, 40]),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'lstm':
        model = LSTM(
            input_dim=input_dim,
            hidden_size=model_config.get('hidden_size', 44),
            num_layers=model_config.get('num_layers', 1),
            dropout=model_config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


def create_missing_data(X, missing_rate, pattern='MCAR', seed=42):
    """生成缺失数据"""
    rng = np.random.RandomState(seed)
    if pattern == 'MCAR':
        mask = rng.rand(*X.shape) < missing_rate
    elif pattern == 'MAR':
        mask = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(0, X.shape[1], 2):
                if j+1 < X.shape[1]:
                    threshold = np.percentile(X[i, j], 40)
                    if X[i, j] < threshold:
                        mask[i, j+1] = rng.rand() < missing_rate * 2
    else:  # MNAR
        mask = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[0]):
            threshold = np.percentile(X[i], 50)
            low_val_mask = X[i] < threshold
            mask[i] = low_val_mask & (rng.rand(*X[i].shape) < missing_rate * 2)
    
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def apply_imputation(X_missing, method='zero'):
    """应用插补方法"""
    X_imputed = X_missing.copy()
    if method == 'zero':
        X_imputed = np.nan_to_num(X_imputed, nan=0.0)
    elif method == 'mean':
        col_means = np.nanmean(X_imputed, axis=0)
        for i in range(X_imputed.shape[1]):
            X_imputed[np.isnan(X_imputed[:, i]), i] = col_means[i]
    elif method == 'iterative':
        X_imputed = np.nan_to_num(X_imputed, nan=0.0)
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3, weights='distance')
        X_imputed = imputer.fit_transform(X_imputed)
    return X_imputed


def evaluate_model(model, X_test, y_test, batch_size=256):
    """评估模型性能"""
    device = next(model.parameters()).device
    predictions = []
    
    # 检查模型类型并调整输入
    model_type = model.__class__.__name__
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            
            # CNN 需要调整输入维度: (batch, features) -> (batch, channels, seq_len)
            if model_type == 'CNN1D':
                batch_x = batch_x.unsqueeze(1)  # (batch, 1, features)
            # LSTM 需要调整输入维度: (batch, features) -> (seq_len, batch, features)
            elif model_type == 'LSTM':
                batch_x = batch_x.unsqueeze(0)  # (1, batch, features)
            
            pred = model(batch_x).cpu().numpy()
            
            # 处理不同输出形状
            if pred.ndim > 1:
                pred = pred.flatten()
            predictions.extend(pred)
    
    predictions = np.array(predictions)
    y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    
    # 确保长度一致
    min_len = min(len(predictions), len(y_true))
    predictions = predictions[:min_len]
    y_true = y_true[:min_len]
    
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def process_all_models():
    """处理所有九宫格模型"""
    data_dir = PROJECT_ROOT / "paper" / "experiment_data" / "nine_grid_complete"
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 加载测试数据
    print("加载测试数据...")
    loader = XJTUDatasetLoader(
        data_dir=PROJECT_ROOT / "data" / "XJTU data",
        batch='2C'
    )
    
    # 使用prepare_data加载数据
    feature_cols = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ]
    data = loader.prepare_data(
        feature_cols=feature_cols,
        target_col='capacity',
        test_size=0.2,
        val_size=0.1,
        random_seed=42
    )
    X_test = data['X_test']
    y_test = data['y_test']
    
    model_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('nine_')]
    print(f"找到 {len(model_dirs)} 个模型目录")
    
    results_summary = []
    
    for idx, model_dir in enumerate(model_dirs, 1):
        print(f"\n[{idx}/{len(model_dirs)}] 处理: {model_dir.name}")
        
        # 解析目录名获取信息
        parts = model_dir.name.split('_')
        arch = parts[1]
        pattern = parts[4]
        imputation = parts[5]
        config = parts[6]
        seed = int(parts[7].replace('seed', ''))
        
        # 找到.pt文件
        pt_files = list(model_dir.rglob("*.pt"))
        if not pt_files:
            print(f"  ⚠️ 未找到模型文件")
            continue
        
        pt_path = pt_files[0]
        
        try:
            # 加载模型
            model, checkpoint = load_model_from_checkpoint(pt_path)
            use_mim = checkpoint.get('config', {}).get('training', {}).get('use_mim', False)
            
            # 准备测试数据
            missing_rates = [0.05 * i for i in range(1, 20)]
            mr_results = {}
            
            for mr in missing_rates:
                X_missing, mask = create_missing_data(X_test, mr, pattern, seed)
                X_imputed = apply_imputation(X_missing, imputation)
                
                if use_mim:
                    X_input = np.concatenate([X_imputed, mask.astype(float)], axis=1)
                else:
                    X_input = X_imputed
                
                metrics = evaluate_model(model, X_input, y_test)
                mr_results[f"{mr:.2f}"] = metrics
            
            # 保存结果
            result_file = model_dir / "evaluated_metrics.json"
            with open(result_file, 'w') as f:
                json.dump(mr_results, f, indent=2)
            
            # 记录摘要
            avg_mae = np.mean([m['MAE'] for m in mr_results.values()])
            results_summary.append({
                'model': model_dir.name,
                'arch': arch,
                'pattern': pattern,
                'imputation': imputation,
                'config': config,
                'seed': seed,
                'avg_mae': avg_mae,
                'status': 'success'
            })
            print(f"  ✅ 完成, 平均 MAE: {avg_mae:.4f}")
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")
            results_summary.append({
                'model': model_dir.name,
                'arch': arch,
                'pattern': pattern,
                'imputation': imputation,
                'config': config,
                'seed': seed,
                'status': f'error: {str(e)[:50]}'
            })
    
    # 保存汇总
    summary_file = data_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    success_count = sum(1 for r in results_summary if r['status'] == 'success')
    print(f"\n{'='*60}")
    print(f"评估完成: {success_count}/{len(model_dirs)} 成功")
    print(f"汇总保存: {summary_file}")


if __name__ == '__main__':
    process_all_models()
