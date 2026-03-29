#!/usr/bin/env python3
"""
交叉插补分析脚本
评估训练插补方法 × 测试插补方法的16种组合
"""
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig, load_model_config
from src.data.loader import XJTUDatasetLoader, prepare_datasets_for_training
from src.models.factory import create_model


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_with_imputation(
    model,
    config: ExperimentConfig,
    test_loader,
    imputation_method: str,
    device: str
) -> dict:
    """
    使用特定插补方法评估模型
    
    Returns:
        metrics: 包含MAE, RMSE, MAPE的字典
    """
    from src.data.imputation import Imputer
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # 收集所有测试数据用于拟合imputer
    all_X_test = []
    for X_batch, y_batch in test_loader:
        all_X_test.append(X_batch.cpu().numpy())
    X_test_full = np.concatenate(all_X_test, axis=0)
    
    imputer = Imputer(method=imputation_method)
    imputer.fit(X_test_full)  # 拟合插补器
    missing_rates = config.evaluation.missing_rates
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            # 对每个缺失率生成缺失数据并评估
            for mr in missing_rates:
                rng = np.random.default_rng(42)
                mask = rng.random(X_batch.shape) < mr
                X_imputed = imputer.transform(X_batch, mask)
                
                # 构造序列输入
                if model.use_mim:
                    missing_indicators = mask.astype(np.float32)
                    X_combined = np.concatenate([X_imputed, missing_indicators], axis=1)
                else:
                    X_combined = X_imputed
                
                # 转换为序列 (batch, seq_len, features)
                X_tensor = torch.FloatTensor(X_combined).to(device)
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1).expand(-1, config.training.seq_len, -1)
                
                predictions = model(X_tensor).cpu().numpy()
                
                all_predictions.extend(predictions.tolist() if predictions.ndim > 0 else [predictions.item()])
                all_targets.extend(y_batch.tolist() if y_batch.ndim > 0 else [y_batch.item()])
    
    # 计算指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def evaluate_cross_combinations(
    base_config: ExperimentConfig,
    model_arch_config: dict,
    model_dir: Path,
    output_dir: Path,
    seeds: list,
    train_imputations: list,
    test_imputations: list,
    logger
):
    """
    评估所有训练插补 × 测试插补的组合
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = model_arch_config.get('model_type', 'lstm')
    use_mim = base_config.training.use_mim
    
    results = []
    total = len(seeds) * len(train_imputations) * len(test_imputations)
    counter = 0
    
    # 加载测试数据
    loader = XJTUDatasetLoader(
        data_dir=base_config.data.data_dir,
        batch=base_config.data.batch
    )
    data_dict = loader.prepare_data(
        feature_cols=base_config.data.feature_cols,
        target_col=base_config.data.target_col,
        test_size=base_config.data.test_size,
        val_size=base_config.data.val_size,
        random_seed=42
    )
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    
    from torch.utils.data import TensorDataset, DataLoader
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=base_config.training.batch_size,
        shuffle=False
    )
    
    for seed in seeds:
        for train_imp in train_imputations:
            for test_imp in test_imputations:
                counter += 1
                logger.info(f"[{counter}/{total}] 评估: seed={seed}, train={train_imp}, test={test_imp}")
                
                # 加载模型
                model_name = f"model_{base_config.data.batch}_{model_type}_mim{use_mim}_{train_imp}_seed{seed}"
                model_path = model_dir / f"{model_name}.pt"
                
                if not model_path.exists():
                    logger.warning(f"模型不存在: {model_path}")
                    continue
                
                # 创建并加载模型
                input_dim = len(base_config.data.feature_cols)
                model_kwargs = {k: v for k, v in model_arch_config.items() 
                               if k not in ['model_type', 'name', 'level']}
                
                model = create_model(
                    model_type=model_type,
                    input_dim=input_dim,
                    use_mim=use_mim,
                    device=device,
                    **model_kwargs
                )
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                
                # 评估
                metrics = evaluate_with_imputation(
                    model, base_config, test_loader, test_imp, device
                )
                
                results.append({
                    'seed': seed,
                    'train_imputation': train_imp,
                    'test_imputation': test_imp,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape']
                })
                
                logger.info(f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_path = output_dir / 'cross_imputation_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n结果已保存: {output_path}")
    
    # 计算汇总统计
    summary = results_df.groupby(['train_imputation', 'test_imputation']).agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std']
    }).round(4)
    
    summary_path = output_dir / 'cross_imputation_summary.csv'
    summary.to_csv(summary_path)
    logger.info(f"汇总已保存: {summary_path}")
    
    # 打印表格
    print("\n" + "="*80)
    print("交叉插补评估结果 (MAE)")
    print("="*80)
    pivot = results_df.pivot_table(
        values='mae', 
        index='train_imputation', 
        columns='test_imputation',
        aggfunc='mean'
    )
    print(pivot)
    print("="*80)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="交叉插补分析")
    parser.add_argument("--config", required=True, help="实验配置文件")
    parser.add_argument("--model-config", required=True, help="模型架构配置文件")
    parser.add_argument("--model-dir", required=True, help="模型目录")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="随机种子列表")
    parser.add_argument("--train-imputations", nargs="+", default=['zero', 'mean', 'knn', 'iterative'])
    parser.add_argument("--test-imputations", nargs="+", default=['zero', 'mean', 'knn', 'iterative'])
    parser.add_argument("--output-dir", default="results/multi_imp")
    
    args = parser.parse_args()
    
    from src.utils.logger import setup_logger
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        'cross_imputation',
        log_file=output_dir / 'cross_imputation.log',
        level='INFO'
    )
    
    base_config = ExperimentConfig.from_yaml(args.config)
    model_arch_config = load_model_config(args.model_config)
    
    results = evaluate_cross_combinations(
        base_config, model_arch_config,
        Path(args.model_dir), output_dir,
        args.seeds, args.train_imputations, args.test_imputations,
        logger
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
