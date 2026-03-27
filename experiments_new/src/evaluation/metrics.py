"""
评估指标
"""
import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        包含MAE, RMSE, R2的字典
    """
    # 确保是1D数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }


def calculate_improvement_rate(
    baseline_mae: float,
    mim_mae: float
) -> float:
    """
    计算改进率
    
    参数:
        baseline_mae: 基线MAE
        mim_mae: MIM MAE
    
    返回:
        改进率（百分比）
    """
    if baseline_mae == 0:
        return 0.0
    return ((baseline_mae - mim_mae) / baseline_mae) * 100


def summarize_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    汇总多个实验的指标
    
    参数:
        metrics_list: 指标字典列表
    
    返回:
        包含均值、标准差、最小值、最大值的字典
    """
    if not metrics_list:
        return {}
    
    result = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            result[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return result
