"""评估指标模块

提供用于回归任务的评估指标计算功能。
"""

import logging
from typing import Dict, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置日志记录器
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算回归评估指标
    
    计算 MAE、RMSE 和 R² 三个常用回归指标。
    
    Args:
        y_true: 真实目标值，形状为 [N,] 或 [N, 1]
        y_pred: 预测目标值，形状为 [N,] 或 [N, 1]
        
    Returns:
        包含以下键的字典:
            - mae: 平均绝对误差 (Mean Absolute Error)
            - rmse: 均方根误差 (Root Mean Squared Error)
            - r2: 决定系数 (R-squared)
            
    Raises:
        ValueError: 当输入数组为空或形状不匹配时
        
    Example:
        >>> import numpy as np
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    """
    try:
        # 验证输入
        if y_true is None or y_pred is None:
            raise ValueError("y_true 和 y_pred 不能为 None")
        
        # 展平数组以确保一维
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        # 检查长度
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 和 y_pred 长度不匹配: {len(y_true)} vs {len(y_pred)}"
            )
        
        # 检查空数组
        if len(y_true) == 0:
            raise ValueError("输入数组不能为空")
        
        # 检查 NaN 和 Inf
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            logger.warning("输入数据包含 NaN 值，将在计算前移除")
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            logger.warning("输入数据包含 Inf 值，将在计算前移除")
            mask = ~(np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        # 计算指标
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        
        # 记录指标
        logger.debug(
            f"Metrics computed: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.6f}"
        )
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
        
    except Exception as e:
        logger.error(f"计算指标时发生错误: {e}")
        raise


def compute_additional_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算额外的回归评估指标
    
    包括最大误差、中位数绝对误差等补充指标。
    
    Args:
        y_true: 真实目标值
        y_pred: 预测目标值
        
    Returns:
        包含额外指标的字典:
            - max_error: 最大绝对误差
            - median_ae: 中位数绝对误差
            - mape: 平均绝对百分比误差 (处理 y_true=0 时返回 inf)
            - explained_variance: 解释方差
    """
    try:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        # 基础指标
        errors = np.abs(y_true - y_pred)
        max_error = float(np.max(errors))
        median_ae = float(np.median(errors))
        
        # MAPE (处理零值)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100)
        
        # 解释方差
        explained_variance = float(
            1 - np.var(y_true - y_pred) / np.var(y_true)
            if np.var(y_true) > 0 else 0.0
        )
        
        return {
            "max_error": max_error,
            "median_ae": median_ae,
            "mape": mape,
            "explained_variance": explained_variance,
        }
        
    except Exception as e:
        logger.error(f"计算额外指标时发生错误: {e}")
        raise
