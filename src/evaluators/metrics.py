"""评估指标"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred):
    """计算MAE"""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """计算R²"""
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true, y_pred):
    """计算所有指标"""
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }
