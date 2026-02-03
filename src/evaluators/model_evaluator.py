"""模型评估器"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import calculate_all_metrics


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: 计算设备
        """
        self.device = device
    
    def evaluate(self, model, test_loader: DataLoader) -> dict:
        """
        评估PyTorch模型
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            
        Returns:
            dict: 评估指标
        """
        if hasattr(model, 'model'):
            model = model.model
        
        model.eval()
        predictions = []
        targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                # 确保输出是1维数组，避免0维标量问题
                outputs_flat = outputs.cpu().numpy().reshape(-1)
                targets_flat = batch_y.numpy().reshape(-1)
                predictions.extend(outputs_flat)
                targets.extend(targets_flat)
        
        inference_time = time.time() - start_time
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = calculate_all_metrics(targets, predictions)
        metrics['inference_time'] = inference_time
        metrics['n_samples'] = len(targets)
        
        return metrics, predictions, targets
    
    def evaluate_xgboost(self, model, X_test, y_test) -> dict:
        """
        评估XGBoost模型
        
        Args:
            model: XGBoost模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 评估指标
        """
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        
        metrics = calculate_all_metrics(y_test, predictions)
        metrics['inference_time'] = inference_time
        metrics['n_samples'] = len(y_test)
        
        return metrics, predictions, y_test
