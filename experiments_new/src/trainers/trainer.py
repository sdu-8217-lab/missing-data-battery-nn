"""
神经网络训练器
支持早停、学习率调度和多MR验证聚合 (META §2.5)
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Union
import numpy as np


class Trainer:
    """
    神经网络训练器
    
    支持功能:
    - 早停 (Early Stopping)
    - 学习率调度 (ReduceLROnPlateau)
    - 多MR验证聚合 (META §2.5)
    - 自动设备检测
    - 训练历史记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        use_mixed_precision: bool = False
    ):
        """
        参数:
            model: PyTorch模型
            device: 计算设备 ('auto', 'cuda', 'cpu')
            use_mixed_precision: 是否使用混合精度训练
        """
        self.model = model
        
        # 自动检测设备
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 混合精度
        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision and device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loaders: Union[DataLoader, Dict[str, DataLoader]],
        epochs: int = 200,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 30,
        min_epochs: int = 10,
        use_scheduler: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        base_metric: str = "mae",
        aggregation: str = "mean",
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        训练模型（支持多MR验证）
        
        参数:
            train_loader: 训练数据加载器
            val_loaders: 验证数据加载器或字典 {mr: loader}
            epochs: 最大训练轮数
            lr: 学习率
            weight_decay: 权重衰减 (META §6.3: 建议1e-5~1e-4)
            patience: 早停耐心值 (META §6.2: 建议20-50)
            min_epochs: 最小训练轮数
            use_scheduler: 是否使用学习率调度
            scheduler_patience: 调度器耐心值
            scheduler_factor: 学习率衰减因子
            base_metric: 基础指标 ("mae", "mse", "rmse")
            aggregation: 聚合方式 ("mean", "max", "min", "median", "worst")
            verbose: 是否打印进度
        
        返回:
            训练历史字典
        """
        # 统一验证加载器格式
        if isinstance(val_loaders, DataLoader):
            val_loaders = {"single": val_loaders}
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience
            )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],  # 聚合后的验证损失
            'val_mae': [],   # 聚合后的验证MAE
            'val_metrics': {},  # 各MR的详细指标
            'lr': []
        }
        
        # 早停相关
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_mae = self._train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)
            
            # 验证阶段（多MR）
            val_results = {}
            for mr_name, val_loader in val_loaders.items():
                val_loss, val_mae = self._validate_epoch(val_loader)
                val_results[mr_name] = {
                    'loss': val_loss,
                    'mae': val_mae
                }
            
            # 聚合验证指标
            val_losses = [r['loss'] for r in val_results.values()]
            val_maes = [r['mae'] for r in val_results.values()]
            
            aggregated_loss = self._aggregate_metrics(val_losses, aggregation)
            aggregated_mae = self._aggregate_metrics(val_maes, aggregation)
            
            history['val_loss'].append(aggregated_loss)
            history['val_mae'].append(aggregated_mae)
            history['val_metrics'][epoch] = val_results
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step(aggregated_loss)
            
            # 早停检查
            if aggregated_loss < best_val_loss:
                best_val_loss = aggregated_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                mr_details = ", ".join([f"{k}={v['mae']:.4f}" for k, v in val_results.items()])
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train: {train_loss:.6f}, "
                      f"Val({aggregation}): {aggregated_mae:.6f}, "
                      f"LR: {current_lr:.6f}")
                if len(val_results) > 1:
                    print(f"  MR details: {mr_details}")
            
            # 早停判断（超过最小轮数后）
            if epoch >= min_epochs and patience_counter >= patience:
                if verbose:
                    print(f"早停于 epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        history['training_time'] = training_time
        history['best_epoch'] = len(history['train_loss']) - patience_counter
        history['best_val_loss'] = best_val_loss
        history['best_val_mae'] = history['val_mae'][history['best_epoch'] - 1]
        
        return history
    
    def _aggregate_metrics(self, values: List[float], method: str) -> float:
        """
        聚合多个指标值
        
        参数:
            values: 指标值列表
            method: 聚合方法 ("mean", "max", "min", "median", "worst")
        
        返回:
            聚合后的值
        """
        if method == "mean":
            return np.mean(values)
        elif method == "max":
            return np.max(values)
        elif method == "min":
            return np.min(values)
        elif method == "median":
            return np.median(values)
        elif method == "worst":
            # 对于损失/误差，worst = max
            return np.max(values)
        else:
            raise ValueError(f"未知的聚合方法: {method}")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> tuple:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)
                    outputs = outputs.view(-1)
                    batch_y = batch_y.view(-1)
                    loss = self.criterion(outputs, batch_y)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_x)
                outputs = outputs.view(-1)
                batch_y = batch_y.view(-1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 统计
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_size
            n_samples += batch_size
        
        return total_loss / n_samples, total_mae / n_samples
    
    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                outputs = outputs.view(-1)
                batch_y = batch_y.view(-1)
                loss = self.criterion(outputs, batch_y)
                
                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_size
                n_samples += batch_size
        
        return total_loss / n_samples, total_mae / n_samples
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
            data_loader: 数据加载器
        
        返回:
            预测结果numpy数组
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)
