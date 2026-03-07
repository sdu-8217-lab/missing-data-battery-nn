"""PyTorch Lightning模块 - 用于src/main.py的兼容性"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Any, Optional

from .model_factory import create_model


class BatterySOHModule(pl.LightningModule):
    """电池SOH预测的Lightning模块
    
    兼容src/main.py中的引用，包装模型以适配PyTorch Lightning。
    """
    
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **model_kwargs
    ):
        """
        Args:
            model_type: 模型类型 ('mlp', 'lstm', 'gru', 'cnn1d')
            input_dim: 输入维度
            lr: 学习率
            weight_decay: 权重衰减
            **model_kwargs: 模型特定参数
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 创建模型
        self.model = create_model(model_type, input_dim, **model_kwargs)
        self.criterion = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.criterion(y_hat, y)
        
        # 计算MAE
        mae = torch.mean(torch.abs(y_hat - y))
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_mae', mae, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.criterion(y_hat, y)
        
        # 计算MAE
        mae = torch.mean(torch.abs(y_hat - y))
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mae', mae, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_mae': mae}
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.criterion(y_hat, y)
        
        # 计算指标
        mae = torch.mean(torch.abs(y_hat - y))
        rmse = torch.sqrt(loss)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        ss_res = torch.sum((y - y_hat) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_mae', mae, on_epoch=True)
        self.log('test_rmse', rmse, on_epoch=True)
        self.log('test_r2', r2, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
