"""PyTorch Lightning模块定义"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, SGD
from typing import Dict, Any, Optional


class SOHLightningModule(pl.LightningModule):
    """
    SOH预测的Lightning模块
    
    封装PyTorch模型，提供标准化的训练/验证/测试流程
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        weight_decay: float = 0.0,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        """
        Args:
            model: PyTorch模型
            learning_rate: 学习率
            optimizer_name: 优化器名称 (Adam, AdamW, SGD)
            weight_decay: 权重衰减
            scheduler_patience: 学习率调度耐心值
            scheduler_factor: 学习率衰减因子
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def _compute_loss(self, batch):
        """计算损失"""
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.squeeze()
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss, y_hat, y = self._compute_loss(batch)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # 计算MAE用于监控
        mae = torch.mean(torch.abs(y_hat - y))
        self.log('train_mae', mae, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        loss, y_hat, y = self._compute_loss(batch)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        mae = torch.mean(torch.abs(y_hat - y))
        self.log('val_mae', mae, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_mae': mae}
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss, y_hat, y = self._compute_loss(batch)
        
        self.log('test_loss', loss, on_epoch=True)
        
        mae = torch.mean(torch.abs(y_hat - y))
        self.log('test_mae', mae, on_epoch=True)
        
        return {'test_loss': loss, 'test_mae': mae}
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 选择优化器
        optimizer_class = {
            'Adam': Adam,
            'AdamW': AdamW,
            'SGD': SGD,
        }.get(self.hparams.optimizer_name, Adam)
        
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # 学习率调度器 - ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        x, _ = batch
        y_hat = self(x).squeeze()
        return y_hat


