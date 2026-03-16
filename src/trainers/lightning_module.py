"""PyTorch Lightning模块定义 - 统一版本"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, SGD
from typing import Dict, Any, Optional

from ..models.model_factory import create_model


class SOHLightningModule(pl.LightningModule):
    """
    SOH预测的Lightning模块 - 统一版本
    
    支持两种初始化方式:
    1. 传入已创建的模型: SOHLightningModule(model=nn.Module, ...)
    2. 传入模型参数: SOHLightningModule(model_type='mlp', input_dim=16, ...)
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_type: Optional[str] = None,
        input_dim: Optional[int] = None,
        learning_rate: float = 1e-3,
        lr: Optional[float] = None,  # 兼容旧接口
        optimizer_name: str = "Adam",
        weight_decay: float = 0.0,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        **model_kwargs
    ):
        """
        Args:
            model: 已创建的PyTorch模型（方式1）
            model_type: 模型类型如'mlp','lstm'等（方式2）
            input_dim: 输入维度（方式2）
            learning_rate/lr: 学习率（lr为兼容旧接口）
            optimizer_name: 优化器名称
            weight_decay: 权重衰减
            scheduler_patience: 学习率调度耐心值
            scheduler_factor: 学习率衰减因子
            **model_kwargs: 模型特定参数（方式2）
        """
        super().__init__()
        
        # 兼容旧接口：lr 优先级高于 learning_rate
        lr_value = lr if lr is not None else learning_rate
        
        if model is not None:
            # 方式1：使用传入的模型
            self.model = model
            self.save_hyperparameters(ignore=['model'])
        elif model_type is not None and input_dim is not None:
            # 方式2：内部创建模型
            self.model = create_model(model_type, input_dim, **model_kwargs)
            self.save_hyperparameters()
        else:
            raise ValueError("必须提供 model 参数，或提供 model_type + input_dim 参数")
        
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


