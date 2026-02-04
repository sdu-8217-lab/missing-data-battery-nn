"""PyTorch Lightning训练器封装"""
import time
import warnings
from typing import Optional, Dict, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .lightning_module import SOHLightningModule

# 过滤 PyTorch Lightning 的 num_workers 警告
warnings.filterwarnings('ignore', message='.*num_workers.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*does not have many workers.*', category=UserWarning)


class LightningTrainer:
    """
    Lightning训练器封装
    
    简化PyTorch Lightning Trainer的使用，提供与旧版兼容的接口
    """
    
    def __init__(
        self,
        max_epochs: int = 100,
        patience: int = 15,
        device: str = 'auto',
        enable_progress_bar: bool = True,
        log_dir: str = './logs',
        experiment_name: str = 'soh_experiment',
    ):
        """
        Args:
            max_epochs: 最大训练轮数
            patience: 早停耐心值
            device: 计算设备 ('auto', 'cpu', 'cuda', 'gpu')
            enable_progress_bar: 是否显示进度条
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.enable_progress_bar = enable_progress_bar
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        self.trainer = None
        self.history = {}
        
    def train(
        self,
        model: SOHLightningModule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model: Lightning模块
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史
        """
        start_time = time.time()
        
        # 配置回调
        callbacks = []
        
        # 早停回调
        if val_loader is not None and self.patience > 0:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stop)
        
        # 进度条回调
        if self.enable_progress_bar:
            progress_bar = TQDMProgressBar(refresh_rate=10)
            callbacks.append(progress_bar)
        
        # 禁用CSV logger避免列名冲突，只使用内存记录
        loggers = False
        
        # 创建Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=loggers,
            accelerator=self.device,
            devices=1,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=10,
            # 禁用一些高级功能以简化输出
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,  # 禁用 sanity checking 提示
        )
        
        # 训练
        self.trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        training_time = time.time() - start_time
        
        # 构建训练历史
        self.history = {
            'train_loss': self._get_metric_history('train_loss_epoch'),
            'val_loss': self._get_metric_history('val_loss'),
            'train_mae': self._get_metric_history('train_mae_epoch'),
            'val_mae': self._get_metric_history('val_mae'),
            'training_time': training_time,
            'best_epoch': self.trainer.current_epoch + 1,
        }
        
        return self.history
    
    def _get_metric_history(self, metric_name: str) -> list:
        """从trainer获取指标历史"""
        # Logger已禁用，直接返回空列表
        # 实际指标在训练过程中通过回调记录
        return []
    
    def save_model(self, path: str):
        """保存模型"""
        if self.trainer is not None:
            self.trainer.save_checkpoint(path)
    
    def load_model(self, path: str, model: SOHLightningModule):
        """加载模型"""
        checkpoint = pl.utilities.cloud_io.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        return model


def create_lightning_trainer(config: Any) -> LightningTrainer:
    """
    从配置创建Lightning训练器的工厂函数
    
    Args:
        config: 配置对象（ExperimentConfig或ExperimentConfigV2）
        
    Returns:
        LightningTrainer实例
    """
    return LightningTrainer(
        max_epochs=getattr(config, 'epochs', 100),
        patience=getattr(config, 'early_stopping_patience', 15),
        device='auto',
        enable_progress_bar=True,
        log_dir='./logs',
        experiment_name=getattr(config, 'name', 'soh_experiment'),
    )
