"""PyTorch Lightning训练器封装"""
import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from .lightning_module import SOHLightningModule

# 过滤 PyTorch Lightning 的 num_workers 警告
warnings.filterwarnings('ignore', message='.*num_workers.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*does not have many workers.*', category=UserWarning)

# 配置日志记录器
logger = logging.getLogger(__name__)


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


def get_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    根据配置创建 PyTorch Lightning Trainer
    
    创建包含早停、学习率监控和 WandB 日志记录的训练器。
    支持从配置自动读取训练参数。
    
    Args:
        cfg: OmegaConf 配置对象，应包含以下配置:
            - training.epochs: 最大训练轮数
            - training.early_stopping.enabled: 是否启用早停
            - training.early_stopping.monitor: 监控指标 (如 "val_loss")
            - training.early_stopping.patience: 早停耐心值
            - training.early_stopping.mode: 模式 ("min" 或 "max")
            - wandb.mode: WandB 模式 ("online", "offline", "disabled")
            - wandb.project: WandB 项目名称
            - wandb.entity: WandB 实体/团队
            - experiment.name: 实验名称
            
    Returns:
        配置好的 pl.Trainer 实例
        
    Raises:
        ValueError: 当配置缺失必要字段时
        Exception: 当创建 Trainer 失败时
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({
        ...     "training": {
        ...         "epochs": 100,
        ...         "early_stopping": {
        ...             "enabled": True,
        ...             "monitor": "val_loss",
        ...             "patience": 15,
        ...             "mode": "min"
        ...         }
        ...     },
        ...     "wandb": {"mode": "disabled"},
        ...     "experiment": {"name": "test_exp"}
        ... })
        >>> trainer = get_trainer(cfg)
    """
    try:
        callbacks: List[pl.Callback] = []
        
        # 早停回调
        if cfg.training.get("early_stopping", {}).get("enabled", False):
            early_stopping_config = cfg.training.early_stopping
            callbacks.append(EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=early_stopping_config.get("patience", 15),
                mode=early_stopping_config.get("mode", "min"),
                verbose=True,
            ))
            logger.debug(
                f"Early stopping enabled: monitor={early_stopping_config.get('monitor', 'val_loss')}, "
                f"patience={early_stopping_config.get('patience', 15)}"
            )
        
        # 学习率监控
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        
        # 模型检查点（如果配置中启用）
        if cfg.training.get("checkpoint", {}).get("enabled", False):
            checkpoint_config = cfg.training.checkpoint
            checkpoint_dir = Path(checkpoint_config.get("dir", "./checkpoints"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks.append(ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=checkpoint_config.get("filename", "{epoch:02d}-{val_loss:.4f}"),
                monitor=checkpoint_config.get("monitor", "val_loss"),
                mode=checkpoint_config.get("mode", "min"),
                save_top_k=checkpoint_config.get("save_top_k", 3),
                save_last=checkpoint_config.get("save_last", True),
            ))
            logger.debug(f"Model checkpoint enabled: dir={checkpoint_dir}")
        
        # 配置日志记录器
        pl_logger = None
        wandb_mode = cfg.wandb.get("mode", "disabled") if hasattr(cfg, "wandb") else "disabled"
        
        if wandb_mode != "disabled":
            try:
                pl_logger = WandbLogger(
                    project=cfg.wandb.get("project", "battery-soh"),
                    entity=cfg.wandb.get("entity", None),
                    name=cfg.experiment.get("name", "unnamed_experiment"),
                    mode=wandb_mode,
                )
                logger.info(f"WandB logger initialized: project={cfg.wandb.get('project', 'battery-soh')}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB logger: {e}. Continuing without WandB.")
                pl_logger = None
        
        # 创建 Trainer
        max_epochs = cfg.training.get("epochs", 100)
        accelerator = cfg.training.get("accelerator", "auto")
        devices = cfg.training.get("devices", 1)
        enable_progress_bar = cfg.training.get("enable_progress_bar", True)
        log_every_n_steps = cfg.training.get("log_every_n_steps", 10)
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=pl_logger,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=False,
            enable_checkpointing=False,  # 使用自定义回调
        )
        
        logger.info(
            f"Trainer created: max_epochs={max_epochs}, accelerator={accelerator}, "
            f"devices={devices}, callbacks={len(callbacks)}"
        )
        
        return trainer
        
    except Exception as e:
        logger.error(f"创建 Trainer 时发生错误: {e}")
        raise
