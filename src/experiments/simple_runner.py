"""
简化版实验运行器

利用 Hydra + PyTorch Lightning，代码量最小化
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

from src.data.xjtu_loader import XJTUDataLoader
from src.utils.seed_manager import set_seed
from src.utils.hydra_utils import get_model_input_dim, get_missing_rates


def load_and_prepare_data(cfg: DictConfig) -> Tuple:
    """
    加载并准备数据
    
    Returns:
        (X_train, y_train, X_val, y_val, train_mean, train_std)
    """
    # 加载原始数据
    loader = XJTUDataLoader(batch_id=cfg.data.batch, data_dir=cfg.data.data_dir)
    df = loader.load_data()
    
    # 简单的电池分割
    np.random.seed(cfg.seed.seed)
    batteries = df['battery_id'].unique()
    np.random.shuffle(batteries)
    
    n_train = max(4, len(batteries) // 2)
    n_val = max(2, len(batteries) // 4)
    
    train_bats = batteries[:n_train]
    val_bats = batteries[n_train:n_train + n_val]
    
    # 提取特征
    feature_cols = [c for c in df.columns 
                   if c not in ['battery_id', 'cycle', 'capacity']]
    
    train_df = df[df['battery_id'].isin(train_bats)]
    val_df = df[df['battery_id'].isin(val_bats)]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['capacity'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['capacity'].values.astype(np.float32)
    
    # 标准化
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1.0
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    
    return X_train, y_train, X_val, y_val, train_mean, train_std


def prepare_mim_data(
    X: np.ndarray,
    y: np.ndarray,
    missing_rates: list,
    imputation: str,
    seed: int
) -> list:
    """准备 MIM 多 MR 训练数据"""
    from src.missing.missing_data_simple import (
        generate_missing, impute, build_mim_input
    )
    
    data = []
    for mr in missing_rates:
        if mr == 0.0:
            X_mr = X
            mask = np.zeros_like(X)
        else:
            mask = generate_missing(X, 'mcar', mr, seed + int(mr * 100))
            X_missing = X.copy()
            X_missing[mask] = np.nan
            X_mr = impute(X_missing, imputation, fit_data=X)
        
        X_mim = build_mim_input(X_mr, mask)
        data.append((X_mim, y))
    
    return data


def create_dataloaders(data: list, batch_size: int, shuffle: bool = True):
    """创建 DataLoader 列表"""
    loaders = []
    for X, y in data:
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    return loaders


def run_training(cfg: DictConfig) -> dict:
    """
    运行训练
    
    简化版：
    1. 加载数据
    2. 准备训练数据（MIM 或 Baseline）
    3. 创建 Lightning 模块
    4. 训练
    """
    # 设置种子
    set_seed(cfg.seed.seed)
    
    # 加载数据
    X_train, y_train, X_val, y_val, train_mean, train_std = load_and_prepare_data(cfg)
    
    # 确定输入维度
    input_dim = get_model_input_dim(cfg)
    
    # 准备训练数据
    if cfg.mim.use_mim:
        missing_rates = get_missing_rates(cfg)
        train_data = prepare_mim_data(
            X_train, y_train, missing_rates, 
            cfg.mim.train_imputation, cfg.seed.seed
        )
    else:
        # Baseline: 完整数据
        X_tensor = torch.FloatTensor(X_train)
        if cfg.model.model_type in ['lstm', 'cnn']:
            X_tensor = X_tensor.unsqueeze(1)
        train_data = [(X_tensor, y_train)]
    
    # 创建验证数据
    X_val_tensor = torch.FloatTensor(X_val)
    if cfg.model.model_type in ['lstm', 'cnn']:
        X_val_tensor = X_val_tensor.unsqueeze(1)
    val_dataset = TensorDataset(X_val_tensor, torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    
    # 创建模型
    from src.models.factory_simple import create_model
    model = create_model(
        cfg.model.model_type,
        input_dim,
        **{k: v for k, v in cfg.model.items() if k != 'model_type'}
    )
    
    # 创建 Lightning 模块
    from src.trainers.lightning_module import SOHLightningModule
    pl_module = SOHLightningModule(
        model=model,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    
    # 回调
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg.training.patience, mode='min'),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.paths.model_dir,
            filename=f"seed{cfg.seed.seed}_{cfg.data.batch}_{cfg.model.model_type}",
            save_top_k=1,
            mode='min'
        )
    ]
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        enable_progress_bar=cfg.get('verbose', True),
        logger=False,  # 简化版不使用 logger
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    # 训练
    if cfg.mim.use_mim:
        # MIM: 使用多个 DataLoader
        train_loaders = create_dataloaders(
            train_data, cfg.training.batch_size, shuffle=True
        )
        # 轮流从每个 loader 取数据
        combined_loader = CombinedLoader(train_loaders, mode='max_size_cycle')
        trainer.fit(pl_module, combined_loader, val_loader)
    else:
        # Baseline: 单一 DataLoader
        train_dataset = TensorDataset(*train_data[0])
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True
        )
        trainer.fit(pl_module, train_loader, val_loader)
    
    return {
        'best_val_loss': trainer.callbacks[0].best_score.item(),
        'model_path': trainer.callbacks[1].best_model_path
    }


class CombinedLoader:
    """简单的多 DataLoader 组合器"""
    
    def __init__(self, loaders, mode='max_size_cycle'):
        self.loaders = loaders
        self.mode = mode
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.loaders]
        while True:
            batches = []
            for it in iterators:
                try:
                    batches.append(next(it))
                except StopIteration:
                    if self.mode == 'max_size_cycle':
                        iterators = [iter(loader) for loader in self.loaders]
                        batches.append(next(iterators[0]))
                    else:
                        return
            
            # 合并批次
            X = torch.cat([b[0] for b in batches])
            y = torch.cat([b[1] for b in batches])
            yield X, y
    
    def __len__(self):
        return max(len(loader) for loader in self.loaders)


__all__ = ['run_training']
