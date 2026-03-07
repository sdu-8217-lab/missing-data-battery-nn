"""Main experiment runner - 支持 MIM 和插补 Baseline."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.loader import load_dataset, create_dataloaders
from src.models.lightning_module import BatterySOHModule
from src.utils.seed_manager import set_seed


def get_model_config(cfg: DictConfig, input_dim: int) -> dict:
    """
    获取模型配置。
    
    优先使用 cfg.model 中的配置，如果没有则使用默认配置。
    支持 Paper 配置和向后兼容。
    """
    model_cfg = cfg.get('model', {})
    model_type = model_cfg.get('name') or model_cfg.get('type')
    
    # 如果配置中已包含完整模型参数，直接使用
    if 'hidden_dims' in model_cfg or 'hidden_size' in model_cfg or 'channels' in model_cfg:
        # 从配置中提取模型参数
        kwargs = {}
        
        # MLP 参数
        if 'hidden_dims' in model_cfg:
            kwargs['hidden_dims'] = model_cfg.hidden_dims
        if 'dropout' in model_cfg:
            kwargs['dropout'] = model_cfg.dropout
            
        # LSTM/GRU 参数
        if 'hidden_size' in model_cfg:
            kwargs['hidden_size'] = model_cfg.hidden_size
        if 'num_layers' in model_cfg:
            kwargs['num_layers'] = model_cfg.num_layers
            
        # CNN1D 参数
        if 'channels' in model_cfg:
            kwargs['channels'] = model_cfg.channels
        if 'kernel_size' in model_cfg:
            kwargs['kernel_size'] = model_cfg.kernel_size
            
        return kwargs
    
    # 否则使用默认配置（向后兼容）
    print(f"  Warning: Using default model config for {model_type}. "
          f"Consider adding model parameters to config file.")
    
    if model_type == 'mlp':
        if input_dim == 32:  # MIM
            return {'hidden_dims': [84, 56, 28], 'dropout': 0.15}
        else:  # Baseline (16)
            return {'hidden_dims': [100, 64, 32], 'dropout': 0.15}
    
    elif model_type == 'lstm':
        return {'hidden_size': 48, 'num_layers': 2, 'dropout': 0.2}
    
    elif model_type == 'gru':
        return {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}
    
    elif model_type == 'cnn1d':
        if input_dim == 32:
            return {'channels': [64, 32], 'kernel_size': 4, 'dropout': 0.1}
        else:
            return {'channels': [72, 32], 'kernel_size': 4, 'dropout': 0.1}
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def detect_method(cfg: DictConfig) -> tuple[str, bool]:
    """
    检测实验方法类型。
    
    Returns:
        (method_name, is_mim)
    """
    # 支持新的配置结构
    method_cfg = cfg.get('method', {})
    
    # OmegaConf DictConfig 不是标准 dict，需要特殊处理
    if isinstance(method_cfg, (dict, DictConfig)) and not isinstance(method_cfg, str):
        method_name = method_cfg.get('name', 'mim')
        is_mim = method_cfg.get('use_missing_indicator', method_name == 'mim')
    else:
        # 向后兼容：method 是字符串
        method_name = str(method_cfg)
        is_mim = (method_name == 'mim')
    
    return method_name, is_mim


def get_missing_rates(cfg: DictConfig) -> list:
    """
    获取缺失率列表。
    
    支持多种配置结构。
    """
    # 优先检查顶层 missing_rates
    if 'missing_rates' in cfg:
        return list(cfg.missing_rates)
    
    # 检查 experiment.missing_rates
    if 'experiment' in cfg and 'missing_rates' in cfg.experiment:
        return list(cfg.experiment.missing_rates)
    
    # 向后兼容：missing.missing_rates_eval
    if 'missing' in cfg and 'missing_rates_eval' in cfg.missing:
        return list(cfg.missing.missing_rates_eval)
    
    # 默认
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def get_seeds(cfg: DictConfig) -> list:
    """
    获取随机种子列表。
    """
    if 'experiment' in cfg and 'seeds' in cfg.experiment:
        return list(cfg.experiment.seeds)
    
    if 'seeds' in cfg:
        return list(cfg.seeds)
    
    return [42]


def get_training_config(cfg: DictConfig) -> dict:
    """
    获取训练配置。
    """
    training = cfg.get('training', {})
    
    return {
        'epochs': training.get('max_epochs', training.get('epochs', 100)),
        'batch_size': training.get('batch_size', 64),
        'learning_rate': training.get('learning_rate', 0.001),
        'weight_decay': training.get('weight_decay', 0.0),
        'patience': training.get('early_stopping_patience', training.get('patience', 15)),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print("=" * 60)
    
    # 检测方法
    method, is_mim = detect_method(cfg)
    input_dim = 32 if is_mim else 16
    
    print(f"\nMethod: {method.upper()}")
    print(f"Input dimension: {input_dim}")
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # 获取训练配置
    train_cfg = get_training_config(cfg)
    seeds = get_seeds(cfg)
    missing_rates = get_missing_rates(cfg)
    
    print(f"\nTraining config: {train_cfg}")
    print(f"Seeds: {len(seeds)} seeds")
    print(f"Missing rates: {missing_rates}")
    
    # 加载数据
    print("\n[1/4] Loading dataset...")
    data_dict = load_dataset(cfg)
    
    results_all = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        # 创建模型
        print("[2/4] Creating model...")
        model_kwargs = get_model_config(cfg, input_dim)
        
        model_type = cfg.model.get('name') or cfg.model.get('type')
        module = BatterySOHModule(
            model_type=model_type,
            input_dim=input_dim,
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
            **model_kwargs
        )
        
        total_params = sum(p.numel() for p in module.parameters())
        print(f"  Model: {model_type}")
        print(f"  Input dim: {input_dim}")
        print(f"  Model kwargs: {model_kwargs}")
        print(f"  Total parameters: {total_params:,}")
        
        # 日志
        wandb_logger = None
        wandb_cfg = cfg.get('wandb', {})
        if wandb_cfg.get('enabled', False):
            try:
                wandb_logger = WandbLogger(
                    project=wandb_cfg.get('project', 'battery-soh'),
                    entity=wandb_cfg.get('entity'),
                    tags=[f"seed_{seed}", model_type, method],
                    name=f"{cfg.experiment.name}_{model_type}_{method}_seed{seed}"
                )
                # 强制触发初始化以检测错误
                _ = wandb_logger.experiment
                print(f"  WandB logging enabled: {wandb_logger.experiment.url}")
            except Exception as e:
                print(f"  Warning: WandB initialization failed ({e}). Continuing without WandB.")
                print(f"  To use WandB: 1) Run 'wandb login' or 2) Set wandb.enabled=false in config")
                wandb_logger = None
        
        # 回调
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=train_cfg['patience'], mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
        ]
        
        # 训练器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(
            max_epochs=train_cfg['epochs'],
            accelerator=device,
            callbacks=callbacks,
            logger=wandb_logger,
            enable_progress_bar=True
        )
        
        # 训练
        print("[3/4] Training...")
        train_loader, val_loader = create_dataloaders(data_dict, cfg, mode='train', method=method)
        trainer.fit(module, train_loader, val_loader)
        
        # 评估
        print("[4/4] Evaluating across missing rates...")
        for mr in missing_rates:
            _, _, test_loader = create_dataloaders(data_dict, cfg, mode='eval', method=method, missing_rate=mr)
            results = trainer.test(module, test_loader, verbose=False)
            
            results_all.append({
                'seed': seed,
                'missing_rate': mr,
                'model': model_type,
                'method': method,
                **results[0]
            })
            print(f"  MR={mr:.1f}: MAE={results[0]['test_mae']:.4f}")
    
    # 保存结果
    import pandas as pd
    results_df = pd.DataFrame(results_all)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{cfg.experiment.name}_{model_type}_{method}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
