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
    """根据 input_dim 获取模型配置."""
    model_type = cfg.model.type
    
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print("=" * 60)
    
    # 确定方法
    method = cfg.get('method', 'mim')  # 'mim', 'mean', 'median', 'knn', 'zero'
    is_mim = (method == 'mim')
    input_dim = 32 if is_mim else 16
    
    print(f"\nMethod: {method.upper()}")
    print(f"Input dimension: {input_dim}")
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # 加载数据
    print("\n[1/4] Loading dataset...")
    data_dict = load_dataset(cfg)
    
    results_all = []
    
    for seed in cfg.experiment.seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        # 创建模型
        print("[2/4] Creating model...")
        model_kwargs = get_model_config(cfg, input_dim)
        
        module = BatterySOHModule(
            model_type=cfg.model.type,
            input_dim=input_dim,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            **model_kwargs
        )
        
        total_params = sum(p.numel() for p in module.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # 日志
        wandb_logger = None
        if cfg.wandb.enabled:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                tags=[f"seed_{seed}", cfg.model.type, method],
                name=f"{cfg.experiment.name}_{cfg.model.type}_{method}_seed{seed}"
            )
        
        # 回调
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=cfg.training.patience, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
        ]
        
        # 训练器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
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
        for mr in cfg.missing.missing_rates_eval:
            _, _, test_loader = create_dataloaders(data_dict, cfg, mode='eval', method=method, missing_rate=mr)
            results = trainer.test(module, test_loader, verbose=False)
            
            results_all.append({
                'seed': seed,
                'missing_rate': mr,
                'model': cfg.model.type,
                'method': method,
                **results[0]
            })
            print(f"  MR={mr:.1f}: MAE={results[0]['test_mae']:.4f}")
    
    # 保存结果
    import pandas as pd
    results_df = pd.DataFrame(results_all)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{cfg.experiment.name}_{cfg.model.type}_{method}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
