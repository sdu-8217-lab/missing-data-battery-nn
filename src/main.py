"""Main experiment runner using Hydra + PyTorch Lightning."""
import sys
from pathlib import Path

# Add project root to path
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


def get_model_config(cfg: DictConfig) -> dict:
    """
    根据 use_mim 获取模型配置
    保持模型参数量大致一致（~10K）
    """
    model_type = cfg.model.type
    use_mim = cfg.model.use_mim
    
    if model_type == 'mlp':
        # No MIM (16 -> 100): ~10K params
        # With MIM (32 -> 84): ~9K params (close enough)
        if use_mim:
            return {
                'hidden_dims': cfg.model.get('mim_hidden_dims', [84, 56, 28]),
                'dropout': cfg.model.get('dropout', 0.15)
            }
        else:
            return {
                'hidden_dims': cfg.model.get('hidden_dims', [100, 64, 32]),
                'dropout': cfg.model.get('dropout', 0.15)
            }
    
    elif model_type in ['lstm', 'gru']:
        # LSTM/GRU 参数主要来自隐藏层，输入维度影响较小
        # 保持隐藏层配置一致
        return {
            'hidden_size': cfg.model.get('hidden_size', 48 if model_type == 'lstm' else 64),
            'num_layers': cfg.model.get('num_layers', 2),
            'dropout': cfg.model.get('dropout', 0.2)
        }
    
    elif model_type == 'cnn1d':
        # CNN 参数主要来自卷积核，输入维度影响 channels[0]
        # No MIM: [72, 32], With MIM: [64, 32] to balance params
        if use_mim:
            return {
                'channels': cfg.model.get('mim_channels', [64, 32]),
                'kernel_size': cfg.model.get('kernel_size', 4),
                'dropout': cfg.model.get('dropout', 0.1)
            }
        else:
            return {
                'channels': cfg.model.get('channels', [72, 32]),
                'kernel_size': cfg.model.get('kernel_size', 4),
                'dropout': cfg.model.get('dropout', 0.1)
            }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print("=" * 60)
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # Determine input dimension (16 features + 16 mask = 32)
    input_dim = 32 if cfg.model.use_mim else 16
    
    # Load data once
    print("\n[1/4] Loading dataset...")
    data_dict = load_dataset(cfg)
    
    results_all = []
    
    for seed in cfg.experiment.seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        # Create model with appropriate config
        print("[2/4] Creating model...")
        model_kwargs = get_model_config(cfg)
        print(f"  Model config: {model_kwargs}")
        
        module = BatterySOHModule(
            model_type=cfg.model.type,
            input_dim=input_dim,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            **model_kwargs
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in module.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Setup logger
        wandb_logger = None
        if cfg.wandb.enabled:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                tags=cfg.wandb.tags + [f"seed_{seed}", cfg.model.type, f"mim_{cfg.model.use_mim}"],
                name=f"{cfg.experiment.name}_{cfg.model.type}_mim{cfg.model.use_mim}_seed{seed}"
            )
        
        # Setup callbacks with adjusted patience
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=cfg.training.patience,
                mode='min'
            ),
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename=f'seed{seed}' + '-{epoch:02d}-{val_loss:.4f}'
            )
        ]
        
        # Setup trainer
        device = cfg.training.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            accelerator=device,
            callbacks=callbacks,
            logger=wandb_logger,
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        
        # Train
        print("[3/4] Training...")
        
        # MIM 训练：混合多种缺失率
        if cfg.model.use_mim:
            mim_rates = cfg.missing.get('mim_train_rates', [i/10.0 for i in range(10)])
            print(f"  MIM training with missing rates: {mim_rates}")
            train_loader, val_loader = create_dataloaders(
                data_dict, cfg, mode='train', 
                use_mim=True, mim_train_rates=mim_rates
            )
        else:
            # Baseline：使用训练时指定的单一缺失率
            train_mr = cfg.missing.get('missing_rate_train', 0.0)
            print(f"  Baseline training with missing rate: {train_mr}")
            train_loader, val_loader = create_dataloaders(
                data_dict, cfg, mode='train', 
                missing_rate=train_mr, use_mim=False
            )
        
        trainer.fit(module, train_loader, val_loader)
        
        # Evaluate across missing rates
        print("[4/4] Evaluating across missing rates...")
        for mr in cfg.missing.missing_rates_eval:
            _, _, test_loader = create_dataloaders(
                data_dict, cfg, mode='eval', missing_rate=mr
            )
            results = trainer.test(module, test_loader, verbose=False)
            
            result_row = {
                'seed': seed,
                'missing_rate': mr,
                'model': cfg.model.type,
                'use_mim': cfg.model.use_mim,
                **results[0]
            }
            results_all.append(result_row)
            print(f"  MR={mr:.1f}: MAE={results[0]['test_mae']:.4f}")
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results_all)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mim_tag = "mim" if cfg.model.use_mim else "baseline"
    results_file = output_dir / f"{cfg.experiment.name}_{cfg.model.type}_{mim_tag}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
