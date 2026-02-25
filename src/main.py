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
from src.missing_data.mcar import simulate_mcar
from src.missing_data.mar import simulate_mar
from src.models.lightning_module import BatterySOHModule
from src.utils.seed_manager import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print("=" * 60)
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # Determine input dimension
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
        
        # Create model
        print("[2/4] Creating model...")
        model_kwargs = {
            'hidden_dims': cfg.model.get('hidden_dims', [100, 64, 32]),
            'dropout': cfg.model.get('dropout', 0.15),
        }
        
        module = BatterySOHModule(
            model_type=cfg.model.type,
            input_dim=input_dim,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            **model_kwargs
        )
        
        # Setup logger
        wandb_logger = None
        if cfg.wandb.enabled:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                tags=cfg.wandb.tags + [f"seed_{seed}", cfg.model.type],
                name=f"{cfg.experiment.name}_{cfg.model.type}_seed{seed}"
            )
        
        # Setup callbacks
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
        train_loader, val_loader = create_dataloaders(
            data_dict, cfg, mode='train', missing_rate=cfg.missing.missing_rate_train
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
    results_file = output_dir / f"{cfg.experiment.name}_{cfg.model.type}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
