"""
Model training using PyTorch Lightning.

Provides a clean interface for training SOH prediction models
with early stopping and learning rate scheduling.
"""

from dataclasses import dataclass
from typing import Any

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from battery_soh.core.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience (epochs)
        weight_decay: L2 regularization coefficient
        device: Computation device ('auto', 'cpu', 'cuda')
    """
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    patience: int = DEFAULT_PATIENCE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    device: str = "auto"


@dataclass(frozen=True)
class TrainingResult:
    """Result of model training.
    
    Attributes:
        best_val_loss: Best validation loss achieved
        best_epoch: Epoch number with best validation loss
        total_epochs: Total epochs trained
        history: Dictionary of metric histories
    """
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    history: dict[str, list[float]]


class SOHLightningModule(pl.LightningModule):
    """Lightning module for SOH prediction.
    
    Wraps a PyTorch model with training logic.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class LightningTrainer:
    """Trainer for SOH prediction models.
    
    Wraps PyTorch Lightning with simplified interface.
    
    Example:
        >>> config = TrainingConfig(epochs=200, patience=30)
        >>> trainer = LightningTrainer(config)
        >>> result = trainer.fit(model, train_loader, val_loader)
        >>> print(f"Best val loss: {result.best_val_loss:.4f}")
    """
    
    def __init__(self, config: TrainingConfig | None = None):
        """Initialize trainer.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self._trainer: pl.Trainer | None = None
    
    def fit(
        self,
        model: torch.nn.Module,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> TrainingResult:
        """Train a model.
        
        Args:
            model: PyTorch model to train
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val), optional
            
        Returns:
            TrainingResult with metrics
        """
        X_train, y_train = train_data
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train) if isinstance(X_train, torch.Tensor) == False else X_train,
            torch.from_numpy(y_train) if isinstance(y_train, torch.Tensor) == False else y_train
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = TensorDataset(
                torch.from_numpy(X_val) if isinstance(X_val, torch.Tensor) == False else X_val,
                torch.from_numpy(y_val) if isinstance(y_val, torch.Tensor) == False else y_val
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # Create Lightning module
        lightning_module = SOHLightningModule(
            model=model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create trainer with early stopping
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor='val_loss' if val_loader else 'train_loss',
                patience=self.config.patience,
                mode='min'
            )
        ]
        
        self._trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=self.config.device,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,  # Disable default logger for simplicity
            enable_checkpointing=False  # Disable checkpoint to avoid conflicts
        )
        
        # Train
        self._trainer.fit(lightning_module, train_loader, val_loader)
        
        # Extract results
        best_val_loss = float(self._trainer.callback_metrics.get('val_loss', 0))
        
        return TrainingResult(
            best_val_loss=best_val_loss,
            best_epoch=self._trainer.current_epoch,
            total_epochs=self._trainer.current_epoch,
            history={}
        )
