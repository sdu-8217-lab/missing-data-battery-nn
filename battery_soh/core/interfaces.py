"""
Core interfaces for the Battery SOH Framework.

All interfaces are defined as Protocol classes for structural subtyping.
This allows any class that implements the required methods to be used,
without explicit inheritance.

Design Principles
-----------------
1. **Minimal**: Only essential methods are required
2. **Clear**: Method signatures are fully type-annotated
3. **Composable**: Small interfaces that can be combined

Example
-------
>>> class MyModel(Model):
...     def __call__(self, x: torch.Tensor) -> torch.Tensor:
...         return self.network(x)
...
>>> # MyModel can be used anywhere Model is expected
"""

from typing import Protocol, runtime_checkable
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from battery_soh.core.types import (
    Seed,
    MissingRate,
    Features,
    Labels,
    Mask,
    BatchId,
)


# =============================================================================
# Data Layer Interfaces
# =============================================================================

@runtime_checkable
class DataLoader(Protocol):
    """Protocol for battery dataset loaders.
    
    All dataset loaders must implement this interface to be used
    interchangeably in the framework.
    """
    
    def load_batch(
        self, 
        batch_id: BatchId
    ) -> tuple[Features, Labels, Features]:
        """Load a specific batch of battery data.
        
        Args:
            batch_id: Batch identifier (e.g., "2C", "3C")
            
        Returns:
            Tuple of (features, labels, battery_ids):
                - features: Array of shape [N_SAMPLES, N_FEATURES]
                - labels: Array of shape [N_SAMPLES] containing SOH values
                - battery_ids: Array of shape [N_SAMPLES] containing battery identifiers
                
        Raises:
            FileNotFoundError: If batch data cannot be found
            ValueError: If batch_id is invalid
        """
        ...
    
    def list_batches(self) -> list[BatchId]:
        """Return list of available batch identifiers.
        
        Returns:
            List of valid batch IDs that can be loaded
        """
        ...


# =============================================================================
# Model Layer Interfaces  
# =============================================================================

@runtime_checkable
class Model(Protocol):
    """Protocol for SOH prediction models.
    
    Any PyTorch module that implements forward pass can be used.
    The framework expects models to output single values per sample.
    """
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [BATCH_SIZE, N_FEATURES]
               or [BATCH_SIZE, SEQ_LEN, N_FEATURES] for sequential models
               
        Returns:
            Predictions of shape [BATCH_SIZE]
        """
        ...
    
    def parameters(self):
        """Return model parameters for optimization."""
        ...
    
    def to(self, device: str | torch.device):
        """Move model to device."""
        ...
    
    def eval(self):
        """Set model to evaluation mode."""
        ...
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        ...


# =============================================================================
# Training Layer Interfaces
# =============================================================================

@runtime_checkable  
class Trainer(Protocol):
    """Protocol for model trainers.
    
    Trainers encapsulate the training loop and optimization logic.
    """
    
    def fit(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None
    ) -> "TrainingResult":
        """Train a model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            TrainingResult containing metrics and history
        """
        ...


# =============================================================================
# Missing Data Layer Interfaces
# =============================================================================

@runtime_checkable
class MissingGenerator(Protocol):
    """Protocol for missing pattern generators.
    
    Implements different missing data mechanisms (MCAR, MAR, MNAR).
    """
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed,
        **kwargs
    ) -> tuple[Features, Mask]:
        """Generate missing pattern.
        
        Args:
            X: Original feature matrix
            missing_rate: Target proportion of missing values
            seed: Random seed for reproducibility
            **kwargs: Additional parameters specific to the mechanism
            
        Returns:
            Tuple of (X_with_missing, mask):
                - X_with_missing: Feature matrix with NaN for missing values
                - mask: Boolean mask, True = observed, False = missing
        """
        ...
    
    @property
    def name(self) -> str:
        """Return the name of the missing mechanism."""
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for missing value imputation methods.
    
    Imputers follow the scikit-learn fit/transform pattern.
    """
    
    def fit(self, X: Features) -> "Imputer":
        """Fit imputer on complete data.
        
        Args:
            X: Complete feature matrix (no missing values)
            
        Returns:
            Self for method chaining
        """
        ...
    
    def transform(self, X_missing: Features) -> Features:
        """Impute missing values.
        
        Args:
            X_missing: Feature matrix with NaN for missing values
            
        Returns:
            Feature matrix with imputed values
        """
        ...
    
    def fit_transform(self, X: Features, mask: Mask | None = None) -> Features:
        """Fit and transform in one step.
        
        Args:
            X: Feature matrix (with or without NaN)
            mask: Optional boolean mask indicating observed values
            
        Returns:
            Feature matrix with imputed values
        """
        ...
    
    @property
    def name(self) -> str:
        """Return the name of the imputation method."""
        ...


# =============================================================================
# Result Types
# =============================================================================

from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingResult:
    """Result of model training.
    
    Attributes:
        best_val_loss: Best validation loss achieved
        best_epoch: Epoch number with best validation loss
        total_epochs: Total number of epochs trained
        history: Dictionary mapping metric names to lists of values
    """
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    history: dict[str, list[float]]


@dataclass(frozen=True)
class EvaluationResult:
    """Result of model evaluation.
    
    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R-squared coefficient
        mape: Mean Absolute Percentage Error (optional)
    """
    mae: float
    rmse: float
    r2: float
    mape: float | None = None
