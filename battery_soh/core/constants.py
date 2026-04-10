"""
Core constants for the Battery SOH Framework.

All physical and configuration constants are defined here for single source of truth.
"""

from battery_soh.core.types import BatchId, MissingMode, ImputationMethod, ModelType

# =============================================================================
# Feature Dimensions
# =============================================================================

N_FEATURES: int = 16
"""Number of input features (voltage, current, charge statistics)."""

N_FEATURES_WITH_MIM: int = 32
"""Number of input features with MIM (16 features + 16 missing indicators)."""

# =============================================================================
# Valid Values
# =============================================================================

XJTU_BATCHES: tuple[BatchId, ...] = ("2C", "3C", "R2.5", "R3", "RW", "Sim_satellite")
"""All valid XJTU dataset batch identifiers."""

MISSING_MODES: tuple[MissingMode, ...] = ("MCAR", "MAR", "MNAR")
"""All supported missing data mechanisms."""

IMPUTATION_METHODS: tuple[ImputationMethod, ...] = ("mean", "knn", "iterative", "zero")
"""All supported imputation methods."""

MODEL_TYPES: tuple[ModelType, ...] = ("mlp", "lstm", "cnn")
"""All supported model architectures."""

# =============================================================================
# Parameter Budget (from meta.md)
# =============================================================================

MIN_PARAMS: int = 16_384  # 2^14
"""Minimum number of model parameters (inclusive)."""

MAX_PARAMS: int = 32_768  # 2^15
"""Maximum number of model parameters (inclusive)."""

# =============================================================================
# Training Defaults
# =============================================================================

DEFAULT_EPOCHS: int = 200
"""Default number of training epochs."""

DEFAULT_BATCH_SIZE: int = 64
"""Default training batch size."""

DEFAULT_LEARNING_RATE: float = 1e-3
"""Default learning rate for Adam optimizer."""

DEFAULT_PATIENCE: int = 30
"""Default early stopping patience (epochs)."""

DEFAULT_WEIGHT_DECAY: float = 1e-5
"""Default L2 regularization coefficient."""

# =============================================================================
# Data Splitting
# =============================================================================

TEST_SIZE: float = 0.25
"""Proportion of batteries for test set."""

VALIDATION_SIZE: float = 0.25
"""Proportion of training batteries for validation set."""

# =============================================================================
# Missing Data Configuration
# =============================================================================

TRAIN_MISSING_RATES: tuple[float, ...] = tuple(i * 0.05 for i in range(20))
"""Missing rates for MIM training: 0.0, 0.05, 0.10, ..., 0.95."""

TEST_MISSING_RATES: tuple[float, ...] = tuple(i * 0.05 for i in range(20))
"""Missing rates for testing: 0.0, 0.05, 0.10, ..., 0.95."""

# =============================================================================
# File Paths
# =============================================================================

DEFAULT_DATA_DIR: str = "data/XJTU data"
"""Default directory for XJTU dataset."""

DEFAULT_MODEL_DIR: str = "models"
"""Default directory for saving trained models."""

DEFAULT_RESULTS_DIR: str = "results"
"""Default directory for saving experiment results."""


# =============================================================================
# Random Seed Management
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
