"""Core module - Types, interfaces, and constants."""

from battery_soh.core.types import (
    Seed,
    MissingRate,
    BatchId,
    ModelType,
    MissingMode,
    ImputationMethod,
    Features,
    Labels,
    Mask,
    BatteryIds,
)
from battery_soh.core.interfaces import (
    DataLoader,
    Model,
    Trainer,
    MissingGenerator,
    Imputer,
)
from battery_soh.core.constants import (
    N_FEATURES,
    N_FEATURES_WITH_MIM,
    XJTU_BATCHES,
    MISSING_MODES,
    IMPUTATION_METHODS,
    MODEL_TYPES,
    set_seed,
)

__all__ = [
    # Types
    "Seed",
    "MissingRate",
    "BatchId",
    "ModelType",
    "MissingMode",
    "ImputationMethod",
    "Features",
    "Labels",
    "Mask",
    "BatteryIds",
    # Interfaces
    "DataLoader",
    "Model",
    "Trainer",
    "MissingGenerator",
    "Imputer",
    # Constants
    "N_FEATURES",
    "N_FEATURES_WITH_MIM",
    "XJTU_BATCHES",
    "MISSING_MODES",
    "IMPUTATION_METHODS",
    "MODEL_TYPES",
    # Functions
    "set_seed",
]
