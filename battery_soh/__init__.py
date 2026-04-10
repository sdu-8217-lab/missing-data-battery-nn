"""
Battery SOH Prediction Framework

A long-term open-source framework for battery state-of-health prediction
under missing data scenarios.

Architecture
------------
The framework follows a 9-level experimental architecture (see docs/ARCHITECTURE.md):

- Level 1-6 (Above the Divide): Training phase configurations
- Level 7-9 (Below the Divide): Testing phase configurations

Key Concepts
------------
- **MIM (Missing Indicator Method)**: Augments input with missingness mask
- **Imputation**: Fills missing values before feeding to model
- **Battery-wise Split**: Prevents data leakage by splitting at battery level

Example
-------
>>> from battery_soh.data import XJTULoader
>>> from battery_soh.models import create_model
>>> 
>>> # Load data
>>> loader = XJTULoader()
>>> X, y, battery_ids = loader.load_batch("2C")
>>> 
>>> # Create model (without MIM)
>>> model = create_model("mlp", use_mim=False)

License
-------
MIT License - See LICENSE file for details.
"""

__version__ = "0.5.0-alpha.1"
__author__ = "Battery SOH Research Team"

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
)

__all__ = [
    "Seed",
    "MissingRate", 
    "BatchId",
    "ModelType",
    "MissingMode",
    "ImputationMethod",
    "Features",
    "Labels",
    "Mask",
]
