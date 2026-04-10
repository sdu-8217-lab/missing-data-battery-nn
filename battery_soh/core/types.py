"""
Core type definitions for the Battery SOH Framework.

This module defines type aliases and constants used throughout the framework.
All types are self-documenting and use Python's typing system for clarity.

Design Principles
-----------------
1. **Explicit over implicit**: All types are explicitly defined
2. **Self-documenting**: Type names clearly indicate their purpose
3. **Consistency**: Type aliases ensure consistency across the codebase
"""

from typing import NewType, Literal, TypeVar
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Identifier Types (NewType for type safety)
# =============================================================================

Seed = NewType("Seed", int)
"""Random seed for reproducibility.

Controls all random processes:
- Data splitting
- Model initialization  
- Missing pattern generation
- Training shuffling
"""

MissingRate = NewType("MissingRate", float)
"""Missing data rate, must be in [0.0, 1.0].

Examples:
    - 0.0: No missing data
    - 0.3: 30% missing data
    - 0.95: 95% missing data (high missingness scenario)
"""

# =============================================================================
# Enumeration Types (Literal for static checking)
# =============================================================================

BatchId = Literal["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
"""XJTU dataset batch identifiers.

Each batch represents different battery charging conditions:
    - "2C": 2C charging rate
    - "3C": 3C charging rate
    - "R2.5": Random walk with 2.5A amplitude
    - "R3": Random walk with 3A amplitude
    - "RW": Random walk profile
    - "Sim_satellite": Simulated satellite profile
"""

ModelType = Literal["mlp", "lstm", "cnn"]
"""Supported neural network architectures.

- "mlp": Multi-Layer Perceptron (feedforward)
- "lstm": Long Short-Term Memory (sequential)
- "cnn": Convolutional Neural Network (local patterns)
"""

MissingMode = Literal["MCAR", "MAR", "MNAR"]
"""Missing data mechanisms (Rubin's classification).

- "MCAR": Missing Completely At Random
    Missingness independent of both observed and unobserved data
    
- "MAR": Missing At Random  
    Missingness depends on observed data (e.g., SOH value)
    
- "MNAR": Missing Not At Random
    Missingness depends on unobserved data (e.g., missing values themselves)

Reference:
    Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
"""

ImputationMethod = Literal["mean", "knn", "iterative", "zero"]
"""Methods for filling missing values.

- "mean": Fill with feature mean (univariate)
- "knn": K-Nearest Neighbors imputation (multivariate)
- "iterative": Iterative regression imputation (MICE-style)
- "zero": Fill with zeros (simple baseline)
"""

# =============================================================================
# Array Types (NDArray for numerical operations)
# =============================================================================

Features = NDArray[np.float32]
"""Feature matrix with shape [N_SAMPLES, N_FEATURES].

N_FEATURES = 16 (voltage, current, charge features, etc.)
Data type is float32 for GPU efficiency.
"""

Labels = NDArray[np.float32]
"""Target labels (SOH values) with shape [N_SAMPLES].

SOH (State of Health) is typically in range [0.0, 1.0].
Calculated as: current_capacity / nominal_capacity
"""

Mask = NDArray[np.bool_]
"""Boolean mask indicating observed values.

- True: Value is observed (present)
- False: Value is missing

Shape matches the data being masked, typically [N_SAMPLES, N_FEATURES].
"""

BatteryIds = NDArray[np.str_]
"""Battery identifier strings with shape [N_SAMPLES].

Used for battery-wise splitting to prevent data leakage.
Example: ["2C_battery-1", "2C_battery-1", "2C_battery-2", ...]
"""

# =============================================================================
# Generic Types
# =============================================================================

T = TypeVar("T")
"""Generic type variable for flexible typing."""

# =============================================================================
# Type Validation Helpers
# =============================================================================

def validate_missing_rate(rate: float) -> MissingRate:
    """Validate and convert to MissingRate type.
    
    Args:
        rate: Missing rate value
        
    Returns:
        Validated MissingRate
        
    Raises:
        ValueError: If rate is outside [0.0, 1.0]
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Missing rate must be in [0.0, 1.0], got {rate}")
    return MissingRate(rate)


def validate_seed(seed: int) -> Seed:
    """Validate and convert to Seed type.
    
    Args:
        seed: Random seed value
        
    Returns:
        Validated Seed
        
    Raises:
        ValueError: If seed is negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    return Seed(seed)
