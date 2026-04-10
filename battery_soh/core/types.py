"""Core type definitions."""

from typing import Literal
import numpy as np
from numpy.typing import NDArray

# Basic type aliases
Seed = int
MissingRate = float

# Literal types for valid values
BatchId = Literal["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
ModelType = Literal["mlp", "lstm", "cnn"]
MissingMode = Literal["MCAR", "MAR", "MNAR"]
ImputationMethod = Literal["mean", "knn", "iterative", "zero"]

# Array types
Features = NDArray[np.float32]
Labels = NDArray[np.float32]
Mask = NDArray[np.bool_]
BatteryIds = NDArray[np.str_]
