"""
Battery-wise data splitting strategies.

This module implements battery-wise train/validation/test splitting
to prevent data leakage. All cycles from the same battery must be
in the same split.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from battery_soh.core.types import Seed, Features, Labels, BatteryIds
from battery_soh.core.constants import TEST_SIZE, VALIDATION_SIZE


class BatteryWiseSplit:
    """Battery-wise data splitter.
    
    Splits data at the battery level to prevent data leakage.
    All cycles from the same battery go into the same split.
    
    Args:
        test_size: Proportion of batteries for test set
        val_size: Proportion of training batteries for validation
        seed: Random seed for reproducibility
        
    Example:
        >>> splitter = BatteryWiseSplit(seed=42)
        >>> train, val, test = splitter.split(X, y, battery_ids)
        >>> print(f"Train batteries: {len(np.unique(train.battery_ids))}")
        
    Note:
        This is the key design pattern to prevent data leakage in battery
        prognostics. Never split at the cycle level.
    """
    
    def __init__(
        self,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE,
        seed: Seed = Seed(42)
    ) -> None:
        """Initialize splitter.
        
        Args:
            test_size: Fraction of batteries for test (default: 0.25)
            val_size: Fraction of train batteries for validation (default: 0.25)
            seed: Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
    
    def split(
        self,
        X: Features,
        y: Labels,
        battery_ids: BatteryIds
    ) -> tuple["DataSplit", "DataSplit", "DataSplit"]:
        """Split data into train/validation/test sets.
        
        Args:
            X: Feature matrix [N_SAMPLES, N_FEATURES]
            y: Labels [N_SAMPLES]
            battery_ids: Battery identifiers [N_SAMPLES]
            
        Returns:
            Tuple of (train_split, val_split, test_split)
            
        Raises:
            ValueError: If not enough batteries for splitting
        """
        # Get unique batteries
        unique_batteries = np.unique(battery_ids)
        n_batteries = len(unique_batteries)
        
        if n_batteries < 3:
            raise ValueError(
                f"Need at least 3 batteries for train/val/test split, "
                f"got {n_batteries}"
            )
        
        # Split batteries (not individual samples!)
        train_val_batteries, test_batteries = train_test_split(
            unique_batteries,
            test_size=self.test_size,
            random_state=int(self.seed)
        )
        
        train_batteries, val_batteries = train_test_split(
            train_val_batteries,
            test_size=self.val_size / (1 - self.test_size),
            random_state=int(self.seed)
        )
        
        # Create masks for each split
        train_mask = np.isin(battery_ids, train_batteries)
        val_mask = np.isin(battery_ids, val_batteries)
        test_mask = np.isin(battery_ids, test_batteries)
        
        # Create splits
        train_split = DataSplit(
            X=X[train_mask],
            y=y[train_mask],
            battery_ids=battery_ids[train_mask]
        )
        
        val_split = DataSplit(
            X=X[val_mask],
            y=y[val_mask],
            battery_ids=battery_ids[val_mask]
        )
        
        test_split = DataSplit(
            X=X[test_mask],
            y=y[test_mask],
            battery_ids=battery_ids[test_mask]
        )
        
        return train_split, val_split, test_split


from dataclasses import dataclass

@dataclass(frozen=True)
class DataSplit:
    """Container for a data split.
    
    Attributes:
        X: Feature matrix
        y: Labels
        battery_ids: Battery identifiers for each sample
    """
    X: Features
    y: Labels
    battery_ids: BatteryIds
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.X)
    
    @property
    def n_batteries(self) -> int:
        """Return number of unique batteries in this split."""
        return len(np.unique(self.battery_ids))
