"""
XJTU dataset loader implementation.

This module provides efficient loading of the XJTU battery dataset,
with support for all 6 battery batches.
"""

from pathlib import Path
from typing import Self
import pandas as pd
import numpy as np

from battery_soh.core.types import BatchId, Features, Labels, BatteryIds
from battery_soh.core.constants import XJTU_BATCHES, DEFAULT_DATA_DIR
from battery_soh.data.transforms import extract_features, compute_soh


class XJTULoader:
    """XJTU battery dataset loader.
    
    Loads battery cycling data from the XJTU dataset. The dataset contains
    lithium-ion batteries tested under different charging conditions.
    
    Args:
        data_dir: Path to the XJTU data directory. If None, uses default.
        
    Example:
        >>> loader = XJTULoader()
        >>> X, y, battery_ids = loader.load_batch("2C")
        >>> print(f"Loaded {len(X)} samples")
        Loaded 12345 samples
        
    Reference:
        XJTU Battery Dataset: https://doi.org/10.1016/j.jpowsour.2021.230639
    """
    
    BATTERIES: tuple[BatchId, ...] = XJTU_BATCHES
    """Valid batch identifiers."""
    
    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize loader.
        
        Args:
            data_dir: Path to XJTU data directory. Uses DEFAULT_DATA_DIR if None.
        """
        self.data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download the XJTU dataset or specify correct path."
            )
    
    def load_batch(
        self, 
        batch_id: BatchId
    ) -> tuple[Features, Labels, BatteryIds]:
        """Load a specific batch.
        
        Loads all batteries from the specified batch and combines them
        into feature matrix, labels, and battery identifiers.
        
        Args:
            batch_id: One of "2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"
            
        Returns:
            Tuple of (features, labels, battery_ids):
                - features: [N_SAMPLES, N_FEATURES] array
                - labels: [N_SAMPLES] array of SOH values
                - battery_ids: [N_SAMPLES] array of battery identifiers
                
        Raises:
            ValueError: If batch_id is invalid
            FileNotFoundError: If batch data files not found
        """
        if batch_id not in self.BATTERIES:
            raise ValueError(
                f"Invalid batch_id: {batch_id}. "
                f"Must be one of: {self.BATTERIES}"
            )
        
        # Find all battery files for this batch
        pattern = f"{batch_id}_battery-*.csv"
        battery_files = sorted(self.data_dir.glob(pattern))
        
        if not battery_files:
            raise FileNotFoundError(
                f"No battery files found for batch {batch_id} "
                f"with pattern {pattern} in {self.data_dir}"
            )
        
        # Load and combine all batteries
        features_list = []
        labels_list = []
        battery_ids_list = []
        
        for file_path in battery_files:
            battery_id = file_path.stem
            df = pd.read_csv(file_path)
            
            # Extract features and labels
            X = extract_features(df)
            y = compute_soh(df)
            
            features_list.append(X)
            labels_list.append(y)
            battery_ids_list.extend([battery_id] * len(X))
        
        # Combine
        X = np.vstack(features_list).astype(np.float32)
        y = np.concatenate(labels_list).astype(np.float32)
        battery_ids = np.array(battery_ids_list, dtype=str)
        
        return X, y, battery_ids
    
    def list_batches(self) -> list[BatchId]:
        """Return list of available batches.
        
        Returns:
            List of batch IDs that can be loaded
        """
        return list(self.BATTERIES)
    
    def list_available_batches(self) -> list[BatchId]:
        """Return list of batches that have data files.
        
        Unlike list_batches(), this checks which batches actually
        have data files present.
        
        Returns:
            List of batch IDs with available data
        """
        available = []
        for batch_id in self.BATTERIES:
            pattern = f"{batch_id}_battery-*.csv"
            if list(self.data_dir.glob(pattern)):
                available.append(batch_id)
        return available
