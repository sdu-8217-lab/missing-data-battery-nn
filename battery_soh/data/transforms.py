"""
Feature engineering and data transformations.

This module handles the conversion from raw battery cycling data
to machine learning features.
"""

import numpy as np
import pandas as pd

from battery_soh.core.types import Features, Labels
from battery_soh.core.constants import N_FEATURES


def extract_features(df: pd.DataFrame) -> Features:
    """Extract features from raw battery data.
    
    Extracts 16 statistical features from voltage and current measurements:
    - Voltage statistics: mean, std, kurtosis, skewness
    - Current statistics: mean, std, kurtosis, skewness
    - Charge features: CC Q, CC charge time, CV Q, CV charge time
    - Advanced features: voltage slope, current slope, voltage entropy, current entropy
    
    Args:
        df: Raw battery data DataFrame containing voltage, current, etc.
        
    Returns:
        Feature matrix of shape [N_SAMPLES, N_FEATURES]
        
    Raises:
        KeyError: If required columns are missing
    """
    required_cols = [
        "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
        "current mean", "current std", "current kurtosis", "current skewness",
        "CC Q", "CC charge time", "CV Q", "CV charge time",
        "voltage slope", "current slope", "voltage entropy", "current entropy"
    ]
    
    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Extract features
    features = df[required_cols].values.astype(np.float32)
    
    # Clean data: replace inf with nan, then fill nan with column mean
    features = np.where(np.isfinite(features), features, np.nan)
    col_means = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        mask = np.isnan(features[:, i])
        if mask.any():
            features[mask, i] = col_means[i]
    
    # Clip extreme values to prevent training instability
    features = np.clip(features, -1e6, 1e6)
    
    return features


def compute_soh(df: pd.DataFrame, nominal_capacity: float | None = None) -> Labels:
    """Compute State of Health (SOH) from capacity.
    
    SOH is defined as current_capacity / nominal_capacity.
    If nominal_capacity is not provided, uses the first capacity value.
    
    Args:
        df: DataFrame containing 'capacity' column
        nominal_capacity: Nominal capacity value. If None, uses first capacity.
        
    Returns:
        SOH values as array of shape [N_SAMPLES]
        
    Raises:
        KeyError: If 'capacity' column is missing
        ValueError: If nominal_capacity is zero
    """
    if "capacity" not in df.columns:
        raise KeyError("DataFrame must contain 'capacity' column")
    
    capacity = df["capacity"].values.astype(np.float32)
    
    if nominal_capacity is None:
        nominal_capacity = float(capacity[0])
    
    if nominal_capacity == 0:
        raise ValueError("Nominal capacity cannot be zero")
    
    soh = capacity / nominal_capacity
    
    return soh


def standardize_features(
    X_train: Features,
    X_val: Features | None = None,
    X_test: Features | None = None
) -> tuple[Features, ...]:
    """Standardize features using training set statistics.
    
    Uses (X - mean) / std transformation.
    Handles zero std by setting to 1 to avoid division by zero.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        
    Returns:
        Tuple of standardized features (train, val, test)
        Only returns non-None inputs.
    """
    # Compute statistics from training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    # Handle zero std
    std = np.where(std == 0, 1.0, std)
    
    # Standardize
    X_train_scaled = (X_train - mean) / std
    
    result = [X_train_scaled.astype(np.float32)]
    
    if X_val is not None:
        X_val_scaled = ((X_val - mean) / std).astype(np.float32)
        result.append(X_val_scaled)
    
    if X_test is not None:
        X_test_scaled = ((X_test - mean) / std).astype(np.float32)
        result.append(X_test_scaled)
    
    return tuple(result)
