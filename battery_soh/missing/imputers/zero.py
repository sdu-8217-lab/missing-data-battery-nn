"""Zero imputation method."""

import numpy as np

from battery_soh.core.types import Features
from battery_soh.missing.imputers.base import BaseImputer


class ZeroImputer(BaseImputer):
    """Impute missing values with zeros.
    
    Simplest baseline method. Assumes features are centered
    or that zero is a meaningful default value.
    
    Note:
        This should only be used if features are standardized
        (mean=0), otherwise it introduces bias.
    """
    
    @property
    def name(self) -> str:
        """Return imputer name."""
        return "zero"
    
    def fit(self, X: Features) -> "ZeroImputer":
        """No fitting required for zero imputation.
        
        Args:
            X: Complete feature matrix (unused)
            
        Returns:
            Self
        """
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values with zeros.
        
        Args:
            X_missing: Feature matrix with NaN
            
        Returns:
            Feature matrix with NaN replaced by 0.0
        """
        X_filled = X_missing.copy()
        X_filled = np.nan_to_num(X_filled, nan=0.0)
        return X_filled
