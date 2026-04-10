"""Mean imputation method."""

import numpy as np

from battery_soh.core.types import Features
from battery_soh.missing.imputers.base import BaseImputer


class MeanImputer(BaseImputer):
    """Impute missing values with column means.
    
    Simple univariate imputation using the mean of each feature.
    This is a baseline method that assumes missing values are
    approximately at the mean.
    
    Example:
        >>> imputer = MeanImputer()
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
        >>> imputer.fit(X_train)
        >>> X_missing = np.array([[np.nan, 2], [3, np.nan]])
        >>> X_filled = imputer.transform(X_missing)
        >>> print(X_filled)
        [[3. 2.]
         [3. 4.]]
    """
    
    def __init__(self):
        self._means: np.ndarray | None = None
    
    @property
    def name(self) -> str:
        """Return imputer name."""
        return "mean"
    
    def fit(self, X: Features) -> "MeanImputer":
        """Compute column means from complete data.
        
        Args:
            X: Complete feature matrix
            
        Returns:
            Self
        """
        self._means = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values with learned means.
        
        Args:
            X_missing: Feature matrix with NaN
            
        Returns:
            Feature matrix with NaN replaced by means
            
        Raises:
            RuntimeError: If fit() not called before transform()
        """
        if self._means is None:
            raise RuntimeError("Must call fit() before transform()")
        
        X_filled = X_missing.copy()
        
        # Fill each column with its mean
        for i in range(X_missing.shape[1]):
            mask = np.isnan(X_filled[:, i])
            if mask.any():
                X_filled[mask, i] = self._means[i]
        
        return X_filled
