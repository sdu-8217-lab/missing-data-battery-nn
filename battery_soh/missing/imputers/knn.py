"""K-Nearest Neighbors imputation method."""

import numpy as np

try:
    from sklearn.impute import KNNImputer as SklearnKNNImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from battery_soh.core.types import Features
from battery_soh.missing.imputers.base import BaseImputer


class KNNImputer(BaseImputer):
    """Impute missing values using K-Nearest Neighbors.
    
    For each sample with missing values, finds K nearest neighbors
    in the feature space and uses their mean as the imputed value.
    
    This is a multivariate method that can capture feature correlations.
    
    Args:
        n_neighbors: Number of neighbors to use (default: 5)
        weights: Weighting scheme, 'uniform' or 'distance' (default: 'uniform')
        
    Example:
        >>> imputer = KNNImputer(n_neighbors=5)
        >>> imputer.fit(X_train)
        >>> X_filled = imputer.transform(X_missing)
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        if not HAS_SKLEARN:
            raise ImportError(
                "KNNImputer requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer: SklearnKNNImputer | None = None
    
    @property
    def name(self) -> str:
        """Return imputer name."""
        return "knn"
    
    def fit(self, X: Features) -> "KNNImputer":
        """Fit KNN imputer on complete data.
        
        Args:
            X: Complete feature matrix
            
        Returns:
            Self
        """
        self._imputer = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values using KNN.
        
        Args:
            X_missing: Feature matrix with NaN
            
        Returns:
            Feature matrix with imputed values
            
        Raises:
            RuntimeError: If fit() not called before transform()
        """
        if self._imputer is None:
            raise RuntimeError("Must call fit() before transform()")
        
        return self._imputer.transform(X_missing)
