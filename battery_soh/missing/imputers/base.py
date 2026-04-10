"""Base class for imputation methods."""

from abc import ABC, abstractmethod
import numpy as np

from battery_soh.core.types import Features, Mask


class BaseImputer(ABC):
    """Abstract base class for imputation methods.
    
    All imputers follow the scikit-learn fit/transform pattern:
    1. Fit on complete training data to learn statistics
    2. Transform incomplete data by filling missing values
    
    Example:
        >>> imputer = MeanImputer()
        >>> imputer.fit(X_train)  # Learn means from complete data
        >>> X_filled = imputer.transform(X_missing)  # Fill missing
    """
    
    @abstractmethod
    def fit(self, X: Features) -> "BaseImputer":
        """Fit imputer on complete data.
        
        Args:
            X: Complete feature matrix (no missing values)
            
        Returns:
            Self for method chaining
        """
        ...
    
    @abstractmethod
    def transform(self, X_missing: Features) -> Features:
        """Impute missing values.
        
        Args:
            X_missing: Feature matrix with NaN for missing values
            
        Returns:
            Feature matrix with imputed values
        """
        ...
    
    def fit_transform(
        self, 
        X: Features, 
        mask: Mask | None = None
    ) -> Features:
        """Fit and transform in one step.
        
        Args:
            X: Feature matrix (with or without NaN)
            mask: Boolean mask indicating observed values.
                  If None, assumes NaN indicates missing.
                  
        Returns:
            Feature matrix with imputed values
        """
        if mask is not None:
            # Create copy with NaN for missing
            X_missing = X.copy()
            X_missing[~mask] = np.nan
        else:
            X_missing = X
        
        return self.fit(X).transform(X_missing)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return imputer name."""
        ...
