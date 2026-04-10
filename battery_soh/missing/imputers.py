"""Imputation methods for missing data."""

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

from battery_soh.core.types import Features, Mask


class Imputer(Protocol):
    """Protocol for imputation methods."""
    
    @property
    def name(self) -> str:
        ...
    
    def fit(self, X: Features) -> "Imputer":
        ...
    
    def transform(self, X_missing: Features) -> Features:
        ...


class BaseImputer(ABC):
    """Abstract base class for imputation methods."""
    
    @abstractmethod
    def fit(self, X: Features) -> "BaseImputer":
        """Fit imputer on complete data."""
        ...
    
    @abstractmethod
    def transform(self, X_missing: Features) -> Features:
        """Impute missing values."""
        ...
    
    def fit_transform(self, X: Features, mask: Mask | None = None) -> Features:
        """Fit and transform in one step."""
        if mask is not None:
            X_missing = X.copy()
            X_missing[~mask] = np.nan
        else:
            X_missing = X
        return self.fit(X).transform(X_missing)
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MeanImputer(BaseImputer):
    """Impute missing values with column means."""
    
    def __init__(self):
        self._means: np.ndarray | None = None
    
    @property
    def name(self) -> str:
        return "mean"
    
    def fit(self, X: Features) -> "MeanImputer":
        """Compute column means from complete data."""
        self._means = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values with learned means."""
        if self._means is None:
            raise RuntimeError("Must call fit() before transform()")
        
        X_filled = X_missing.copy()
        for i in range(X_missing.shape[1]):
            mask = np.isnan(X_filled[:, i])
            if mask.any():
                X_filled[mask, i] = self._means[i]
        return X_filled


class ZeroImputer(BaseImputer):
    """Impute missing values with zeros."""
    
    @property
    def name(self) -> str:
        return "zero"
    
    def fit(self, X: Features) -> "ZeroImputer":
        """No fitting required."""
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values with zeros."""
        return np.nan_to_num(X_missing.copy(), nan=0.0)


class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputation."""
    
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        try:
            from sklearn.impute import KNNImputer as SklearnKNNImputer
            self._SklearnKNNImputer = SklearnKNNImputer
        except ImportError:
            raise ImportError("KNNImputer requires scikit-learn")
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = None
    
    @property
    def name(self) -> str:
        return "knn"
    
    def fit(self, X: Features) -> "KNNImputer":
        """Fit KNN imputer on complete data."""
        self._imputer = self._SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values using KNN."""
        if self._imputer is None:
            raise RuntimeError("Must call fit() before transform()")
        return self._imputer.transform(X_missing)


class IterativeImputer(BaseImputer):
    """Iterative imputation (MICE-style)."""
    
    def __init__(
        self,
        max_iter: int = 10,
        estimator: str = "bayesian_ridge",
        random_state: int = 42
    ):
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer as SklearnIterativeImputer
            self._SklearnIterativeImputer = SklearnIterativeImputer
        except ImportError:
            raise ImportError("IterativeImputer requires scikit-learn")
        
        self.max_iter = max_iter
        self.estimator = estimator
        self.random_state = random_state
        self._imputer = None
    
    @property
    def name(self) -> str:
        return "iterative"
    
    def fit(self, X: Features) -> "IterativeImputer":
        """Fit iterative imputer on complete data."""
        if self.estimator == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=10, max_depth=10, random_state=self.random_state
            )
        else:
            from sklearn.linear_model import BayesianRidge
            estimator = BayesianRidge()
        
        self._imputer = self._SklearnIterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=False
        )
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values using iterative regression."""
        if self._imputer is None:
            raise RuntimeError("Must call fit() before transform()")
        return self._imputer.transform(X_missing)
