"""Iterative imputation method (MICE-style)."""

import numpy as np

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer as SklearnIterativeImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from battery_soh.core.types import Features
from battery_soh.missing.imputers.base import BaseImputer


class IterativeImputer(BaseImputer):
    """Impute missing values using iterative regression (MICE).
    
    Models each feature with missing values as a function of other features
    and uses that estimate for imputation. Iterates until convergence.
    
    This is the most sophisticated imputation method, capturing complex
    feature relationships.
    
    Args:
        max_iter: Maximum number of iterations (default: 10)
        estimator: Underlying estimator ('bayesian_ridge' or 'random_forest')
        random_state: Random seed for reproducibility
        
    Reference:
        Buuren, S. V., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
        Imputation by Chained Equations in R. JSS, 45(3), 1-67.
    """
    
    def __init__(
        self,
        max_iter: int = 10,
        estimator: str = "bayesian_ridge",
        random_state: int = 42
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "IterativeImputer requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        self.max_iter = max_iter
        self.estimator = estimator
        self.random_state = random_state
        self._imputer: SklearnIterativeImputer | None = None
    
    @property
    def name(self) -> str:
        """Return imputer name."""
        return "iterative"
    
    def fit(self, X: Features) -> "IterativeImputer":
        """Fit iterative imputer on complete data.
        
        Args:
            X: Complete feature matrix
            
        Returns:
            Self
        """
        # Select estimator
        if self.estimator == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=10,
                max_depth=10,
                random_state=self.random_state
            )
        else:  # bayesian_ridge
            from sklearn.linear_model import BayesianRidge
            estimator = BayesianRidge()
        
        self._imputer = SklearnIterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=False  # Deterministic imputation
        )
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: Features) -> Features:
        """Fill missing values using iterative regression.
        
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
