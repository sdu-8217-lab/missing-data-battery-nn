"""
MNAR (Missing Not At Random) generator.

MNAR: Missingness depends on the missing values themselves.

In battery data, this simulates sensor failures at extreme measurements
(e.g., very low voltage causes sensor to fail).

Formula:
    p_missing(i,j) = alpha * (1 - normalized_X[i,j]) + beta * normalized_X[i,j]^gamma

This creates higher missing probability for extreme (minimum) values.

Reference:
    Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
"""

import numpy as np

from battery_soh.core.types import Features, Mask, MissingRate, Seed


class MNARGenerator:
    """Generate MNAR missing patterns dependent on feature values.
    
    Extreme feature values (typically minima) have higher missing probability.
    This models sensor failures under extreme operating conditions.
    
    Example:
        >>> generator = MNARGenerator()
        >>> X = np.random.randn(100, 16).astype(np.float32)
        >>> X_missing, mask = generator.generate(X, missing_rate=0.3, seed=42)
    """
    
    @property
    def name(self) -> str:
        """Return generator name."""
        return "MNAR"
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed,
        feature_idx: int = 0,
        beta: float = 0.05,
        gamma: float = 1.0,
        **kwargs
    ) -> tuple[Features, Mask]:
        """Generate MNAR missing pattern based on feature values.
        
        Args:
            X: Original feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target overall proportion of missing values
            seed: Random seed for reproducibility
            feature_idx: Which feature drives missingness (default: 0)
            beta: Baseline missing probability (default: 0.05)
            gamma: Non-linearity parameter (default: 1.0)
            **kwargs: Unused
            
        Returns:
            Tuple of (X_with_missing, mask)
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        # Get driving feature
        driver = X[:, feature_idx]
        
        # Normalize to [0, 1]
        f_min, f_max = driver.min(), driver.max()
        if f_max - f_min < 1e-10:
            # Constant feature, use uniform
            normalized = np.ones_like(driver) * 0.5
        else:
            normalized = (driver - f_min) / (f_max - f_min)
        
        # Compute alpha for target missing rate
        # E[p] ≈ alpha * E[1-x] + beta * E[x^gamma]
        expected_1_minus_x = np.mean(1 - normalized)
        expected_x_power = np.mean(normalized ** gamma)
        
        numerator = missing_rate - beta * expected_x_power
        if expected_1_minus_x > 1e-10 and numerator > 0:
            alpha = numerator / expected_1_minus_x
            alpha = min(alpha, 2.0)  # Cap to avoid excessive missingness
        else:
            alpha = 0.0
        
        # Compute per-sample missing probabilities
        # Higher prob for extreme values (close to min)
        missing_probs = alpha * (1 - normalized) + beta * (normalized ** gamma)
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        # Generate mask for all features (same prob per sample)
        rng = np.random.default_rng(int(seed))
        probs_matrix = np.broadcast_to(missing_probs[:, np.newaxis], X.shape)
        mask = rng.random(X.shape) > probs_matrix
        
        # Apply missingness
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
