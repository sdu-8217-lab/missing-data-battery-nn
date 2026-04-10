"""Missing data generators: MCAR, MAR, MNAR.

Reference:
    Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
"""

import numpy as np

from battery_soh.core.types import Features, Labels, Mask, MissingRate, Seed


class MCARGenerator:
    """MCAR (Missing Completely At Random): Missingness independent of all data."""
    
    @property
    def name(self) -> str:
        return "MCAR"
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed,
        **kwargs
    ) -> tuple[Features, Mask]:
        """Generate MCAR missing pattern.
        
        Args:
            X: Feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target proportion of missing values [0, 1]
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (X_with_missing, mask) where mask True = observed
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        rng = np.random.default_rng(int(seed))
        mask = rng.random(X.shape) > missing_rate
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask


class MARGenerator:
    """MAR (Missing At Random): Missingness depends on observed SOH.
    
    Models sensor degradation: lower SOH (aging battery) -> higher missing rate.
    Formula: p_i = alpha * SOH_i^beta + gamma
    """
    
    @property
    def name(self) -> str:
        return "MAR"
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed,
        soh: Labels | None = None,
        beta: float = 2.0,
        gamma: float = 0.05,
        **kwargs
    ) -> tuple[Features, Mask]:
        """Generate MAR missing pattern based on SOH.
        
        Args:
            X: Feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target overall proportion of missing values
            seed: Random seed
            soh: SOH values [N_SAMPLES]. If None, uses uniform degradation
            beta: Exponent controlling non-linearity
            gamma: Baseline missing probability
            
        Returns:
            Tuple of (X_with_missing, mask)
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        n_samples = X.shape[0]
        
        # Default: uniform SOH degradation from 1.0 to 0.7
        if soh is None:
            soh = np.linspace(1.0, 0.7, n_samples).astype(np.float32)
        
        if len(soh) != n_samples:
            raise ValueError(f"SOH length {len(soh)} != X samples {n_samples}")
        
        soh = np.clip(soh, 0.0, 1.0)
        
        # Compute alpha to achieve target missing rate
        expected_soh_power = np.mean(soh ** beta)
        if expected_soh_power < 1e-10:
            alpha = 0.0
        else:
            alpha = (missing_rate - gamma) / expected_soh_power
            alpha = np.clip(alpha, 0.0, 1.0)
        
        # Per-sample missing probabilities
        missing_probs = alpha * (soh ** beta) + gamma
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        # Generate mask
        rng = np.random.default_rng(int(seed))
        probs_matrix = np.broadcast_to(missing_probs[:, np.newaxis], X.shape)
        mask = rng.random(X.shape) > probs_matrix
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask


class MNARGenerator:
    """MNAR (Missing Not At Random): Missingness depends on missing values themselves.
    
    Models sensor failures at extreme measurements (e.g., very low voltage).
    """
    
    @property
    def name(self) -> str:
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
            X: Feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target overall proportion of missing values
            seed: Random seed
            feature_idx: Which feature drives missingness
            beta: Baseline missing probability
            gamma: Non-linearity parameter
            
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
            normalized = np.ones_like(driver) * 0.5
        else:
            normalized = (driver - f_min) / (f_max - f_min)
        
        # Compute alpha for target missing rate
        expected_1_minus_x = np.mean(1 - normalized)
        expected_x_power = np.mean(normalized ** gamma)
        
        numerator = missing_rate - beta * expected_x_power
        if expected_1_minus_x > 1e-10 and numerator > 0:
            alpha = numerator / expected_1_minus_x
            alpha = min(alpha, 2.0)
        else:
            alpha = 0.0
        
        # Higher prob for extreme values (close to min)
        missing_probs = alpha * (1 - normalized) + beta * (normalized ** gamma)
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        # Generate mask
        rng = np.random.default_rng(int(seed))
        probs_matrix = np.broadcast_to(missing_probs[:, np.newaxis], X.shape)
        mask = rng.random(X.shape) > probs_matrix
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
