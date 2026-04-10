"""
MAR (Missing At Random) generator.

MAR: Missingness depends on observed data (SOH), but not on the missing values themselves.

In battery prognostics, this simulates scenarios where sensor reliability
degrades as the battery ages (lower SOH → higher missing rate).

Formula (from meta.md):
    p_i = alpha * SOH_i^beta + gamma

Where:
    - p_i: Missing probability for sample i
    - SOH_i: State of Health for sample i
    - alpha, beta, gamma: Parameters controlling the relationship

Reference:
    Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
"""

import numpy as np

from battery_soh.core.types import Features, Mask, MissingRate, Seed, Labels


class MARGenerator:
    """Generate MAR missing patterns dependent on SOH.
    
    Missing probability increases as SOH decreases (battery ages).
    This models sensor degradation in aging batteries.
    
    Example:
        >>> generator = MARGenerator()
        >>> X = np.random.randn(100, 16).astype(np.float32)
        >>> soh = np.linspace(1.0, 0.7, 100).astype(np.float32)  # Degrading
        >>> X_missing, mask = generator.generate(
        ...     X, missing_rate=0.3, seed=42, soh=soh
        ... )
    """
    
    @property
    def name(self) -> str:
        """Return generator name."""
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
            X: Original feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target overall proportion of missing values
            seed: Random seed for reproducibility
            soh: SOH values [N_SAMPLES]. If None, uses uniform distribution.
            beta: Exponent controlling non-linearity (default: 2.0)
            gamma: Baseline missing probability (default: 0.05)
            **kwargs: Unused
            
        Returns:
            Tuple of (X_with_missing, mask)
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        n_samples = X.shape[0]
        
        # Use uniform SOH if not provided (degrades from 1.0 to 0.7)
        if soh is None:
            soh = np.linspace(1.0, 0.7, n_samples).astype(np.float32)
        
        if len(soh) != n_samples:
            raise ValueError(f"SOH length {len(soh)} != X samples {n_samples}")
        
        # Clip SOH to valid range
        soh = np.clip(soh, 0.0, 1.0)
        
        # Compute alpha to achieve target missing rate
        # E[p] = alpha * E[SOH^beta] + gamma = missing_rate
        expected_soh_power = np.mean(soh ** beta)
        
        if expected_soh_power < 1e-10:
            alpha = 0.0
        else:
            alpha = (missing_rate - gamma) / expected_soh_power
            alpha = np.clip(alpha, 0.0, 1.0)
        
        # Compute per-sample missing probabilities
        # p_i = alpha * SOH_i^beta + gamma
        missing_probs = alpha * (soh ** beta) + gamma
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        # Generate mask
        rng = np.random.default_rng(int(seed))
        
        # Expand probabilities to match X shape
        # Each sample has same prob for all features
        probs_matrix = np.broadcast_to(
            missing_probs[:, np.newaxis], 
            X.shape
        )
        
        mask = rng.random(X.shape) > probs_matrix
        
        # Apply missingness
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
