"""
MCAR (Missing Completely At Random) generator.

MCAR: Missingness is independent of both observed and unobserved data.
This is the simplest missing mechanism, equivalent to uniform random sampling.

Reference:
    Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
"""

import numpy as np

from battery_soh.core.types import Features, Mask, MissingRate, Seed


class MCARGenerator:
    """Generate MCAR missing patterns.
    
    Each element has equal probability of being missing,
    independent of its value or other values.
    
    Example:
        >>> generator = MCARGenerator()
        >>> X = np.random.randn(100, 16).astype(np.float32)
        >>> X_missing, mask = generator.generate(X, missing_rate=0.3, seed=42)
        >>> print(f"Actual missing rate: {(~mask).sum() / mask.size:.2%}")
        Actual missing rate: 29.81%
    """
    
    @property
    def name(self) -> str:
        """Return generator name."""
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
            X: Original feature matrix [N_SAMPLES, N_FEATURES]
            missing_rate: Target proportion of missing values [0, 1]
            seed: Random seed for reproducibility
            **kwargs: Unused (for interface compatibility)
            
        Returns:
            Tuple of (X_with_missing, mask):
                - X_with_missing: Array with NaN for missing values
                - mask: Boolean mask, True = observed, False = missing
                
        Raises:
            ValueError: If missing_rate not in [0, 1]
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        # Create random generator with seed
        rng = np.random.default_rng(int(seed))
        
        # Generate random mask: True = observed, False = missing
        mask = rng.random(X.shape) > missing_rate
        
        # Create copy with NaN for missing values
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
