"""Missing data generators: MCAR, MAR, MNAR, and Block (for A2)."""

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
        """Generate MCAR missing pattern."""
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        rng = np.random.default_rng(int(seed))
        mask = rng.random(X.shape) > missing_rate
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask


class MARGenerator:
    """MAR (Missing At Random): Missingness depends on observed SOH."""
    
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
        """Generate MAR missing pattern based on SOH."""
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        n_samples = X.shape[0]
        
        if soh is None:
            soh = np.linspace(1.0, 0.7, n_samples).astype(np.float32)
        
        if len(soh) != n_samples:
            raise ValueError(f"SOH length {len(soh)} != X samples {n_samples}")
        
        soh = np.clip(soh, 0.0, 1.0)
        
        expected_soh_power = np.mean(soh ** beta)
        if expected_soh_power < 1e-10:
            alpha = 0.0
        else:
            alpha = (missing_rate - gamma) / expected_soh_power
            alpha = np.clip(alpha, 0.0, 1.0)
        
        missing_probs = alpha * (soh ** beta) + gamma
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        rng = np.random.default_rng(int(seed))
        probs_matrix = np.broadcast_to(missing_probs[:, np.newaxis], X.shape)
        mask = rng.random(X.shape) > probs_matrix
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask


class MNARGenerator:
    """MNAR (Missing Not At Random): Missingness depends on missing values themselves."""
    
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
        """Generate MNAR missing pattern based on feature values."""
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        driver = X[:, feature_idx]
        
        f_min, f_max = driver.min(), driver.max()
        if f_max - f_min < 1e-10:
            normalized = np.ones_like(driver) * 0.5
        else:
            normalized = (driver - f_min) / (f_max - f_min)
        
        expected_1_minus_x = np.mean(1 - normalized)
        expected_x_power = np.mean(normalized ** gamma)
        
        numerator = missing_rate - beta * expected_x_power
        if expected_1_minus_x > 1e-10 and numerator > 0:
            alpha = numerator / expected_1_minus_x
            alpha = min(alpha, 2.0)
        else:
            alpha = 0.0
        
        missing_probs = alpha * (1 - normalized) + beta * (normalized ** gamma)
        missing_probs = np.clip(missing_probs, 0.0, 1.0)
        
        rng = np.random.default_rng(int(seed))
        probs_matrix = np.broadcast_to(missing_probs[:, np.newaxis], X.shape)
        mask = rng.random(X.shape) > probs_matrix
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask


class BlockMissingGenerator:
    """Block missing pattern (for A2): Continuous cycles missing, simulating sensor failure.
    
    This generates realistic sensor failure patterns where multiple consecutive
    measurements are lost together.
    
    Args:
        block_size: Number of consecutive samples to mask together (default: 5)
    """
    
    def __init__(self, block_size: int = 5):
        self.block_size = block_size
    
    @property
    def name(self) -> str:
        return f"Block({self.block_size})"
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed,
        **kwargs
    ) -> tuple[Features, Mask]:
        """Generate block missing pattern.
        
        Strategy:
        1. Calculate how many blocks to mask based on missing_rate
        2. Randomly select starting positions for blocks
        3. Mask block_size consecutive samples from each position
        
        Example:
            X shape: (100, 16), block_size=5, missing_rate=0.3
            -> Mask ~30 blocks (30*5=150 samples out of 1600 total features)
            -> But we work per-feature: mask 30% of feature values in blocks
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        
        n_samples, n_features = X.shape
        rng = np.random.default_rng(int(seed))
        
        # Initialize mask (True = observed)
        mask = np.ones((n_samples, n_features), dtype=bool)
        
        # For each feature, create block missing patterns
        for feat_idx in range(n_features):
            # Calculate number of blocks to achieve target missing rate
            # Each block masks block_size samples
            n_total_blocks = n_samples // self.block_size
            n_missing_blocks = int(n_total_blocks * missing_rate)
            
            if n_missing_blocks == 0:
                continue
            
            # Randomly select starting block positions
            available_blocks = list(range(n_total_blocks))
            rng.shuffle(available_blocks)
            selected_blocks = available_blocks[:n_missing_blocks]
            
            # Mask the selected blocks
            for block_idx in selected_blocks:
                start = block_idx * self.block_size
                end = min(start + self.block_size, n_samples)
                mask[start:end, feat_idx] = False
        
        # Apply mask
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
