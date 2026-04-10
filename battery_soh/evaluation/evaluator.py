"""Model evaluator for missing data scenarios."""

import numpy as np
import torch

from battery_soh.core.types import (
    Features, Labels, Mask, Seed, MissingRate, MissingMode, ImputationMethod
)
from battery_soh.missing.generators import MCARGenerator, MARGenerator, MNARGenerator, BlockMissingGenerator
from battery_soh.missing.imputers import (
    MeanImputer, KNNImputer, IterativeImputer, ZeroImputer
)
from battery_soh.evaluation.metrics import compute_metrics, Metrics


class Evaluator:
    """Evaluate model under missing data scenarios.
    
    Supports A1 ablation study variants:
    - G1 (MIM): z = [x̂ | m] - standard MIM
    - G2 (Random): z = [x̂ | r], r ~ Bernoulli(0.4) - test regularization hypothesis
    - G3 (Shuffled): z = [x̂ | m_shuffled] - test information hypothesis  
    - G4 (Copy): z = [x̂ | x̂] - test dimensionality hypothesis
    
    And A2 missing patterns:
    - B1 (MCAR): Random uniform missing
    - B2 (Block): Continuous block missing (simulates sensor failure)
    """
    
    def __init__(
        self,
        missing_mode: MissingMode | str = "MCAR",
        missing_rate: MissingRate = 0.3,
        imputation_method: ImputationMethod = "mean",
        use_mim: bool = False,
        mim_variant: str = "standard",  # "standard", "random", "shuffled", "copy"
        block_size: int | None = None,  # For block missing pattern
    ):
        """Initialize evaluator.
        
        Args:
            missing_mode: MCAR, MAR, MNAR, or "block" for block missing
            missing_rate: Proportion of missing values
            imputation_method: mean, knn, iterative, or zero
            use_mim: Whether to use Missing Indicator Method
            mim_variant: For A1 study - "standard", "random", "shuffled", "copy"
            block_size: For block missing pattern (A2), e.g., 5 for 5-cycle blocks
        """
        self.missing_mode = missing_mode
        self.missing_rate = missing_rate
        self.imputation_method = imputation_method
        self.use_mim = use_mim
        self.mim_variant = mim_variant
        self.block_size = block_size
        
        # Create generator
        self._generator = self._create_generator(missing_mode, block_size)
        
        # Create imputer
        self._imputer = self._create_imputer(imputation_method)
    
    def _create_generator(self, mode: MissingMode | str, block_size: int | None):
        """Create missing pattern generator."""
        if mode == "block" or block_size is not None:
            return BlockMissingGenerator(block_size or 5)
        
        generators = {
            "MCAR": MCARGenerator(),
            "MAR": MARGenerator(),
            "MNAR": MNARGenerator()
        }
        return generators[mode]
    
    def _create_imputer(self, method: ImputationMethod):
        """Create imputation method."""
        imputers = {
            "mean": MeanImputer(),
            "knn": KNNImputer(),
            "iterative": IterativeImputer(),
            "zero": ZeroImputer()
        }
        return imputers[method]
    
    def _construct_mim_input(
        self,
        X_imputed: Features,
        mask: Mask,
        seed: Seed
    ) -> Features:
        """Construct MIM input with variant handling for A1 study.
        
        Args:
            X_imputed: Imputed features
            mask: Boolean mask (True = observed)
            seed: Random seed
            
        Returns:
            Input array with MIM variant applied
        """
        n_samples, n_features = X_imputed.shape
        rng = np.random.default_rng(int(seed) + 1000)  # Different seed from missing generation
        
        if self.mim_variant == "standard":
            # G1: Standard MIM - z = [x̂ | m]
            mask_indicator = (~mask).astype(np.float32)
            return np.concatenate([X_imputed, mask_indicator], axis=1)
        
        elif self.mim_variant == "random":
            # G2: Random indicator - z = [x̂ | r], r ~ Bernoulli(0.4)
            random_mask = rng.random((n_samples, n_features)) < 0.4
            random_indicator = random_mask.astype(np.float32)
            return np.concatenate([X_imputed, random_indicator], axis=1)
        
        elif self.mim_variant == "shuffled":
            # G3: Shuffled mask - z = [x̂ | m_shuffled]
            mask_indicator = (~mask).astype(np.float32)
            # Shuffle each column independently
            shuffled_indicator = np.zeros_like(mask_indicator)
            for j in range(n_features):
                col = mask_indicator[:, j].copy()
                rng.shuffle(col)
                shuffled_indicator[:, j] = col
            return np.concatenate([X_imputed, shuffled_indicator], axis=1)
        
        elif self.mim_variant == "copy":
            # G4: Copy imputed values - z = [x̂ | x̂]
            return np.concatenate([X_imputed, X_imputed.copy()], axis=1)
        
        else:
            raise ValueError(f"Unknown mim_variant: {self.mim_variant}")
    
    def evaluate(
        self,
        model: torch.nn.Module,
        X: Features,
        y: Labels,
        seed: Seed,
        soh: Labels | None = None
    ) -> Metrics:
        """Evaluate model on data with missing values."""
        # 1. Generate missing pattern
        if self.missing_mode == "MAR" and soh is not None:
            X_missing, mask = self._generator.generate(
                X, self.missing_rate, seed, soh=soh
            )
        else:
            X_missing, mask = self._generator.generate(
                X, self.missing_rate, seed
            )
        
        # 2. Impute missing values
        X_imputed = self._imputer.fit_transform(X_missing)
        
        # 3. Add MIM mask if needed
        if self.use_mim:
            X_input = self._construct_mim_input(X_imputed, mask, seed)
        else:
            X_input = X_imputed
        
        # 4. Run inference
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_input).float()
            y_pred = model(X_tensor).numpy()
        
        # 5. Compute metrics
        metrics = compute_metrics(y, y_pred)
        
        return metrics
