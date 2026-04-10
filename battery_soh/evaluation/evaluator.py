"""Model evaluator for missing data scenarios."""

import numpy as np
import torch

from battery_soh.core.types import (
    Features, Labels, Seed, MissingRate, MissingMode, ImputationMethod
)
from battery_soh.core.interfaces import MissingGenerator, Imputer
from battery_soh.missing.generators import MCARGenerator, MARGenerator, MNARGenerator
from battery_soh.missing.imputers import (
    MeanImputer, KNNImputer, IterativeImputer, ZeroImputer
)
from battery_soh.evaluation.metrics import compute_metrics, Metrics


class Evaluator:
    """Evaluate model under missing data scenarios.
    
    This class orchestrates the complete evaluation pipeline:
    1. Generate missing pattern
    2. Apply imputation
    3. Add MIM mask if needed
    4. Run model inference
    5. Compute metrics
    
    Example:
        >>> evaluator = Evaluator(
        ...     missing_mode="MCAR",
        ...     missing_rate=0.3,
        ...     imputation_method="mean",
        ...     use_mim=False
        ... )
        >>> metrics = evaluator.evaluate(model, X_test, y_test, seed=42)
        >>> print(metrics)
    """
    
    def __init__(
        self,
        missing_mode: MissingMode,
        missing_rate: MissingRate,
        imputation_method: ImputationMethod,
        use_mim: bool = False
    ):
        """Initialize evaluator.
        
        Args:
            missing_mode: MCAR, MAR, or MNAR
            missing_rate: Proportion of missing values
            imputation_method: mean, knn, iterative, or zero
            use_mim: Whether to use Missing Indicator Method
        """
        self.missing_mode = missing_mode
        self.missing_rate = missing_rate
        self.imputation_method = imputation_method
        self.use_mim = use_mim
        
        # Create generator
        self._generator = self._create_generator(missing_mode)
        
        # Create imputer
        self._imputer = self._create_imputer(imputation_method)
    
    def _create_generator(self, mode: MissingMode) -> MissingGenerator:
        """Create missing pattern generator."""
        generators = {
            "MCAR": MCARGenerator(),
            "MAR": MARGenerator(),
            "MNAR": MNARGenerator()
        }
        return generators[mode]
    
    def _create_imputer(self, method: ImputationMethod) -> Imputer:
        """Create imputation method."""
        imputers = {
            "mean": MeanImputer(),
            "knn": KNNImputer(),
            "iterative": IterativeImputer(),
            "zero": ZeroImputer()
        }
        return imputers[method]
    
    def evaluate(
        self,
        model: torch.nn.Module,
        X: Features,
        y: Labels,
        seed: Seed,
        soh: Labels | None = None
    ) -> Metrics:
        """Evaluate model on data with missing values.
        
        Args:
            model: Trained PyTorch model
            X: Test features
            y: Test labels (ground truth)
            seed: Random seed for reproducibility
            soh: SOH values (required for MAR mode)
            
        Returns:
            Evaluation metrics
        """
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
        # For imputation, we need a reference (train stats would be ideal)
        # Here we use the observed values in X_missing as approximation
        X_imputed = self._imputer.fit_transform(X_missing)
        
        # 3. Add MIM mask if needed
        if self.use_mim:
            # Concatenate imputed features with missing mask
            # mask: True = observed, so 1-mask: 1 = missing
            mask_indicator = (~mask).astype(np.float32)
            X_input = np.concatenate([X_imputed, mask_indicator], axis=1)
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
