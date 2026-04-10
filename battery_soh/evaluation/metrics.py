"""
Evaluation metrics for SOH prediction.

All metrics follow the convention: lower is better (except R²).
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from battery_soh.core.types import Labels


@dataclass(frozen=True)
class Metrics:
    """Container for evaluation metrics.
    
    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R-squared (coefficient of determination)
        mape: Mean Absolute Percentage Error (optional)
    """
    mae: float
    rmse: float
    r2: float
    mape: float | None = None
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        s = f"MAE: {self.mae:.4f}, RMSE: {self.rmse:.4f}, R²: {self.r2:.4f}"
        if self.mape is not None:
            s += f", MAPE: {self.mape:.2%}"
        return s


def compute_metrics(
    y_true: Labels,
    y_pred: Labels,
    compute_mape: bool = False
) -> Metrics:
    """Compute evaluation metrics.
    
    Args:
        y_true: Ground truth SOH values
        y_pred: Predicted SOH values
        compute_mape: Whether to compute MAPE (requires y_true > 0)
        
    Returns:
        Metrics object with all computed metrics
        
    Example:
        >>> y_true = np.array([0.95, 0.90, 0.85])
        >>> y_pred = np.array([0.94, 0.92, 0.83])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(metrics)
        MAE: 0.0133, RMSE: 0.0153, R²: 0.9850
    """
    # Flatten arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Check lengths
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid samples after removing NaN")
    
    # Compute metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    # Compute MAPE if requested
    mape = None
    if compute_mape:
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100)
    
    return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape)
