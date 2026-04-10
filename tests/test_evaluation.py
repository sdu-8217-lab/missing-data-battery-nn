"""Tests for evaluation module."""

import numpy as np
import pytest

from battery_soh.evaluation.metrics import compute_metrics, Metrics


class TestComputeMetrics:
    """Test metrics computation."""
    
    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        y_pred = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics.mae == pytest.approx(0.0, abs=1e-6)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-6)
        assert metrics.r2 == pytest.approx(1.0, abs=1e-6)
    
    def test_constant_prediction(self):
        """Test metrics with constant predictions."""
        y_true = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        y_pred = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        
        metrics = compute_metrics(y_true, y_pred)
        
        # MAE should be mean of [0.1, 0.0, 0.1] = 0.0667
        assert metrics.mae == pytest.approx(0.0667, abs=1e-4)
        assert metrics.r2 < 1.0
    
    def test_with_nans(self):
        """Test handling of NaN values."""
        y_true = np.array([0.9, np.nan, 0.7], dtype=np.float32)
        y_pred = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        metrics = compute_metrics(y_true, y_pred)
        
        # Should ignore NaN and compute on remaining values
        assert metrics.mae == pytest.approx(0.0, abs=1e-6)
    
    def test_length_mismatch_raises(self):
        """Test mismatched lengths raise ValueError."""
        y_true = np.array([0.9, 0.8])
        y_pred = np.array([0.9, 0.8, 0.7])
        
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred)
    
    def test_all_nan_raises(self):
        """Test all NaN values raise ValueError."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([0.9, 0.8])
        
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred)
    
    def test_mape_computation(self):
        """Test MAPE computation."""
        y_true = np.array([1.0, 0.5], dtype=np.float32)
        y_pred = np.array([0.9, 0.4], dtype=np.float32)
        
        metrics = compute_metrics(y_true, y_pred, compute_mape=True)
        
        assert metrics.mape is not None
        # MAPE = mean(|(1.0-0.9)/1.0|, |(0.5-0.4)/0.5|) * 100
        #      = mean(0.1, 0.2) * 100 = 15.0
        assert metrics.mape == pytest.approx(15.0, abs=1e-6)


class TestMetrics:
    """Test Metrics dataclass."""
    
    def test_str_representation(self):
        """Test string representation."""
        metrics = Metrics(mae=0.05, rmse=0.07, r2=0.95)
        
        s = str(metrics)
        
        assert "MAE" in s
        assert "0.05" in s
        assert "RMSE" in s
        assert "R²" in s
    
    def test_str_with_mape(self):
        """Test string representation with MAPE."""
        metrics = Metrics(mae=0.05, rmse=0.07, r2=0.95, mape=5.0)
        
        s = str(metrics)
        
        assert "MAPE" in s
