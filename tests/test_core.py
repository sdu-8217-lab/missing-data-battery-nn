"""Tests for core module."""

import numpy as np
import pytest

from battery_soh import (
    Seed,
    MissingRate,
    BatchId,
    ModelType,
    MissingMode,
    ImputationMethod,
)
from battery_soh.core.types import validate_missing_rate, validate_seed
from battery_soh.core.constants import (
    N_FEATURES,
    N_FEATURES_WITH_MIM,
    MIN_PARAMS,
    MAX_PARAMS,
    XJTU_BATCHES,
)


class TestTypes:
    """Test type aliases and validation."""
    
    def test_validate_missing_rate_valid(self):
        """Test valid missing rates."""
        assert validate_missing_rate(0.0) == MissingRate(0.0)
        assert validate_missing_rate(0.5) == MissingRate(0.5)
        assert validate_missing_rate(1.0) == MissingRate(1.0)
    
    def test_validate_missing_rate_invalid(self):
        """Test invalid missing rates raise ValueError."""
        with pytest.raises(ValueError):
            validate_missing_rate(-0.1)
        with pytest.raises(ValueError):
            validate_missing_rate(1.1)
    
    def test_validate_seed_valid(self):
        """Test valid seeds."""
        assert validate_seed(0) == Seed(0)
        assert validate_seed(42) == Seed(42)
    
    def test_validate_seed_invalid(self):
        """Test negative seed raises ValueError."""
        with pytest.raises(ValueError):
            validate_seed(-1)


class TestConstants:
    """Test constants."""
    
    def test_feature_dimensions(self):
        """Test feature dimension constants."""
        assert N_FEATURES == 16
        assert N_FEATURES_WITH_MIM == 32
    
    def test_parameter_budget(self):
        """Test parameter budget constants."""
        assert MIN_PARAMS == 16384  # 2^14
        assert MAX_PARAMS == 32768  # 2^15
    
    def test_batch_ids(self):
        """Test valid batch IDs."""
        assert len(XJTU_BATCHES) == 6
        assert "2C" in XJTU_BATCHES
        assert "3C" in XJTU_BATCHES


class TestLiterals:
    """Test literal type values."""
    
    def test_model_types(self):
        """Test model type literals."""
        valid_types: list[ModelType] = ["mlp", "lstm", "cnn"]
        assert len(valid_types) == 3
    
    def test_missing_modes(self):
        """Test missing mode literals."""
        valid_modes: list[MissingMode] = ["MCAR", "MAR", "MNAR"]
        assert len(valid_modes) == 3
    
    def test_imputation_methods(self):
        """Test imputation method literals."""
        valid_methods: list[ImputationMethod] = [
            "mean", "knn", "iterative", "zero"
        ]
        assert len(valid_methods) == 4
