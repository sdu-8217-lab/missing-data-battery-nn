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
from battery_soh.core.constants import (
    N_FEATURES,
    N_FEATURES_WITH_MIM,
    MIN_PARAMS,
    MAX_PARAMS,
    XJTU_BATCHES,
    MISSING_MODES,
    IMPUTATION_METHODS,
    MODEL_TYPES,
)


class TestTypes:
    """Test type aliases."""

    def test_missing_rate_is_float(self):
        """MissingRate 是 float 的类型别名."""
        assert MissingRate is float

    def test_seed_is_int(self):
        """Seed 是 int 的类型别名."""
        assert Seed is int

    def test_batch_id_values(self):
        """BatchId 允许的取值."""
        valid: list[BatchId] = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
        assert len(valid) == 6


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

    def test_missing_modes(self):
        """Test missing mode constants."""
        assert len(MISSING_MODES) == 3
        assert "MCAR" in MISSING_MODES
        assert "MAR" in MISSING_MODES
        assert "MNAR" in MISSING_MODES


class TestLiterals:
    """Test literal type values."""

    def test_model_types(self):
        """Test model type literals."""
        valid_types: list[ModelType] = ["mlp", "lstm", "cnn"]
        assert len(valid_types) == 3
        assert all(t in MODEL_TYPES for t in valid_types)

    def test_missing_modes_literal(self):
        """Test missing mode literals."""
        valid_modes: list[MissingMode] = ["MCAR", "MAR", "MNAR"]
        assert len(valid_modes) == 3
        assert all(m in MISSING_MODES for m in valid_modes)

    def test_imputation_methods(self):
        """Test imputation method literals."""
        valid_methods: list[ImputationMethod] = [
            "mean", "knn", "iterative", "zero"
        ]
        assert len(valid_methods) == 4
        assert all(m in IMPUTATION_METHODS for m in valid_methods)
