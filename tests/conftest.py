"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import torch

from battery_soh import Seed


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample feature matrix."""
    return np.random.randn(100, 16).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample labels."""
    return np.linspace(1.0, 0.7, 100).astype(np.float32)


@pytest.fixture
def sample_battery_ids() -> np.ndarray:
    """Create sample battery IDs."""
    return np.array(["bat_01"] * 50 + ["bat_02"] * 50)


@pytest.fixture
def sample_missing_mask() -> np.ndarray:
    """Create sample missing mask."""
    np.random.seed(42)
    return np.random.rand(100, 16) > 0.3


@pytest.fixture
def rng() -> np.random.Generator:
    """Create random number generator with fixed seed."""
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for all tests."""
    from battery_soh.utils import set_seed
    set_seed(Seed(42))
    yield
