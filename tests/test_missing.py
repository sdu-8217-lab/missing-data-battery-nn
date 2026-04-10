"""Tests for missing data generators and imputers."""

import numpy as np
import pytest

from battery_soh import Seed, MissingRate
from battery_soh.missing.generators import MCARGenerator, MARGenerator, MNARGenerator
from battery_soh.missing.imputers import MeanImputer, ZeroImputer


class TestMCARGenerator:
    """Test MCAR missing pattern generator."""
    
    def test_generate_shape(self, sample_features):
        """Test output shapes."""
        gen = MCARGenerator()
        X_missing, mask = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42)
        )
        
        assert X_missing.shape == sample_features.shape
        assert mask.shape == sample_features.shape
        assert mask.dtype == bool
    
    def test_missing_rate_approximate(self, sample_features):
        """Test actual missing rate is close to target."""
        gen = MCARGenerator()
        target_rate = 0.3
        
        X_missing, mask = gen.generate(
            sample_features,
            MissingRate(target_rate),
            Seed(42)
        )
        
        actual_rate = (~mask).sum() / mask.size
        assert abs(actual_rate - target_rate) < 0.05  # Within 5%
    
    def test_reproducibility(self, sample_features):
        """Test same seed produces same result."""
        gen = MCARGenerator()
        
        _, mask1 = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42)
        )
        _, mask2 = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42)
        )
        
        assert np.array_equal(mask1, mask2)
    
    def test_invalid_missing_rate(self, sample_features):
        """Test invalid missing rate raises error."""
        gen = MCARGenerator()
        
        with pytest.raises(ValueError):
            gen.generate(sample_features, MissingRate(-0.1), Seed(42))
        
        with pytest.raises(ValueError):
            gen.generate(sample_features, MissingRate(1.1), Seed(42))
    
    def test_name(self):
        """Test generator name."""
        gen = MCARGenerator()
        assert gen.name == "MCAR"


class TestMARGenerator:
    """Test MAR missing pattern generator."""
    
    def test_generate_with_soh(self, sample_features, sample_labels):
        """Test MAR generation with SOH values."""
        gen = MARGenerator()
        
        X_missing, mask = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42),
            soh=sample_labels
        )
        
        assert X_missing.shape == sample_features.shape
        assert mask.shape == sample_features.shape
    
    def test_generate_without_soh(self, sample_features):
        """Test MAR generation without SOH (uses default)."""
        gen = MARGenerator()
        
        X_missing, mask = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42)
        )
        
        assert X_missing.shape == sample_features.shape
    
    def test_name(self):
        """Test generator name."""
        gen = MARGenerator()
        assert gen.name == "MAR"


class TestMNARGenerator:
    """Test MNAR missing pattern generator."""
    
    def test_generate(self, sample_features):
        """Test MNAR generation."""
        gen = MNARGenerator()
        
        X_missing, mask = gen.generate(
            sample_features,
            MissingRate(0.3),
            Seed(42)
        )
        
        assert X_missing.shape == sample_features.shape
        assert mask.shape == sample_features.shape
    
    def test_name(self):
        """Test generator name."""
        gen = MNARGenerator()
        assert gen.name == "MNAR"


class TestMeanImputer:
    """Test mean imputation method."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        # Create data with known means
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype=np.float32)
        
        imputer = MeanImputer()
        imputer.fit(X)
        
        # Create missing data
        X_missing = np.array([
            [np.nan, 2.0],
            [3.0, np.nan]
        ], dtype=np.float32)
        
        X_filled = imputer.transform(X_missing)
        
        # Check filled values (means are 3.0 and 4.0)
        assert X_filled[0, 0] == pytest.approx(3.0)
        assert X_filled[1, 1] == pytest.approx(4.0)
        assert X_filled[0, 1] == 2.0  # Unchanged
        assert X_filled[1, 0] == 3.0  # Unchanged
    
    def test_fit_transform_with_mask(self):
        """Test fit_transform with explicit mask."""
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype=np.float32)
        
        mask = np.array([
            [True, True],
            [True, True],
            [True, True]
        ])
        
        imputer = MeanImputer()
        X_filled = imputer.fit_transform(X, mask)
        
        # Should be unchanged
        assert np.array_equal(X_filled, X)
    
    def test_name(self):
        """Test imputer name."""
        imputer = MeanImputer()
        assert imputer.name == "mean"
    
    def test_transform_before_fit_raises(self):
        """Test transform before fit raises RuntimeError."""
        imputer = MeanImputer()
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        
        with pytest.raises(RuntimeError):
            imputer.transform(X)


class TestZeroImputer:
    """Test zero imputation method."""
    
    def test_transform(self):
        """Test zero imputation."""
        X_missing = np.array([
            [np.nan, 2.0],
            [3.0, np.nan]
        ], dtype=np.float32)
        
        imputer = ZeroImputer()
        X_filled = imputer.fit_transform(X_missing)
        
        assert X_filled[0, 0] == 0.0
        assert X_filled[1, 1] == 0.0
        assert X_filled[0, 1] == 2.0
        assert X_filled[1, 0] == 3.0
    
    def test_name(self):
        """Test imputer name."""
        imputer = ZeroImputer()
        assert imputer.name == "zero"
