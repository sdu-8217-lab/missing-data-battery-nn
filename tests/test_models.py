"""Tests for models module."""

import pytest
import torch

from battery_soh.models import create_model, count_parameters, check_parameter_budget
from battery_soh.core.constants import MIN_PARAMS, MAX_PARAMS


class TestModelFactory:
    """Test model factory functions."""
    
    @pytest.mark.parametrize("model_type", ["mlp", "lstm", "cnn"])
    @pytest.mark.parametrize("use_mim", [False, True])
    def test_create_model(self, model_type, use_mim):
        """Test creating all model types with/without MIM."""
        model = create_model(model_type, use_mim=use_mim)
        assert model is not None
        
        # Check input dimension
        expected_dim = 32 if use_mim else 16
        
        # Test forward pass
        if model_type == "mlp":
            x = torch.randn(10, expected_dim)
        else:  # lstm, cnn
            x = torch.randn(10, 1, expected_dim)
        
        y = model(x)
        assert y.shape == (10,)
    
    def test_create_model_invalid_type(self):
        """Test creating model with invalid type raises ValueError."""
        with pytest.raises(ValueError):
            create_model("invalid_model")
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = create_model("mlp", use_mim=False)
        n_params = count_parameters(model)
        assert isinstance(n_params, int)
        assert n_params > 0
    
    @pytest.mark.parametrize("model_type", ["mlp", "lstm", "cnn"])
    @pytest.mark.parametrize("use_mim", [False, True])
    def test_parameter_budget(self, model_type, use_mim):
        """Test all models are within parameter budget."""
        model = create_model(model_type, use_mim=use_mim)
        n_params = count_parameters(model)
        
        assert MIN_PARAMS <= n_params <= MAX_PARAMS, (
            f"{model_type} (use_mim={use_mim}) has {n_params} params, "
            f"but budget is [{MIN_PARAMS}, {MAX_PARAMS}]"
        )
    
    def test_check_parameter_budget_verbose(self, capsys):
        """Test parameter budget check with verbose output."""
        model = create_model("mlp", use_mim=False)
        result = check_parameter_budget(model, verbose=True)
        
        captured = capsys.readouterr()
        assert "MLP" in captured.out
        assert "parameters" in captured.out
        assert result is True


class TestModelArchitectures:
    """Test individual model architectures."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from battery_soh.models.architectures.mlp import MLP
        
        model = MLP(input_dim=16, hidden_dims=[64, 32])
        x = torch.randn(5, 16)
        y = model(x)
        
        assert y.shape == (5,)
    
    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        from battery_soh.models.architectures.lstm import LSTM
        
        model = LSTM(input_dim=16, hidden_size=32, num_layers=2)
        x = torch.randn(5, 10, 16)  # batch, seq, features
        y = model(x)
        
        assert y.shape == (5,)
    
    def test_cnn_forward(self):
        """Test CNN forward pass."""
        from battery_soh.models.architectures.cnn import CNN1D
        
        model = CNN1D(input_dim=16, channels=[32, 64])
        x = torch.randn(5, 10, 16)  # batch, seq, features
        y = model(x)
        
        assert y.shape == (5,)
