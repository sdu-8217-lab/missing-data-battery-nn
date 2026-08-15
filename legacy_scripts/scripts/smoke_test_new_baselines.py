"""新基线模型集成 smoke test"""
import sys
sys.path.insert(0, '/home/chen/research/missing-data-battery-nn')

import torch
from src.models.model_factory import ModelFactory


def test_model(name, model_type, use_mim, input_dim=16, seq_len=5, **kwargs):
    print(f"\nTesting {name} (model_type={model_type}, use_mim={use_mim})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    actual_input_dim = input_dim * 2 if use_mim else input_dim
    if model_type in ['mlp', 'lstm', 'gru', 'cnn1d', 'transformer', 'itransformer', 'saits', 'neural_cde']:
        x = torch.randn(4, seq_len, actual_input_dim).to(device)
    elif model_type == 'grin':
        x = torch.randn(4, seq_len, actual_input_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = ModelFactory.create_model(
        model_type=model_type,
        input_dim=input_dim,
        use_mim=use_mim,
        device=device,
        seq_len=seq_len,
        **kwargs
    )
    
    out = model.predict(x) if hasattr(model, 'predict') else model(x)
    print(f"  Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == torch.Size([4]), f"Expected [4], got {out.shape}"
    print(f"  ✅ {name} passed")
    return True


def main():
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    test_cases = [
        ('iTransformer-MIM', 'itransformer', True, {}),
        ('iTransformer-MultiMR', 'itransformer', False, {}),
        ('SAITS-MIM', 'saits', True, {}),
        ('SAITS-MultiMR', 'saits', False, {}),
        ('NeuralCDE-MIM', 'neural_cde', True, {}),
        ('NeuralCDE-MultiMR', 'neural_cde', False, {}),
        ('GRIN-MIM', 'grin', True, {'group_ids': [0]*8 + [1]*8}),
        ('GRIN-MultiMR', 'grin', False, {'group_ids': [0]*8 + [1]*8}),
    ]
    
    passed = 0
    failed = 0
    for name, mtype, use_mim, kwargs in test_cases:
        try:
            test_model(name, mtype, use_mim, **kwargs)
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
