# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""楠岃瘉鍙傛暟閲忚绠楀噯纭€?""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.model_factory import ModelFactory


def count_params_pytorch(model):
    """浣跨敤PyTorch绮剧‘璁＄畻鍙傛暟閲?""
    if hasattr(model, 'model'):
        model = model.model
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params
    }


def compare_param_counts():
    """瀵规瘮浼扮畻鍊煎拰瀹為檯鍊?""
    
    test_configs = [
        # MLP configs
        ('mlp', {'hidden_layers': [64], 'dropout': 0.0}, 'mlp_v1'),
        ('mlp', {'hidden_layers': [256, 128, 64], 'dropout': 0.2}, 'mlp_v8'),
        
        # LSTM configs
        ('lstm', {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}, 'lstm_v3'),
        ('lstm', {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.5}, 'lstm_v7'),
        
        # GRU configs
        ('gru', {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}, 'gru_v4'),
        
        # CNN1D configs
        ('cnn1d', {'channels': [32, 16], 'kernel_size': 3, 'dropout': 0.2}, 'cnn1d_v3'),
        ('cnn1d', {'channels': [128, 64], 'kernel_size': 3, 'dropout': 0.2}, 'cnn1d_v5'),
    ]
    
    from src.experiments.architecture_search import ArchitectureSearchSpace
    
    print('=' * 80)
    print('鍙傛暟閲忚绠楀噯纭€ч獙璇?)
    print('=' * 80)
    print(f"{'Model':<12} {'Config':<12} {'Estimated':<12} {'Actual':<12} {'Error':<10}")
    print('-' * 80)
    
    for model_type, config, name in test_configs:
        # 鍒涘缓妯″瀷
        model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=16,
            use_mim=True,  # MIM妯″紡锛岃緭鍏ョ淮搴?2
            device='cpu',
            **config
        )
        
        # 浼扮畻鍊?
        estimated = ArchitectureSearchSpace.estimate_params(model_type, config)
        
        # 瀹為檯鍊?
        actual_dict = count_params_pytorch(model)
        actual = actual_dict['trainable']
        
        # 璇樊
        error_pct = abs(estimated - actual) / actual * 100 if actual > 0 else 0
        
        print(f"{model_type:<12} {name:<12} {estimated:<12,} {actual:<12,} {error_pct:>8.1f}%")
    
    print('=' * 80)


def detailed_breakdown():
    """璇︾粏鍒嗚ВCNN1D-v5鐨勫弬鏁伴噺"""
    print('\n' + '=' * 80)
    print('CNN1D-v5 鍙傛暟閲忚缁嗗垎瑙?)
    print('=' * 80)
    
    model = ModelFactory.create_model(
        model_type='cnn1d',
        input_dim=16,
        use_mim=True,
        device='cpu',
        channels=[128, 64],
        kernel_size=3,
        dropout=0.2
    )
    
    if hasattr(model, 'model'):
        model = model.model
    
    print(f"\n妯″瀷缁撴瀯:")
    print(model)
    
    print(f"\n鍚勫眰鍙傛暟閲?")
    total = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        print(f"  {name:<30} {list(param.shape):<20} {num:>10,}")
    
    print(f"\n鎬昏: {total:,} 鍙傛暟")


if __name__ == '__main__':
    compare_param_counts()
    detailed_breakdown()

