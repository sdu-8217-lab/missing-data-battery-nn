#!/usr/bin/env python3
"""
生成第三轮精细搜索配置 (configs_v3)
基于configs_v2结果进行精细化加密搜索
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.models.model_factory import ModelFactory

PARAM_MAX = 65536


def calc_exact_params(model_type, config, use_mim=False):
    """精确计算参数量"""
    try:
        model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=16,
            use_mim=use_mim,
            device='cpu',
            **config
        )
        actual_model = model.model if hasattr(model, 'model') else model
        return sum(p.numel() for p in actual_model.parameters())
    except Exception as e:
        return 0


def generate_cnn1d_v3_configs():
    """
    CNN1D第三轮精细搜索
    基于最佳: [160,80,40] MAE=0.0061 和 [64,24] MAE=0.0063(7K params)
    """
    configs = []
    
    # ========== 3层精细搜索 (围绕 [160,80,40]) ==========
    # 第一层: 144, 160, 176 (±16)
    # 第二层: 72, 80, 88 (±8)  
    # 第三层: 36, 40, 44, 48 (±8)
    layer1_options = [144, 160, 176]
    layer2_options = [72, 80, 88]
    layer3_options = [36, 40, 44, 48]
    kernels_3l = [3, 4, 5]
    dropouts_3l = [0.1, 0.15, 0.2, 0.25]
    
    for ch1 in layer1_options:
        for ch2 in layer2_options:
            for ch3 in layer3_options:
                if ch2 >= ch1 or ch3 >= ch2:  # 确保递减
                    continue
                for kernel in kernels_3l:
                    for dropout in dropouts_3l:
                        cfg = {'channels': [ch1, ch2, ch3], 'kernel_size': kernel, 'dropout': dropout}
                        params = calc_exact_params('cnn1d', cfg)
                        if params <= PARAM_MAX:
                            configs.append({
                                'config_id': f'cnn_l3_{len([c for c in configs if c["layer_type"]==3]):03d}',
                                'layer_type': 3,
                                'hidden_config': f'[{ch1},{ch2},{ch3}]',
                                'kernel': kernel,
                                'dropout': dropout,
                                'estimated_params': params,
                                'actual_params': '',
                                'mae': '',
                                'rmse': '',
                                'r2': '',
                                'training_time': '',
                                'status': 'pending'
                            })
    
    # ========== 2层高效区扩展 (围绕 [64,24]) ==========
    # [56,24], [64,28], [64,32], [72,32], [80,40]
    efficient_2l = [
        [56, 24], [60, 28], [64, 24], [64, 28], [64, 32],
        [68, 32], [72, 32], [76, 36], [80, 40], [80, 48]
    ]
    kernels_2l = [4, 5]
    dropouts_2l = [0.1, 0.15, 0.2]
    
    for channels in efficient_2l:
        for kernel in kernels_2l:
            for dropout in dropouts_2l:
                cfg = {'channels': channels, 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg)
                if params <= PARAM_MAX and params < 20000:  # 保持高效
                    configs.append({
                        'config_id': f'cnn_l2_{len([c for c in configs if c["layer_type"]==2]):03d}',
                        'layer_type': 2,
                        'hidden_config': f'[{channels[0]},{channels[1]}]',
                        'kernel': kernel,
                        'dropout': dropout,
                        'estimated_params': params,
                        'actual_params': '',
                        'mae': '',
                        'rmse': '',
                        'r2': '',
                        'training_time': '',
                        'status': 'pending'
                    })
    
    return configs


def generate_lstm_v3_configs():
    """
    LSTM第三轮精细搜索
    基于最佳: h=56, l=2, MAE=0.0069
    """
    configs = []
    
    # 围绕 h=56 精细搜索: 48, 52, 56, 60, 64
    hidden_sizes = [48, 52, 56, 60, 64]
    dropouts = [0.2, 0.25, 0.3]
    
    for hidden in hidden_sizes:
        for dropout in dropouts:
            cfg = {
                'hidden_size': hidden,
                'num_layers': 2,
                'dropout': dropout
            }
            params = calc_exact_params('lstm', cfg)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'lstm_l2_{len(configs):03d}',
                    'layer_type': 2,
                    'hidden_config': f'h={hidden},l=2',
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    return configs


def generate_gru_v3_configs():
    """
    GRU第三轮精细搜索
    基于最佳: h=72, l=2, MAE=0.0085
    """
    configs = []
    
    # 围绕 h=72 精细搜索: 64, 68, 72, 76, 80
    hidden_sizes = [64, 68, 72, 76, 80]
    dropouts = [0.2, 0.25, 0.3]
    
    for hidden in hidden_sizes:
        for dropout in dropouts:
            cfg = {
                'hidden_size': hidden,
                'num_layers': 2,
                'dropout': dropout
            }
            params = calc_exact_params('gru', cfg)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'gru_l2_{len(configs):03d}',
                    'layer_type': 2,
                    'hidden_config': f'h={hidden},l=2',
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    return configs


def generate_mlp_v3_configs():
    """
    MLP第三轮精细搜索
    基于最佳: [224,112,56,28] MAE=0.0097
    验证4层架构
    """
    configs = []
    
    # 4层精细搜索
    layer4_configs = [
        [192, 96, 48, 24],
        [224, 112, 56, 28],
        [256, 128, 64, 32],
    ]
    dropouts = [0.15, 0.2, 0.25]
    
    for layers in layer4_configs:
        for dropout in dropouts:
            cfg = {'hidden_layers': layers, 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'mlp_l4_{len(configs):03d}',
                    'layer_type': 4,
                    'hidden_config': str(layers),
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    return configs


def main():
    """生成第三轮精细搜索配置"""
    output_dir = Path('./configs_v3')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("生成第三轮精细搜索配置 (configs_v3)")
    print("=" * 70)
    
    generators = {
        'mlp': generate_mlp_v3_configs,
        'lstm': generate_lstm_v3_configs,
        'gru': generate_gru_v3_configs,
        'cnn1d': generate_cnn1d_v3_configs,
    }
    
    total_configs = 0
    for model_name, generator in generators.items():
        print(f"\n生成 {model_name.upper()} v3 配置...")
        configs = generator()
        
        if not configs:
            print(f"  警告: {model_name} 没有生成配置")
            continue
            
        df = pd.DataFrame(configs)
        output_file = output_dir / f'{model_name}_configs.csv'
        df.to_csv(output_file, index=False)
        
        total_configs += len(configs)
        print(f"  生成 {len(configs)} 个配置")
        print(f"  保存到: {output_file}")
        
        # 统计
        for layer in [2, 3, 4]:
            count = len([c for c in configs if c['layer_type'] == layer])
            if count > 0:
                sub = [c for c in configs if c['layer_type'] == layer]
                params = [c['estimated_params'] for c in sub]
                print(f"    层{layer}: {count:3d}个, 参数: {min(params):,}-{max(params):,}")
    
    print("\n" + "=" * 70)
    print("第三轮配置生成完成!")
    print(f"总配置数: {total_configs}")
    print(f"配置文件目录: {output_dir}")
    print("\n搜索策略:")
    print("  - CNN1D: 3层精细加密 + 2层高效扩展")
    print("  - LSTM:  h=48-64 精细搜索")
    print("  - GRU:   h=64-80 精细搜索")
    print("  - MLP:   4层架构验证")
    print("=" * 70)


if __name__ == '__main__':
    main()
