#!/usr/bin/env python3
"""
生成统一架构搜索配置（参数上限 65,536 = 2^16）
无下限，只设上限
精简版：每个模型100-150配置
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.models.model_factory import ModelFactory

# 参数上限: 2^16 = 65536
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
        print(f"计算失败 {model_type} {config}: {e}")
        return 0


def generate_mlp_configs():
    """生成MLP配置 - 精简版：~140个配置"""
    configs = []
    
    # ========== 1层配置 (~25个) ==========
    for hidden in [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]:
        for dropout in [0.0, 0.2, 0.5]:
            cfg = {'hidden_layers': [hidden], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg, use_mim=False)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'mlp_l1_{len([c for c in configs if c["layer_type"]==1]):03d}',
                    'layer_type': 1,
                    'hidden_config': f'[{hidden}]',
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    # ========== 2层配置 (~40个) ==========
    configs_2l = [
        [48, 24], [64, 32], [80, 40], [96, 48], [112, 56], [128, 64], [160, 80], [192, 96],
        [64, 48], [96, 72], [128, 96], [160, 120], [192, 144], [224, 168], [256, 192],
    ]
    for h1, h2 in configs_2l:
        for dropout in [0.0, 0.2, 0.5]:
            cfg = {'hidden_layers': [h1, h2], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg, use_mim=False)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'mlp_l2_{len([c for c in configs if c["layer_type"]==2]):03d}',
                    'layer_type': 2,
                    'hidden_config': f'[{h1},{h2}]',
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    # ========== 3层配置 (~30个) ==========
    configs_3l = [
        [64, 32, 16], [80, 40, 20], [96, 48, 24], [112, 56, 28], [128, 64, 32], [160, 80, 40], [192, 96, 48],
        [96, 64, 32], [128, 80, 48], [160, 96, 64], [192, 112, 80],
    ]
    for h1, h2, h3 in configs_3l:
        for dropout in [0.0, 0.2, 0.5]:
            cfg = {'hidden_layers': [h1, h2, h3], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg, use_mim=False)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'mlp_l3_{len([c for c in configs if c["layer_type"]==3]):03d}',
                    'layer_type': 3,
                    'hidden_config': f'[{h1},{h2},{h3}]',
                    'dropout': dropout,
                    'estimated_params': params,
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                })
    
    # ========== 4层配置 (~35个) ==========
    configs_4l = [
        [96, 48, 24, 12], [128, 64, 32, 16], [160, 80, 40, 20], [192, 96, 48, 24], [256, 128, 64, 32],
        [128, 80, 48, 32], [160, 96, 64, 40], [192, 128, 80, 48],
    ]
    for h1, h2, h3, h4 in configs_4l:
        for dropout in [0.0, 0.2, 0.5]:
            cfg = {'hidden_layers': [h1, h2, h3, h4], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg, use_mim=False)
            if params <= PARAM_MAX:
                configs.append({
                    'config_id': f'mlp_l4_{len([c for c in configs if c["layer_type"]==4]):03d}',
                    'layer_type': 4,
                    'hidden_config': f'[{h1},{h2},{h3},{h4}]',
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


def generate_lstm_configs():
    """生成LSTM配置，~120个配置"""
    configs = []
    
    hidden_vals = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
    
    for num_layers in [1, 2, 3, 4]:
        for hidden in hidden_vals:
            for dropout in [0.0, 0.15, 0.3]:
                actual_dropout = dropout if num_layers > 1 else 0.0
                cfg = {
                    'hidden_size': hidden,
                    'num_layers': num_layers,
                    'dropout': actual_dropout
                }
                params = calc_exact_params('lstm', cfg, use_mim=False)
                if params <= PARAM_MAX:
                    configs.append({
                        'config_id': f'lstm_l{num_layers}_{len([c for c in configs if c["layer_type"]==num_layers]):03d}',
                        'layer_type': num_layers,
                        'hidden_config': f'h={hidden},l={num_layers}',
                        'dropout': actual_dropout,
                        'estimated_params': params,
                        'actual_params': '',
                        'mae': '',
                        'rmse': '',
                        'r2': '',
                        'training_time': '',
                        'status': 'pending'
                    })
    
    return configs


def generate_gru_configs():
    """生成GRU配置，~120个配置"""
    configs = []
    
    hidden_vals = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112]
    
    for num_layers in [1, 2, 3, 4]:
        for hidden in hidden_vals:
            for dropout in [0.0, 0.15, 0.3]:
                actual_dropout = dropout if num_layers > 1 else 0.0
                cfg = {
                    'hidden_size': hidden,
                    'num_layers': num_layers,
                    'dropout': actual_dropout
                }
                params = calc_exact_params('gru', cfg, use_mim=False)
                if params <= PARAM_MAX:
                    configs.append({
                        'config_id': f'gru_l{num_layers}_{len([c for c in configs if c["layer_type"]==num_layers]):03d}',
                        'layer_type': num_layers,
                        'hidden_config': f'h={hidden},l={num_layers}',
                        'dropout': actual_dropout,
                        'estimated_params': params,
                        'actual_params': '',
                        'mae': '',
                        'rmse': '',
                        'r2': '',
                        'training_time': '',
                        'status': 'pending'
                    })
    
    return configs


def generate_cnn_configs():
    """生成CNN1D配置，~150个配置"""
    configs = []
    kernels = [2, 3, 4, 5]
    
    # 1层配置 (~20个)
    ch1_vals = [16, 32, 64, 96, 128, 192, 256, 384]
    for ch1 in ch1_vals:
        for kernel in kernels[:2]:  # 只取前2个kernel
            for dropout in [0.0, 0.2, 0.5]:
                cfg = {'channels': [ch1], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg, use_mim=False)
                if params <= PARAM_MAX:
                    configs.append({
                        'config_id': f'cnn_l1_{len([c for c in configs if c["layer_type"]==1]):03d}',
                        'layer_type': 1,
                        'hidden_config': f'[{ch1}]',
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
    
    # 2层配置 (~50个)
    ch2_configs = [
        [32, 16], [48, 24], [64, 32], [80, 40], [96, 48], [128, 64], [160, 80], [192, 96], [256, 128],
        [48, 32], [64, 48], [80, 56], [96, 72], [128, 96], [160, 120],
    ]
    for ch1, ch2 in ch2_configs:
        for kernel in [3, 5]:
            for dropout in [0.0, 0.2, 0.5]:
                cfg = {'channels': [ch1, ch2], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg, use_mim=False)
                if params <= PARAM_MAX:
                    configs.append({
                        'config_id': f'cnn_l2_{len([c for c in configs if c["layer_type"]==2]):03d}',
                        'layer_type': 2,
                        'hidden_config': f'[{ch1},{ch2}]',
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
    
    # 3层配置 (~45个)
    ch3_configs = [
        [64, 32, 16], [80, 40, 20], [96, 48, 24], [128, 64, 32], [160, 80, 40],
        [96, 64, 32], [128, 80, 48], [160, 96, 64], [192, 112, 80],
    ]
    for ch1, ch2, ch3 in ch3_configs:
        for kernel in [3, 4, 5]:
            for dropout in [0.0, 0.2, 0.5]:
                cfg = {'channels': [ch1, ch2, ch3], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg, use_mim=False)
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
    
    # 4层配置 (~35个)
    ch4_configs = [
        [96, 48, 24, 12], [128, 64, 32, 16], [160, 80, 40, 20], [192, 96, 48, 24],
        [128, 80, 48, 32], [160, 96, 64, 40],
    ]
    for ch1, ch2, ch3, ch4 in ch4_configs:
        for kernel in [4, 5]:
            for dropout in [0.0, 0.2, 0.5]:
                cfg = {'channels': [ch1, ch2, ch3, ch4], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg, use_mim=False)
                if params <= PARAM_MAX:
                    configs.append({
                        'config_id': f'cnn_l4_{len([c for c in configs if c["layer_type"]==4]):03d}',
                        'layer_type': 4,
                        'hidden_config': f'[{ch1},{ch2},{ch3},{ch4}]',
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


def main():
    """生成所有配置文件"""
    output_dir = Path('./configs')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print(f"生成架构搜索配置文件 (参数上限 {PARAM_MAX:,} = 2^16)")
    print("=" * 70)
    
    # 生成各模型配置（不含XGBoost）
    generators = {
        'mlp': generate_mlp_configs,
        'lstm': generate_lstm_configs,
        'gru': generate_gru_configs,
        'cnn1d': generate_cnn_configs,
    }
    
    total_configs = 0
    for model_name, generator in generators.items():
        print(f"\n生成 {model_name.upper()} 配置...")
        configs = generator()
        df = pd.DataFrame(configs)
        
        output_file = output_dir / f'{model_name}_configs.csv'
        df.to_csv(output_file, index=False)
        
        total_configs += len(configs)
        print(f"  生成 {len(configs)} 个配置")
        print(f"  保存到: {output_file}")
        
        # 统计各层数量
        for layer in [1, 2, 3, 4]:
            count = len([c for c in configs if c['layer_type'] == layer])
            if count > 0:
                print(f"    层{layer}: {count}个")
    
    print("\n" + "=" * 70)
    print("配置生成完成!")
    print(f"总配置数: {total_configs}")
    print(f"配置文件目录: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
