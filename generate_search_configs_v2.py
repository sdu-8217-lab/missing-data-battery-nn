#!/usr/bin/env python3
"""
生成优化后的架构搜索配置（基于上一轮结果）
- 围绕最佳点 [192,64,32] 加密搜索
- 扩展高效区 2层配置
- 验证4层潜力
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


def generate_mlp_optimized_configs():
    """
    基于上一轮结果优化的MLP配置
    上一轮最佳: [192,64,32] d0.2, MAE=0.0157, 20,801 params
    """
    configs = []
    
    # ========== 策略1: 围绕最佳点 [192,64,32] 精细搜索 ==========
    base_first = [160, 176, 192, 208, 224]  # 192附近
    base_second = [48, 56, 64, 72, 80]      # 64附近
    base_third = [24, 28, 32, 36, 40]       # 32附近
    
    # 3层精细搜索 - 固定其他两层，变化一层
    # 变化第一层
    for h1 in base_first:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [h1, 64, 32], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t1'))
    
    # 变化第二层
    for h2 in base_second:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [192, h2, 32], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t2'))
    
    # 变化第三层
    for h3 in base_third:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [192, 64, h3], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t3'))
    
    # ========== 策略2: 高效区2层扩展 ==========
    # 上一轮: [64,32] 表现优异，4,225 params, MAE≈0.0163
    efficient_first = [48, 56, 64, 72, 80, 96, 112]
    efficient_second = [24, 28, 32, 36, 40, 48]
    
    for h1 in efficient_first:
        for h2 in efficient_second:
            if h2 >= h1:  # 保证递减
                continue
            for dropout in [0.1, 0.2, 0.3, 0.4]:
                cfg = {'hidden_layers': [h1, h2], 'dropout': dropout}
                params = calc_exact_params('mlp', cfg)
                if params <= PARAM_MAX and params < 15000:  # 保持高效
                    configs.append(create_config_entry(configs, 2, cfg, params, 'efficient'))
    
    # ========== 策略3: 4层验证 ==========
    # 探索更深网络是否能超越3层
    layer4_configs = [
        [128, 64, 32, 16],
        [160, 80, 40, 20],
        [192, 96, 48, 24],
        [192, 64, 32, 16],
        [224, 112, 56, 28],
        [160, 64, 32, 16],
        [192, 80, 40, 20],
    ]
    
    for layers in layer4_configs:
        for dropout in [0.1, 0.2, 0.3]:
            cfg = {'hidden_layers': layers, 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 4, cfg, params, 'deep'))
    
    # ========== 策略4: 1层大宽度探索 ==========
    # 验证简单模型的极限
    for hidden in [256, 320, 384, 448, 512, 640, 768]:
        for dropout in [0.0, 0.1, 0.2]:
            cfg = {'hidden_layers': [hidden], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 1, cfg, params, 'wide'))
    
    return configs


def create_config_entry(existing_configs, layer_type, cfg, params, tag):
    """创建配置条目"""
    layer_count = len([c for c in existing_configs if c['layer_type'] == layer_type])
    return {
        'config_id': f'mlp_l{layer_type}_{tag}_{layer_count:03d}',
        'layer_type': layer_type,
        'hidden_config': str(cfg['hidden_layers']),
        'dropout': cfg['dropout'],
        'estimated_params': params,
        'actual_params': '',
        'mae': '',
        'rmse': '',
        'r2': '',
        'training_time': '',
        'status': 'pending'
    }


def generate_lstm_optimized_configs():
    """基于MLP结果推测LSTM优化方向（更深但更窄可能更好）"""
    configs = []
    
    # 围绕可能的最佳点加密
    # 上一轮最佳倾向: h=56-64, l=2-3
    hidden_range = [40, 48, 56, 64, 72, 80, 88]
    
    for num_layers in [2, 3]:
        for hidden in hidden_range:
            for dropout in [0.1, 0.15, 0.2, 0.25, 0.3]:
                actual_dropout = dropout if num_layers > 1 else 0.0
                cfg = {
                    'hidden_size': hidden,
                    'num_layers': num_layers,
                    'dropout': actual_dropout
                }
                params = calc_exact_params('lstm', cfg)
                if params <= PARAM_MAX:
                    layer_count = len([c for c in configs if c['layer_type'] == num_layers])
                    configs.append({
                        'config_id': f'lstm_l{num_layers}_{layer_count:03d}',
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


def generate_gru_optimized_configs():
    """GRU优化配置"""
    configs = []
    
    hidden_range = [48, 56, 64, 72, 80, 88, 96, 104]
    
    for num_layers in [2, 3]:
        for hidden in hidden_range:
            for dropout in [0.1, 0.15, 0.2, 0.25, 0.3]:
                actual_dropout = dropout if num_layers > 1 else 0.0
                cfg = {
                    'hidden_size': hidden,
                    'num_layers': num_layers,
                    'dropout': actual_dropout
                }
                params = calc_exact_params('gru', cfg)
                if params <= PARAM_MAX:
                    layer_count = len([c for c in configs if c['layer_type'] == num_layers])
                    configs.append({
                        'config_id': f'gru_l{num_layers}_{layer_count:03d}',
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


def generate_cnn_optimized_configs():
    """CNN1D优化配置 - 类似MLP的扩张策略"""
    configs = []
    
    # 2层高效区扩展
    ch2_configs = [
        [48, 32], [56, 32], [64, 32], [72, 32], [80, 40],
        [64, 24], [64, 28], [64, 32], [64, 40], [64, 48],
        [80, 48], [96, 48], [96, 64], [112, 64], [128, 64],
    ]
    
    for ch1, ch2 in ch2_configs:
        for kernel in [2, 3, 4, 5]:
            for dropout in [0.0, 0.1, 0.2, 0.3]:
                cfg = {'channels': [ch1, ch2], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg)
                if params <= PARAM_MAX:
                    layer_count = len([c for c in configs if c['layer_type'] == 2])
                    configs.append({
                        'config_id': f'cnn_l2_{layer_count:03d}',
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
    
    # 3层精细搜索
    ch3_configs = [
        [128, 64, 32], [160, 80, 40], [192, 96, 48],
        [128, 80, 48], [160, 96, 64], [192, 112, 80],
    ]
    
    for ch1, ch2, ch3 in ch3_configs:
        for kernel in [3, 4, 5]:
            for dropout in [0.0, 0.1, 0.2]:
                cfg = {'channels': [ch1, ch2, ch3], 'kernel_size': kernel, 'dropout': dropout}
                params = calc_exact_params('cnn1d', cfg)
                if params <= PARAM_MAX:
                    layer_count = len([c for c in configs if c['layer_type'] == 3])
                    configs.append({
                        'config_id': f'cnn_l3_{layer_count:03d}',
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
    
    return configs


def main():
    """生成优化后的配置文件"""
    output_dir = Path('./configs_v2')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("生成优化后的架构搜索配置 (基于上一轮结果)")
    print("=" * 70)
    
    generators = {
        'mlp': generate_mlp_optimized_configs,
        'lstm': generate_lstm_optimized_configs,
        'gru': generate_gru_optimized_configs,
        'cnn1d': generate_cnn_optimized_configs,
    }
    
    total_configs = 0
    for model_name, generator in generators.items():
        print(f"\n生成 {model_name.upper()} 优化配置...")
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
        for layer in [1, 2, 3, 4]:
            count = len([c for c in configs if c['layer_type'] == layer])
            if count > 0:
                sub = [c for c in configs if c['layer_type'] == layer]
                params = [c['estimated_params'] for c in sub]
                print(f"    层{layer}: {count:3d}个, 参数范围: {min(params):,}-{max(params):,}")
    
    print("\n" + "=" * 70)
    print("优化配置生成完成!")
    print(f"总配置数: {total_configs}")
    print(f"配置文件目录: {output_dir}")
    print("\n搜索策略:")
    print("  - MLP: 围绕[192,64,32]精细搜索 + 2层高效区扩展 + 4层验证")
    print("  - LSTM/GRU: 2-3层窄而深结构")
    print("  - CNN1D: 2-3层通道递减结构")
    print("=" * 70)


if __name__ == '__main__':
    main()
