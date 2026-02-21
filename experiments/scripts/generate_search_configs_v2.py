# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚浼樺寲鍚庣殑鏋舵瀯鎼滅储閰嶇疆锛堝熀浜庝笂涓€杞粨鏋滐級
- 鍥寸粫鏈€浣崇偣 [192,64,32] 鍔犲瘑鎼滅储
- 鎵╁睍楂樻晥鍖?2灞傞厤缃?
- 楠岃瘉4灞傛綔鍔?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.models.model_factory import ModelFactory

PARAM_MAX = 65536


def calc_exact_params(model_type, config, use_mim=False):
    """绮剧‘璁＄畻鍙傛暟閲?""
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
    鍩轰簬涓婁竴杞粨鏋滀紭鍖栫殑MLP閰嶇疆
    涓婁竴杞渶浣? [192,64,32] d0.2, MAE=0.0157, 20,801 params
    """
    configs = []
    
    # ========== 绛栫暐1: 鍥寸粫鏈€浣崇偣 [192,64,32] 绮剧粏鎼滅储 ==========
    base_first = [160, 176, 192, 208, 224]  # 192闄勮繎
    base_second = [48, 56, 64, 72, 80]      # 64闄勮繎
    base_third = [24, 28, 32, 36, 40]       # 32闄勮繎
    
    # 3灞傜簿缁嗘悳绱?- 鍥哄畾鍏朵粬涓ゅ眰锛屽彉鍖栦竴灞?
    # 鍙樺寲绗竴灞?
    for h1 in base_first:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [h1, 64, 32], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t1'))
    
    # 鍙樺寲绗簩灞?
    for h2 in base_second:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [192, h2, 32], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t2'))
    
    # 鍙樺寲绗笁灞?
    for h3 in base_third:
        for dropout in [0.15, 0.2, 0.25, 0.3]:
            cfg = {'hidden_layers': [192, 64, h3], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 3, cfg, params, 'fine_t3'))
    
    # ========== 绛栫暐2: 楂樻晥鍖?灞傛墿灞?==========
    # 涓婁竴杞? [64,32] 琛ㄧ幇浼樺紓锛?,225 params, MAE鈮?.0163
    efficient_first = [48, 56, 64, 72, 80, 96, 112]
    efficient_second = [24, 28, 32, 36, 40, 48]
    
    for h1 in efficient_first:
        for h2 in efficient_second:
            if h2 >= h1:  # 淇濊瘉閫掑噺
                continue
            for dropout in [0.1, 0.2, 0.3, 0.4]:
                cfg = {'hidden_layers': [h1, h2], 'dropout': dropout}
                params = calc_exact_params('mlp', cfg)
                if params <= PARAM_MAX and params < 15000:  # 淇濇寔楂樻晥
                    configs.append(create_config_entry(configs, 2, cfg, params, 'efficient'))
    
    # ========== 绛栫暐3: 4灞傞獙璇?==========
    # 鎺㈢储鏇存繁缃戠粶鏄惁鑳借秴瓒?灞?
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
    
    # ========== 绛栫暐4: 1灞傚ぇ瀹藉害鎺㈢储 ==========
    # 楠岃瘉绠€鍗曟ā鍨嬬殑鏋侀檺
    for hidden in [256, 320, 384, 448, 512, 640, 768]:
        for dropout in [0.0, 0.1, 0.2]:
            cfg = {'hidden_layers': [hidden], 'dropout': dropout}
            params = calc_exact_params('mlp', cfg)
            if params <= PARAM_MAX:
                configs.append(create_config_entry(configs, 1, cfg, params, 'wide'))
    
    return configs


def create_config_entry(existing_configs, layer_type, cfg, params, tag):
    """鍒涘缓閰嶇疆鏉＄洰"""
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
    """鍩轰簬MLP缁撴灉鎺ㄦ祴LSTM浼樺寲鏂瑰悜锛堟洿娣变絾鏇寸獎鍙兘鏇村ソ锛?""
    configs = []
    
    # 鍥寸粫鍙兘鐨勬渶浣崇偣鍔犲瘑
    # 涓婁竴杞渶浣冲€惧悜: h=56-64, l=2-3
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
    """GRU浼樺寲閰嶇疆"""
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
    """CNN1D浼樺寲閰嶇疆 - 绫讳技MLP鐨勬墿寮犵瓥鐣?""
    configs = []
    
    # 2灞傞珮鏁堝尯鎵╁睍
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
    
    # 3灞傜簿缁嗘悳绱?
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
    """鐢熸垚浼樺寲鍚庣殑閰嶇疆鏂囦欢"""
    output_dir = Path('./configs_v2')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("鐢熸垚浼樺寲鍚庣殑鏋舵瀯鎼滅储閰嶇疆 (鍩轰簬涓婁竴杞粨鏋?")
    print("=" * 70)
    
    generators = {
        'mlp': generate_mlp_optimized_configs,
        'lstm': generate_lstm_optimized_configs,
        'gru': generate_gru_optimized_configs,
        'cnn1d': generate_cnn_optimized_configs,
    }
    
    total_configs = 0
    for model_name, generator in generators.items():
        print(f"\n鐢熸垚 {model_name.upper()} 浼樺寲閰嶇疆...")
        configs = generator()
        
        if not configs:
            print(f"  璀﹀憡: {model_name} 娌℃湁鐢熸垚閰嶇疆")
            continue
            
        df = pd.DataFrame(configs)
        output_file = output_dir / f'{model_name}_configs.csv'
        df.to_csv(output_file, index=False)
        
        total_configs += len(configs)
        print(f"  鐢熸垚 {len(configs)} 涓厤缃?)
        print(f"  淇濆瓨鍒? {output_file}")
        
        # 缁熻
        for layer in [1, 2, 3, 4]:
            count = len([c for c in configs if c['layer_type'] == layer])
            if count > 0:
                sub = [c for c in configs if c['layer_type'] == layer]
                params = [c['estimated_params'] for c in sub]
                print(f"    灞倇layer}: {count:3d}涓? 鍙傛暟鑼冨洿: {min(params):,}-{max(params):,}")
    
    print("\n" + "=" * 70)
    print("浼樺寲閰嶇疆鐢熸垚瀹屾垚!")
    print(f"鎬婚厤缃暟: {total_configs}")
    print(f"閰嶇疆鏂囦欢鐩綍: {output_dir}")
    print("\n鎼滅储绛栫暐:")
    print("  - MLP: 鍥寸粫[192,64,32]绮剧粏鎼滅储 + 2灞傞珮鏁堝尯鎵╁睍 + 4灞傞獙璇?)
    print("  - LSTM/GRU: 2-3灞傜獎鑰屾繁缁撴瀯")
    print("  - CNN1D: 2-3灞傞€氶亾閫掑噺缁撴瀯")
    print("=" * 70)


if __name__ == '__main__':
    main()

