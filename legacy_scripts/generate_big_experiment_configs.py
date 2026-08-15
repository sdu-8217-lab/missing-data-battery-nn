#!/usr/bin/env python3
"""生成大实验配置 - 完整的MIM研究设计

实验矩阵:
- 4模型: MLP, LSTM, GRU, CNN1D
- 2策略: baseline, mim
- 9缺失率: 0.1, 0.2, ..., 0.9
- 100次重复
- 总计: 7200次实验
"""
import os
import csv

# 创建目录
os.makedirs('configs_big', exist_ok=True)

# 各模型最佳配置（来自configs_v3）
MODEL_CONFIGS = {
    'mlp': {
        'hidden_config': '[192,96,48,24]',
        'dropout': 0.15,
        'layer_type': 4,
        'params_baseline': 27649,  # 输入16
        'params_mim': 27713,       # 输入32
    },
    'lstm': {
        'hidden_config': 'h=48,l=2',
        'dropout': 0.2,
        'layer_type': 2,
        'params_baseline': 31537,  # 输入16
        'params_mim': 33025,       # 输入32
    },
    'gru': {
        'hidden_config': 'h=64,l=2',
        'dropout': 0.2,
        'layer_type': 2,
        'params_baseline': 40769,  # 输入16
        'params_mim': 42241,       # 输入32
    },
    'cnn1d': {
        'hidden_config': '[72,32]',
        'kernel': 4,
        'dropout': 0.1,
        'layer_type': 2,
        'params_baseline': 16713,  # 输入16
        'params_mim': 17577,       # 输入32
    }
}

MISSING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS = 100
BATCH = '3C'
EPOCHS = 200

def generate_model_configs(model_type):
    """为单个模型生成完整实验配置"""
    base_config = MODEL_CONFIGS[model_type]
    rows = []
    exp_counter = 0
    
    for strategy in ['baseline', 'mim']:
        input_dim = 16 if strategy == 'baseline' else 32
        params = base_config['params_baseline'] if strategy == 'baseline' else base_config['params_mim']
        
        for missing_rate in MISSING_RATES:
            for repeat_id in range(N_REPEATS):
                exp_id = f"{model_type}_{strategy}_mr{int(missing_rate*10):02d}_r{repeat_id:03d}"
                
                row = {
                    'exp_id': exp_id,
                    'model_type': model_type,
                    'strategy': strategy,
                    'input_dim': input_dim,
                    'layer_type': base_config['layer_type'],
                    'hidden_config': base_config['hidden_config'],
                    'dropout': base_config['dropout'],
                    'kernel': base_config.get('kernel', ''),
                    'missing_rate': missing_rate,
                    'batch': BATCH,
                    'epochs': EPOCHS,
                    'repeat': repeat_id,
                    'seed': 10000 + exp_counter,  # 唯一随机种子
                    'estimated_params': params,
                    # 结果字段（初始为空）
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                }
                rows.append(row)
                exp_counter += 1
    
    # 写入CSV
    csv_path = f'configs_big/{model_type}_big_experiment.csv'
    fieldnames = ['exp_id', 'model_type', 'strategy', 'input_dim', 'layer_type', 
                  'hidden_config', 'kernel', 'dropout', 'missing_rate', 'batch', 
                  'epochs', 'repeat', 'seed', 'estimated_params', 'actual_params',
                  'mae', 'rmse', 'r2', 'training_time', 'status']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f'{model_type.upper()}: {len(rows)} experiments -> {csv_path}')
    return csv_path, len(rows)

# 生成所有模型配置
print('='*70)
print('大实验配置生成 - 完整MIM研究设计')
print('='*70)
print(f'配置: 4模型 × 2策略 × 9缺失率 × {N_REPEATS}重复 = 7200次实验')
print(f'批次: {BATCH} | Epochs: {EPOCHS}')
print('='*70)

total_exps = 0
for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
    _, count = generate_model_configs(model)
    total_exps += count

print('='*70)
print(f'总计: {total_exps}次实验')
print('配置文件位置: configs_big/')
