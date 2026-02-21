# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""鐢熸垚澶у疄楠岄厤缃?- 瀹屾暣鐨凪IM鐮旂┒璁捐

瀹為獙鐭╅樀:
- 4妯″瀷: MLP, LSTM, GRU, CNN1D
- 2绛栫暐: baseline, mim
- 9缂哄け鐜? 0.1, 0.2, ..., 0.9
- 100娆￠噸澶?
- 鎬昏: 7200娆″疄楠?
"""
import os
import csv

# 鍒涘缓鐩綍
os.makedirs('configs_big', exist_ok=True)

# 鍚勬ā鍨嬫渶浣抽厤缃紙鏉ヨ嚜configs_v3锛?
MODEL_CONFIGS = {
    'mlp': {
        'hidden_config': '[192,96,48,24]',
        'dropout': 0.15,
        'layer_type': 4,
        'params_baseline': 27649,  # 杈撳叆16
        'params_mim': 27713,       # 杈撳叆32
    },
    'lstm': {
        'hidden_config': 'h=48,l=2',
        'dropout': 0.2,
        'layer_type': 2,
        'params_baseline': 31537,  # 杈撳叆16
        'params_mim': 33025,       # 杈撳叆32
    },
    'gru': {
        'hidden_config': 'h=64,l=2',
        'dropout': 0.2,
        'layer_type': 2,
        'params_baseline': 40769,  # 杈撳叆16
        'params_mim': 42241,       # 杈撳叆32
    },
    'cnn1d': {
        'hidden_config': '[72,32]',
        'kernel': 4,
        'dropout': 0.1,
        'layer_type': 2,
        'params_baseline': 16713,  # 杈撳叆16
        'params_mim': 17577,       # 杈撳叆32
    }
}

MISSING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS = 100
BATCH = '3C'
EPOCHS = 200

def generate_model_configs(model_type):
    """涓哄崟涓ā鍨嬬敓鎴愬畬鏁村疄楠岄厤缃?""
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
                    'seed': 10000 + exp_counter,  # 鍞竴闅忔満绉嶅瓙
                    'estimated_params': params,
                    # 缁撴灉瀛楁锛堝垵濮嬩负绌猴級
                    'actual_params': '',
                    'mae': '',
                    'rmse': '',
                    'r2': '',
                    'training_time': '',
                    'status': 'pending'
                }
                rows.append(row)
                exp_counter += 1
    
    # 鍐欏叆CSV
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

# 鐢熸垚鎵€鏈夋ā鍨嬮厤缃?
print('='*70)
print('澶у疄楠岄厤缃敓鎴?- 瀹屾暣MIM鐮旂┒璁捐')
print('='*70)
print(f'閰嶇疆: 4妯″瀷 脳 2绛栫暐 脳 9缂哄け鐜?脳 {N_REPEATS}閲嶅 = 7200娆″疄楠?)
print(f'鎵规: {BATCH} | Epochs: {EPOCHS}')
print('='*70)

total_exps = 0
for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
    _, count = generate_model_configs(model)
    total_exps += count

print('='*70)
print(f'鎬昏: {total_exps}娆″疄楠?)
print('閰嶇疆鏂囦欢浣嶇疆: configs_big/')

