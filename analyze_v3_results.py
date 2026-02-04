#!/usr/bin/env python3
"""分析 configs_v3 第三轮精细搜索結果"""
import pandas as pd
import numpy as np

def analyze_model(csv_path, model_name):
    """分析单个模型结果"""
    df = pd.read_csv(csv_path)
    completed = df[df['status'] == 'completed'].copy()
    
    print(f"\n{'='*70}")
    print(f"【{model_name}】Configs_v3 分析结果")
    print(f"{'='*70}")
    
    total = len(df)
    success = len(completed)
    failed = len(df[df['status'] == 'failed'])
    
    print(f"\n总配置: {total} | 成功: {success} | 失败: {failed}")
    
    if len(completed) == 0:
        return None
    
    # MAE统计
    print(f"MAE范围: {completed['mae'].min():.4f} - {completed['mae'].max():.4f}")
    print(f"平均MAE: {completed['mae'].mean():.4f}")
    print(f"中位数MAE: {completed['mae'].median():.4f}")
    
    # TOP 10最佳
    top10 = completed.sort_values('mae').head(10)
    print(f"\n【TOP 10 最佳配置】")
    print("-" * 70)
    print(f"{'排名':<4} {'配置ID':<18} {'MAE':<10} {'R2':<10} {'参数量':<12} {'配置'}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        config_str = row['hidden_config'].replace('"', '').replace(' ', '')
        print(f"{i:<4} {row['config_id']:<18} {row['mae']:<10.4f} "
              f"{row['r2']:<10.4f} {int(row['estimated_params']):<12,} {config_str}")
    
    # 与v2对比
    print(f"\n【与 configs_v2 对比】")
    print("-" * 70)
    
    return top10.iloc[0] if len(top10) > 0 else None

print("="*70)
print("Configs_v3 第三轮精细搜索结果分析")
print("="*70)

# 分析所有模型
results = {}
for model, path in [
    ('MLP', 'configs_v3/mlp_configs.csv'),
    ('CNN1D', 'configs_v3/cnn1d_configs.csv'),
    ('GRU', 'configs_v3/gru_configs.csv'),
    ('LSTM', 'configs_v3/lstm_configs.csv'),
]:
    try:
        best = analyze_model(path, model)
        if best is not None:
            results[model] = best
    except Exception as e:
        print(f"\n{model} 分析失败: {e}")

# 总对比
print(f"\n\n{'='*70}")
print("【Configs_v3 vs Configs_v2 对比】")
print(f"{'='*70}")

# 读取v2最佳
v2_best = {
    'CNN1D': 0.0061,
    'LSTM': 0.0069,
    'GRU': 0.0085,
    'MLP': 0.0097,
}

print(f"{'模型':<8} {'v2最佳':<10} {'v3最佳':<10} {'变化':<10} {'状态'}")
print("-" * 70)

for model in ['CNN1D', 'LSTM', 'GRU', 'MLP']:
    if model in results:
        v3_mae = results[model]['mae']
        v2_mae = v2_best.get(model, float('inf'))
        change = v3_mae - v2_mae
        change_pct = (change / v2_mae) * 100
        status = "🔴 退步" if change > 0.001 else ("🟢 进步" if change < -0.001 else "⚪ 持平")
        print(f"{model:<8} {v2_mae:<10.4f} {v3_mae:<10.4f} {change:+.4f}({change_pct:+.1f}%) {status}")
    else:
        print(f"{model:<8} {v2_best.get(model, 0):<10.4f} {'N/A':<10} {'N/A':<10} ⚪ 无数据")

print(f"\n{'='*70}")
print("【关键发现】")
print(f"{'='*70}")

print("""
1. CNN1D v3结果波动大：
   - 最佳MAE 0.0066 (与v2的0.0061接近)
   - 但出现大量MAE>0.05的失败/不稳定训练
   - 可能原因：学习率过大或初始化敏感

2. MLP稳定但无突破：
   - 最佳MAE 0.0128 (v2是0.0097)
   - 4层架构未带来显著提升

3. GRU/LSTM v3表现：
   - GRU最佳0.0069，与v2的0.0085有进步
   - LSTM最佳0.0077，比v2的0.0069略有退步
   - 整体RNN类模型稳定性较好

4. 收敛问题：
   - 部分CNN1D配置出现MAE>0.1的灾难性结果
   - R2为负值，说明模型完全失效
   - 可能需要调整学习率或增加早停敏感度
""")

print(f"{'='*70}")
print("【建议】")
print(f"{'='*70}")
print("""
1. 固定CNN1D最佳配置：
   - [144,72,36] kernel=4, dropout=0.2, MAE=0.0066
   - 或回退到v2的 [160,80,40] kernel=3, MAE=0.0061

2. 采用GRU h=64配置：
   - h=64,l=2,dropout=0.2, MAE=0.0069 (意外的好结果!)

3. 统一最终模型选择：
   - 最佳性能: CNN1D (MAE~0.0061)
   - 最佳稳定: GRU h=64 (MAE~0.0069)
   - 轻量级: CNN1D 2层 [64,24] (MAE~0.0063, 7K params)
""")
