#!/usr/bin/env python3
"""分析configs_v2目录下的模型筛选结果"""
import pandas as pd
import numpy as np
import json

def analyze_csv_results(csv_path, model_name):
    """分析单个CSV文件"""
    df = pd.read_csv(csv_path)
    
    # 只取完成的实验
    completed = df[df['status'] == 'completed'].copy()
    
    print(f"\n{'='*70}")
    print(f"【{model_name}】结果分析")
    print(f"{'='*70}")
    
    # 基本统计
    total = len(df)
    success = len(completed)
    failed = len(df[df['status'] == 'failed'])
    
    print(f"\n总配置: {total}")
    print(f"成功: {success} ({success/total*100:.1f}%)")
    print(f"失败: {failed} ({failed/total*100:.1f}%)")
    
    if len(completed) == 0:
        print("无成功完成的实验")
        return None
    
    # 参数范围
    print(f"\n参数范围: {completed['estimated_params'].min():,} - {completed['estimated_params'].max():,}")
    
    # MAE统计
    print(f"MAE范围: {completed['mae'].min():.4f} - {completed['mae'].max():.4f}")
    print(f"平均MAE: {completed['mae'].mean():.4f}")
    
    # 按MAE排序取TOP 10
    top10 = completed.sort_values('mae').head(10)
    
    print(f"\n【TOP 10 最佳配置】(按MAE排序)")
    print("-" * 70)
    print(f"{'排名':<4} {'配置ID':<20} {'MAE':<10} {'参数量':<12} {'R2':<10} {'配置'}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        config_str = row['hidden_config'].replace('"', '')
        print(f"{i:<4} {row['config_id']:<20} {row['mae']:<10.4f} "
              f"{int(row['estimated_params']):<12,} {row['r2']:<10.4f} {config_str}")
    
    # 效率分析 (MAE/params)
    completed['efficiency'] = completed['mae'] / (completed['estimated_params'] / 10000)
    efficient = completed.sort_values('efficiency').head(5)
    
    print(f"\n【TOP 5 最高效配置】(MAE per 10K params)")
    print("-" * 70)
    for i, (_, row) in enumerate(efficient.iterrows(), 1):
        print(f"{i}. {row['config_id']}: MAE={row['mae']:.4f}, "
              f"params={int(row['estimated_params']):,}, eff={row['efficiency']:.3f}")
    
    # 层数分析
    print(f"\n【层数分析】")
    for layer_type in sorted(completed['layer_type'].unique()):
        layer_df = completed[completed['layer_type'] == layer_type]
        print(f"  {int(layer_type)}层: 数量={len(layer_df)}, "
              f"最佳MAE={layer_df['mae'].min():.4f}, "
              f"平均MAE={layer_df['mae'].mean():.4f}")
    
    return top10.iloc[0]  # 返回最佳配置

# 分析所有模型
print("="*70)
print("Configs_v2 模型筛选结果分析")
print("="*70)

results = {}
best_configs = {}

for model, csv_file in [
    ('MLP', 'configs_v2/mlp_configs.csv'),
    ('CNN1D', 'configs_v2/cnn1d_configs.csv'),
    ('GRU', 'configs_v2/gru_configs.csv'),
    ('LSTM', 'configs_v2/lstm_configs.csv'),
]:
    try:
        best = analyze_csv_results(csv_file, model)
        if best is not None:
            best_configs[model] = best
    except Exception as e:
        print(f"\n{model} 分析失败: {e}")

# 总对比
print(f"\n\n{'='*70}")
print("【模型对比总结】")
print(f"{'='*70}")
print(f"{'模型':<8} {'最佳MAE':<10} {'参数量':<12} {'R2':<10} {'配置'}")
print("-" * 70)

for model, row in best_configs.items():
    config_str = row['hidden_config'].replace('"', '')
    print(f"{model:<8} {row['mae']:<10.4f} {int(row['estimated_params']):<12,} "
          f"{row['r2']:<10.4f} {config_str}")

# 关键发现
print(f"\n{'='*70}")
print("【关键发现与建议】")
print(f"{'='*70}")

# 找出最佳模型
best_model = min(best_configs.items(), key=lambda x: x[1]['mae'])
print(f"\n1. 最佳模型: {best_model[0]}")
print(f"   - 最佳MAE: {best_model[1]['mae']:.4f}")
print(f"   - 配置: {best_model[1]['hidden_config']}")
print(f"   - 参数量: {int(best_model[1]['estimated_params']):,}")

# 效率对比
print(f"\n2. 效率对比(MAE per 10K params，越低越好):")
for model, csv_file in [
    ('MLP', 'configs_v2/mlp_configs.csv'),
    ('CNN1D', 'configs_v2/cnn1d_configs.csv'),
    ('GRU', 'configs_v2/gru_configs.csv'),
    ('LSTM', 'configs_v2/lstm_configs.csv'),
]:
    df = pd.read_csv(csv_file)
    completed = df[df['status'] == 'completed']
    if len(completed) > 0:
        completed['efficiency'] = completed['mae'] / (completed['estimated_params'] / 10000)
        best_eff = completed.sort_values('efficiency').iloc[0]
        print(f"   {model}: {best_eff['efficiency']:.3f} (MAE={best_eff['mae']:.4f}, "
              f"params={int(best_eff['estimated_params']):,})")

print(f"\n3. 下一轮搜索建议:")
print(f"   - CNN1D: 围绕[56-72, 32] kernel=2-5, dropout=0.1-0.3加密搜索")
print(f"   - MLP: 当前结果较差，建议重新审视搜索范围")
print(f"   - GRU/LSTM: 单层宽网络(h=40-64)表现更好，建议减少层数")
