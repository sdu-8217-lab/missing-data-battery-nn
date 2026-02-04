#!/usr/bin/env python3
"""
分析CSV架构搜索结果
汇总各模型的帕累托前沿
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_model_results(csv_path: str, model_type: str):
    """分析单个模型的结果"""
    df = pd.read_csv(csv_path)
    
    # 只分析已完成的
    df_done = df[df['status'] == 'completed'].copy()
    
    if len(df_done) == 0:
        print(f"{model_type.upper()}: 无完成实验")
        return None
    
    # 转换数值列
    for col in ['actual_params', 'mae', 'rmse', 'r2', 'training_time']:
        if col in df_done.columns:
            df_done[col] = pd.to_numeric(df_done[col], errors='coerce')
    
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} 分析结果")
    print(f"{'='*70}")
    print(f"完成配置数: {len(df_done)} / {len(df)}")
    
    # 按层分析
    for layer in sorted(df_done['layer_type'].unique()):
        layer_df = df_done[df_done['layer_type'] == layer]
        if len(layer_df) > 0:
            best = layer_df.loc[layer_df['mae'].idxmin()]
            print(f"\n层{layer}: {len(layer_df)}个配置")
            print(f"  最佳MAE: {best['mae']:.4f}")
            print(f"  配置: {best['config_id']}")
            print(f"  参数量: {best['actual_params']:,.0f}")
    
    # 全局最佳
    best_global = df_done.loc[df_done['mae'].idxmin()]
    print(f"\n{'='*70}")
    print(f"全局最佳:")
    print(f"  配置: {best_global['config_id']}")
    print(f"  MAE: {best_global['mae']:.4f}")
    print(f"  RMSE: {best_global['rmse']:.4f}")
    print(f"  R2: {best_global['r2']:.4f}")
    print(f"  参数量: {best_global['actual_params']:,.0f}")
    print(f"  配置详情: {best_global['hidden_config']}")
    print(f"{'='*70}")
    
    return df_done


def plot_pareto_comparison(all_results: dict, output_dir: Path):
    """绘制帕累托前沿对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'mlp': 'blue', 'lstm': 'orange', 'gru': 'green', 'cnn1d': 'red'}
    
    # 1. 帕累托前沿 - 所有模型
    ax = axes[0, 0]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            ax.scatter(df['actual_params'], df['mae'], 
                      alpha=0.5, s=30, label=model_type.upper(),
                      color=colors.get(model_type, 'gray'))
            
            # 标记最佳
            best = df.loc[df['mae'].idxmin()]
            ax.scatter(best['actual_params'], best['mae'],
                      s=200, marker='*', color=colors.get(model_type, 'gray'),
                      edgecolors='black', zorder=5)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('MAE')
    ax.set_title('Pareto Frontier: All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100000)
    
    # 2. 各层最佳对比
    ax = axes[0, 1]
    layer_best = {}
    for model_type, df in all_results.items():
        if df is not None:
            for layer in [1, 2, 3, 4]:
                layer_df = df[df['layer_type'] == layer]
                if len(layer_df) > 0:
                    best = layer_df.loc[layer_df['mae'].idxmin()]
                    key = f"{model_type}_l{layer}"
                    layer_best[key] = best['mae']
    
    if layer_best:
        items = sorted(layer_best.items(), key=lambda x: x[1])
        names = [x[0] for x in items[:15]]  # 只显示前15
        values = [x[1] for x in items[:15]]
        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('MAE')
        ax.set_title('Best MAE by Model & Layer')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. MAE分布
    ax = axes[1, 0]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            ax.hist(df['mae'], bins=20, alpha=0.4, label=model_type.upper(),
                   color=colors.get(model_type, 'gray'))
    ax.set_xlabel('MAE')
    ax.set_ylabel('Count')
    ax.set_title('MAE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 训练时间 vs 性能
    ax = axes[1, 1]
    for model_type, df in all_results.items():
        if df is not None and len(df) > 0:
            scatter = ax.scatter(df['training_time'], df['mae'],
                               alpha=0.5, s=30, label=model_type.upper(),
                               color=colors.get(model_type, 'gray'))
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('MAE')
    ax.set_title('Training Time vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'pareto_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表保存: {output_path}")


def generate_summary_csv(all_results: dict, output_dir: Path):
    """生成汇总CSV"""
    summary = []
    
    for model_type, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        
        # 全局最佳
        best = df.loc[df['mae'].idxmin()]
        summary.append({
            'model': model_type,
            'layer_type': 'best',
            'config_id': best['config_id'],
            'mae': best['mae'],
            'rmse': best['rmse'],
            'r2': best['r2'],
            'params': best['actual_params'],
            'training_time': best['training_time'],
            'config': best['hidden_config']
        })
        
        # 每层最佳
        for layer in sorted(df['layer_type'].unique()):
            layer_df = df[df['layer_type'] == layer]
            if len(layer_df) > 0:
                best_layer = layer_df.loc[layer_df['mae'].idxmin()]
                summary.append({
                    'model': model_type,
                    'layer_type': layer,
                    'config_id': best_layer['config_id'],
                    'mae': best_layer['mae'],
                    'rmse': best_layer['rmse'],
                    'r2': best_layer['r2'],
                    'params': best_layer['actual_params'],
                    'training_time': best_layer['training_time'],
                    'config': best_layer['hidden_config']
                })
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / 'summary_best_configs.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n汇总表保存: {summary_path}")
    print("\n最佳配置汇总:")
    print(summary_df[summary_df['layer_type'] == 'best'].to_string(index=False))
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='分析CSV搜索结果')
    parser.add_argument('--models', nargs='+', 
                       default=['mlp', 'lstm', 'gru', 'cnn1d'],
                       help='要分析的模型列表')
    parser.add_argument('--output', default='./analysis_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 分析各模型
    all_results = {}
    for model in args.models:
        csv_path = f'configs/{model}_configs.csv'
        if Path(csv_path).exists():
            results = analyze_model_results(csv_path, model)
            all_results[model] = results
        else:
            print(f"未找到: {csv_path}")
    
    # 生成图表
    if all_results:
        plot_pareto_comparison(all_results, output_dir)
        generate_summary_csv(all_results, output_dir)
    
    print("\n" + "="*70)
    print("分析完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
