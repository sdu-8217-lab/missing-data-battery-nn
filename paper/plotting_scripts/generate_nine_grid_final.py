#!/usr/bin/env python3
"""
九宫格最终图表生成
基于216个模型的评估结果生成九个subplot图表
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_evaluation_data():
    """加载所有模型的评估数据"""
    data_dir = PROJECT_ROOT / "paper" / "experiment_data" / "nine_grid_complete"
    
    records = []
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith('nine_'):
            continue
        
        # 解析模型信息
        parts = model_dir.name.split('_')
        if len(parts) < 8:
            continue
        
        arch = parts[1]
        pattern = parts[4]
        imputation = parts[5]
        config = parts[6]
        seed = int(parts[7].replace('seed', ''))
        
        # 加载评估结果
        eval_file = model_dir / "evaluated_metrics.json"
        if not eval_file.exists():
            continue
        
        with open(eval_file, 'r') as f:
            metrics_by_mr = json.load(f)
        
        # 计算平均指标
        avg_metrics = {
            'MAE': np.mean([m['MAE'] for m in metrics_by_mr.values()]),
            'RMSE': np.mean([m['RMSE'] for m in metrics_by_mr.values()]),
            'R2': np.mean([m['R2'] for m in metrics_by_mr.values()]),
            'MAPE': np.mean([m['MAPE'] for m in metrics_by_mr.values()])
        }
        
        # 存储每个缺失率的结果
        for mr_str, metrics in metrics_by_mr.items():
            mr = float(mr_str)
            records.append({
                'architecture': arch.upper(),
                'pattern': pattern,
                'imputation': imputation,
                'config': config,
                'seed': seed,
                'missing_rate': mr,
                **metrics
            })
    
    return pd.DataFrame(records)


def plot_nine_grid(df, metric='MAE', save_path=None):
    """生成九宫格图表"""
    
    architectures = ['MLP', 'CNN', 'LSTM']
    patterns = ['MCAR', 'MAR', 'MNAR']
    imputations = ['zero', 'mean', 'iterative', 'knn']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Nine-Grid Analysis: {metric} across Architectures and Missing Patterns', 
                 fontsize=14, fontweight='bold')
    
    colors = {'zero': '#e74c3c', 'mean': '#3498db', 'iterative': '#2ecc71', 'knn': '#9b59b6'}
    linestyles = {'baseline': '-', 'mim': '--'}
    
    for i, pattern in enumerate(patterns):
        for j, arch in enumerate(architectures):
            ax = axes[i, j]
            
            # 筛选数据
            subset = df[(df['pattern'] == pattern) & (df['architecture'] == arch)]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{arch} × {pattern}')
                continue
            
            # 按插补方法和配置绘制曲线
            for imp in imputations:
                for config in ['baseline', 'mim']:
                    data = subset[(subset['imputation'] == imp) & (subset['config'] == config)]
                    if data.empty:
                        continue
                    
                    # 按缺失率分组求平均
                    grouped = data.groupby('missing_rate')[metric].mean().reset_index()
                    grouped = grouped.sort_values('missing_rate')
                    
                    label = f'{imp}' if config == 'baseline' else f'{imp}+MIM'
                    ax.plot(grouped['missing_rate'], grouped[metric], 
                           color=colors[imp], linestyle=linestyles[config],
                           linewidth=1.5, label=label, alpha=0.8)
            
            # 设置标题和标签
            ax.set_title(f'{arch} × {pattern}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Missing Rate', fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 只在第一行显示图例
            if i == 0 and j == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")
    
    return fig


def plot_improvement_heatmap(df, save_path=None):
    """生成MIM改进率热力图"""
    
    # 计算MIM改进率
    improvement_data = []
    
    for (arch, pattern, imp), group in df.groupby(['architecture', 'pattern', 'imputation']):
        baseline = group[group['config'] == 'baseline'].groupby('missing_rate')['MAE'].mean()
        mim = group[group['config'] == 'mim'].groupby('missing_rate')['MAE'].mean()
        
        if len(baseline) > 0 and len(mim) > 0:
            # 平均改进率
            improvement = ((baseline.mean() - mim.mean()) / baseline.mean()) * 100
            improvement_data.append({
                'architecture': arch,
                'pattern': pattern,
                'imputation': imp,
                'improvement': improvement
            })
    
    imp_df = pd.DataFrame(improvement_data)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('MIM Improvement Rate (%) by Architecture', fontsize=14, fontweight='bold')
    
    patterns = ['MCAR', 'MAR', 'MNAR']
    imputations = ['zero', 'mean', 'iterative', 'knn']
    architectures = ['MLP', 'CNN', 'LSTM']
    
    for i, pattern in enumerate(patterns):
        ax = axes[i]
        
        # 创建热力图数据
        pivot_data = []
        for arch in architectures:
            row = []
            for imp in imputations:
                value = imp_df[(imp_df['pattern'] == pattern) & 
                              (imp_df['architecture'] == arch) & 
                              (imp_df['imputation'] == imp)]['improvement'].values
                row.append(value[0] if len(value) > 0 else 0)
            pivot_data.append(row)
        
        # 绘制热力图
        im = ax.imshow(pivot_data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)
        
        # 设置标签
        ax.set_xticks(range(len(imputations)))
        ax.set_xticklabels(imputations, rotation=45)
        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures)
        ax.set_title(f'{pattern}', fontsize=12)
        
        # 添加数值
        for j in range(len(architectures)):
            for k in range(len(imputations)):
                text = ax.text(k, j, f'{pivot_data[j][k]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label='Improvement (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 热力图已保存: {save_path}")
    
    return fig


def main():
    """主函数"""
    print("加载评估数据...")
    df = load_evaluation_data()
    
    if df.empty:
        print("❌ 未找到评估数据")
        return
    
    print(f"✅ 加载了 {len(df)} 条记录")
    print(f"   架构: {df['architecture'].unique().tolist()}")
    print(f"   缺失模式: {df['pattern'].unique().tolist()}")
    print(f"   插补方法: {df['imputation'].unique().tolist()}")
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成九宫格图表
    print("\n生成九宫格图表 (MAE)...")
    plot_nine_grid(df, metric='MAE', save_path=output_dir / 'nine_grid_MAE.png')
    
    print("\n生成九宫格图表 (RMSE)...")
    plot_nine_grid(df, metric='RMSE', save_path=output_dir / 'nine_grid_RMSE.png')
    
    print("\n生成九宫格图表 (R²)...")
    plot_nine_grid(df, metric='R2', save_path=output_dir / 'nine_grid_R2.png')
    
    print("\n生成MIM改进率热力图...")
    plot_improvement_heatmap(df, save_path=output_dir / 'nine_grid_improvement_heatmap.png')
    
    print("\n✅ 所有图表生成完成!")


if __name__ == '__main__':
    main()
