"""
Figure 4: 模型架构 × 参数规模效应
展示不同架构（MLP/CNN/LSTM）和参数量对 MIM 效果的影响

数据来源: results/plotting_data_v0.5.csv
作者: Kimi
日期: 2026-03-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 配置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'mlp': '#e74c3c',
    'cnn': '#3498db',
    'lstm': '#2ecc71',
}

OUTPUT_DIR = Path(__file__).parent.parent

# ==================== 数据加载 ====================
print("加载数据...")
df = pd.read_csv('results/plotting_data_v0.5.csv')


# 参数数量表（硬编码，对应 Table 1）
PARAM_TABLE = {
    ('mlp', 'level_1'): 5953,
    ('mlp', 'level_2'): 11801,
    ('mlp', 'level_3'): 20537,
    ('mlp', 'level_4'): 44529,
    ('cnn', 'level_1'): 5377,
    ('cnn', 'level_2'): 12249,
    ('cnn', 'level_3'): 20945,
    ('cnn', 'level_4'): 36185,
    ('lstm', 'level_1'): 5181,
    ('lstm', 'level_2'): 10957,
    ('lstm', 'level_3'): 26797,
    ('lstm', 'level_4'): 54337,
}

def get_param_count(arch, level):
    """获取指定架构和层级的参数量"""
    return PARAM_TABLE.get((arch, level), 0)

def calculate_improvement(b, m):
    if pd.isna(b) or pd.isna(m) or b == 0:
        return 0
    return ((b - m) / b * 100)

# ==================== 子图 (a): 三种架构的改进率曲线 ====================
def plot_subplot_a(ax):
    """绘制三种架构的改进率曲线"""
    missing_rates = sorted(df['missing_rate'].unique())
    
    architectures = ["mlp", "cnn"]  # lstm 数据暂未训练
    for arch in architectures:
        arch_data = df[df['architecture'] == arch]
        baseline_data = arch_data[arch_data['config_type'] == 'baseline']
        mim_data = arch_data[arch_data['config_type'] == 'mim']
        
        improvements = []
        for mr in missing_rates:
            b = baseline_data[baseline_data['missing_rate'] == mr]['rmse'].mean()
            m = mim_data[mim_data['missing_rate'] == mr]['rmse'].mean()
            improvements.append(calculate_improvement(b, m))
        
        mr_pct = [mr * 100 for mr in missing_rates]
        ax.plot(mr_pct, improvements, '-', color=COLORS[arch], 
                linewidth=2.5, label=arch.upper(), marker='o', markersize=5)
    
    ax.set_xlabel('Missing Rate (%)', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('(a) Improvement by Architecture\n(All Levels Averaged)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 80)

# ==================== 子图 (b): 参数量 × 架构 热力图 ====================
def plot_subplot_b(ax):
    """绘制参数量和架构的组合热力图"""
    architectures = ['mlp', 'cnn', 'lstm']
    levels = ['level_1', 'level_2', 'level_3', 'level_4']
    
    improvement_matrix = np.zeros((len(architectures), len(levels)))
    param_matrix = np.zeros((len(architectures), len(levels)))
    
    for i, arch in enumerate(architectures):
        for j, level in enumerate(levels):
            subset = df[(df['architecture'] == arch) & (df['level'] == level)]
            if len(subset) == 0:
                improvement_matrix[i, j] = 0
                param_matrix[i, j] = 0
                continue
            
            baseline = subset[subset['config_type'] == 'baseline']
            mim = subset[subset['config_type'] == 'mim']
            
            # 计算平均改进率
            improvements = []
            for mr in [0.2, 0.4, 0.6, 0.8]:
                b = baseline[baseline['missing_rate'] == mr]['rmse'].mean()
                m = mim[mim['missing_rate'] == mr]['rmse'].mean()
                if pd.notna(b) and pd.notna(m):
                    improvements.append(calculate_improvement(b, m))
            
            improvement_matrix[i, j] = np.mean(improvements) if improvements else 0
            param_matrix[i, j] = get_param_count(arch, level) if len(subset) > 0 else 0
    
    # 绘制热力图
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=70)
    
    # 设置刻度
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f'Level-{i+1}' for i in range(len(levels))])
    ax.set_yticks(range(len(architectures)))
    ax.set_yticklabels([arch.upper() for arch in architectures])
    
    # 添加数值标注（改进率和参数量）
    for i in range(len(architectures)):
        for j in range(len(levels)):
            imp_val = improvement_matrix[i, j]
            param_val = int(param_matrix[i, j])
            text_color = 'white' if imp_val < 35 else 'black'
            ax.text(j, i, f'{imp_val:.0f}%\n({param_val:,})', 
                   ha='center', va='center', fontsize=9,
                   color=text_color, fontweight='bold')
    
    ax.set_xlabel('Model Level', fontweight='bold')
    ax.set_ylabel('Architecture', fontweight='bold')
    ax.set_title('(b) Improvement × Architecture × Level\n(Value = Improvement%, Param Count)', 
                 fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=15)

# ==================== 子图 (c): Level 1 vs Level 4 对比 ====================
def plot_subplot_c(ax):
    """绘制小模型 vs 大模型对比"""
    imputations = ['zero', 'mean', 'iterative']
    x = np.arange(len(imputations))
    width = 0.35
    
    # Level 1 (小模型)
    level1_improvements = []
    for imp in imputations:
        subset = df[(df['level'] == 'level_1') & (df['imputation_method'] == imp)]
        baseline = subset[subset['config_type'] == 'baseline']
        mim = subset[subset['config_type'] == 'mim']
        
        improvements = []
        for mr in [0.2, 0.4, 0.6]:
            b = baseline[baseline['missing_rate'] == mr]['rmse'].mean()
            m = mim[mim['missing_rate'] == mr]['rmse'].mean()
            if pd.notna(b) and pd.notna(m):
                improvements.append(calculate_improvement(b, m))
        level1_improvements.append(np.mean(improvements) if improvements else 0)
    
    # Level 4 (大模型)
    level4_improvements = []
    for imp in imputations:
        subset = df[(df['level'] == 'level_4') & (df['imputation_method'] == imp)]
        if len(subset) == 0:
            level4_improvements.append(0)
            continue
        baseline = subset[subset['config_type'] == 'baseline']
        mim = subset[subset['config_type'] == 'mim']
        
        improvements = []
        for mr in [0.2, 0.4, 0.6]:
            b = baseline[baseline['missing_rate'] == mr]['rmse'].mean()
            m = mim[mim['missing_rate'] == mr]['rmse'].mean()
            if pd.notna(b) and pd.notna(m):
                improvements.append(calculate_improvement(b, m))
        level4_improvements.append(np.mean(improvements) if improvements else 0)
    
    bars1 = ax.bar(x - width/2, level1_improvements, width, label='Level-1 (Small)', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, level4_improvements, width, label='Level-4 (Large)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # 添加数值标注
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Imputation Method', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('(c) Small vs Large Model Comparison\n(Level-1 ~6K vs Level-4 ~45K params)', 
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([imp.title() for imp in imputations])
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 80)

# ==================== 主函数 ====================
def main():
    """生成 Figure 4"""
    print("生成 Figure 4: Architecture and Parameter Scaling...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    plot_subplot_a(axes[0])
    plot_subplot_b(axes[1])
    plot_subplot_c(axes[2])
    plt.tight_layout()
    
    # 保存
    output_png = OUTPUT_DIR / 'architecture_scaling.png'
    output_pdf = OUTPUT_DIR / 'architecture_scaling.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 已保存: {output_png}")
    print(f"✅ 已保存: {output_pdf}")
    
    plt.close()

if __name__ == '__main__':
    main()
