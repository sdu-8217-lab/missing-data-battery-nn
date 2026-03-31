"""
Figure 1: MIM 核心效应（全景，2×2 复合图）
展示 MIM 在不同插补方法下的核心效果，包括误差曲线、改进率热力图、种子分布和参数效应

数据来源: results/plotting_data_v0.5.csv
作者: Kimi
日期: 2026-03-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path

# ==================== 配置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# 配色方案
COLORS = {
    'zero': '#e74c3c',      # 红
    'mean': '#f39c12',      # 橙
    'iterative': '#3498db', # 蓝
    'baseline': '#95a5a6',  # 灰
}

OUTPUT_DIR = Path(__file__).parent.parent  # 上级目录


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

# ==================== 数据加载与处理 ====================
print("加载数据...")
df = pd.read_csv('results/plotting_data_v0.5.csv')

# 筛选 2C MLP 数据作为主展示
data = df[(df['batch'] == '2C') & (df['architecture'] == 'mlp')].copy()

# 分离 Baseline (use_mim=False) 和 MIM (use_mim=True)
baseline_data = data[data['config_type'] == 'baseline']
mim_data = data[data['config_type'] == 'mim']

print(f"数据加载完成: Baseline={len(baseline_data)}, MIM={len(mim_data)}")

# ==================== 计算改进率 ====================
def calculate_improvement(baseline_vals, mim_vals):
    """计算百分比改进率"""
    return ((baseline_vals - mim_vals) / baseline_vals * 100)

# ==================== 子图 (a): 全缺失率范围误差曲线 ====================
def plot_subplot_a(ax):
    """绘制 RMSE 随缺失率变化曲线"""
    missing_rates = sorted(data['missing_rate'].unique())
    
    for imp in ['zero', 'mean', 'iterative']:
        baseline_vals = []
        mim_vals = []
        
        for mr in missing_rates:
            b = baseline_data[
                (baseline_data['imputation_method'] == imp) & 
                (baseline_data['missing_rate'] == mr)
            ]['rmse'].mean()
            m = mim_data[
                (mim_data['imputation_method'] == imp) & 
                (mim_data['missing_rate'] == mr)
            ]['rmse'].mean()
            baseline_vals.append(b)
            mim_vals.append(m)
        
        mr_pct = [mr * 100 for mr in missing_rates]
        
        # Baseline (虚线)
        ax.plot(mr_pct, baseline_vals, '--', color=COLORS[imp], 
                linewidth=1.5, alpha=0.6)
        # MIM (实线)
        ax.plot(mr_pct, mim_vals, '-', color=COLORS[imp], 
                linewidth=2.5, label=f'{imp.title()}', marker='o', markersize=4)
    
    # 标注 MR=40% 的改进率
    mr_40_idx = missing_rates.index(0.4)
    for imp in ['zero', 'mean', 'iterative']:
        b = baseline_vals[mr_40_idx] if imp == 'zero' else None
        m = mim_vals[mr_40_idx] if imp == 'zero' else None
        
        # 重新计算
        b_val = baseline_data[
            (baseline_data['imputation_method'] == imp) & 
            (baseline_data['missing_rate'] == 0.4)
        ]['rmse'].mean()
        m_val = mim_data[
            (mim_data['imputation_method'] == imp) & 
            (mim_data['missing_rate'] == 0.4)
        ]['rmse'].mean()
        improvement = calculate_improvement(b_val, m_val)
        
        # 添加数值标注
        ax.annotate(f'{improvement:.1f}%', xy=(40, m_val), 
                   xytext=(45, m_val), fontsize=8, color=COLORS[imp],
                   fontweight='bold')
    
    ax.set_xlabel('Missing Rate (%)', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('(a) RMSE vs Missing Rate\n(Zero/Mean/Iterative)', fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

# ==================== 子图 (b): 改进率热力图 ====================
def plot_subplot_b(ax):
    """绘制改进率热力图"""
    missing_rates = sorted(data['missing_rate'].unique())
    imputations = ['zero', 'mean', 'iterative']
    
    # 构建改进率矩阵
    improvement_matrix = np.zeros((len(imputations), len(missing_rates)))
    
    for i, imp in enumerate(imputations):
        for j, mr in enumerate(missing_rates):
            b = baseline_data[
                (baseline_data['imputation_method'] == imp) & 
                (baseline_data['missing_rate'] == mr)
            ]['rmse'].mean()
            m = mim_data[
                (mim_data['imputation_method'] == imp) & 
                (mim_data['missing_rate'] == mr)
            ]['rmse'].mean()
            improvement_matrix[i, j] = calculate_improvement(b, m)
    
    # 绘制热力图
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=0, vmax=80)
    
    # 设置刻度
    ax.set_xticks(range(0, len(missing_rates), 4))
    ax.set_xticklabels([f'{missing_rates[i]*100:.0f}' for i in range(0, len(missing_rates), 4)])
    ax.set_yticks(range(len(imputations)))
    ax.set_yticklabels([imp.title() for imp in imputations])
    
    # 添加数值标注
    for i in range(len(imputations)):
        for j in range(len(missing_rates)):
            val = improvement_matrix[i, j]
            text_color = 'white' if val < 40 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                   fontsize=7, color=text_color, fontweight='bold')
    
    ax.set_xlabel('Missing Rate (%)', fontweight='bold')
    ax.set_ylabel('Imputation Method', fontweight='bold')
    ax.set_title('(b) Improvement Rate Heatmap\n(% reduction in RMSE)', fontweight='bold')
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=15)

# ==================== 子图 (c): MR=40% 的 10-seed 分布箱线图 ====================
def plot_subplot_c(ax):
    """绘制种子分布箱线图"""
    seeds = sorted(data['seed'].unique())[:10]  # 取前10个种子
    
    box_data = []
    positions = []
    labels = []
    colors = []
    
    pos = 1
    for imp in ['zero', 'mean', 'iterative']:
        # Baseline 数据
        baseline_seeds = []
        mim_seeds = []
        
        for seed in seeds:
            b = baseline_data[
                (baseline_data['imputation_method'] == imp) & 
                (baseline_data['missing_rate'] == 0.4) &
                (baseline_data['seed'] == seed)
            ]['rmse'].values
            m = mim_data[
                (mim_data['imputation_method'] == imp) & 
                (mim_data['missing_rate'] == 0.4) &
                (mim_data['seed'] == seed)
            ]['rmse'].values
            
            if len(b) > 0:
                baseline_seeds.append(b[0])
            if len(m) > 0:
                mim_seeds.append(m[0])
        
        box_data.extend([baseline_seeds, mim_seeds])
        positions.extend([pos, pos+1])
        labels.extend([f'{imp.title()}\nBL', f'{imp.title()}\nMIM'])
        colors.extend(['white', COLORS[imp]])
        pos += 3
    
    # 绘制箱线图
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    # 设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('(c) 10-Seed Distribution at MR=40%\n(Baseline vs MIM)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# ==================== 子图 (d): 参数量 vs 改进率散点图 ====================
def plot_subplot_d(ax):
    """绘制参数量与改进率关系"""
    # 获取所有架构和层级的数据
    all_data = df.copy()
    
    # 计算每个配置的平均改进率
    configs = []
    for arch in ['mlp', 'cnn', 'lstm']:
        for level in ['level_1', 'level_2', 'level_3', 'level_4']:
            subset = all_data[
                (all_data['architecture'] == arch) & 
                (all_data['level'] == level)
            ]
            if len(subset) == 0:
                continue
            
            baseline = subset[subset['config_type'] == 'baseline']
            mim = subset[subset['config_type'] == 'mim']
            
            if len(baseline) == 0 or len(mim) == 0:
                continue
            
            # 计算平均改进率
            improvements = []
            for mr in [0.2, 0.4, 0.6, 0.8]:
                b = baseline[baseline['missing_rate'] == mr]['rmse'].mean()
                m = mim[mim['missing_rate'] == mr]['rmse'].mean()
                if pd.notna(b) and pd.notna(m) and b > 0:
                    improvements.append(calculate_improvement(b, m))
            
            if improvements:
                avg_improvement = np.mean(improvements)
                param_count = get_param_count(arch, level)
                configs.append({
                    'architecture': arch,
                    'level': level,
                    'param_count': param_count,
                    'improvement': avg_improvement
                })
    
    # 绘制散点图
    markers = {'mlp': 'o', 'cnn': 's', 'lstm': '^'}
    colors = {'mlp': '#e74c3c', 'cnn': '#3498db', 'lstm': '#2ecc71'}
    
    for arch in ['mlp', 'cnn', 'lstm']:
        arch_data = [c for c in configs if c['architecture'] == arch]
        if arch_data:
            x = [c['param_count'] for c in arch_data]
            y = [c['improvement'] for c in arch_data]
            ax.scatter(x, y, marker=markers[arch], s=100, c=colors[arch], 
                      label=arch.upper(), alpha=0.7, edgecolors='black', linewidth=1)
            
            # 添加层级标注
            for c in arch_data:
                ax.annotate(c['level'].replace('level_', 'L'), 
                           (c['param_count'], c['improvement']),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('Parameter Count (log scale)', fontweight='bold')
    ax.set_ylabel('Average Improvement (%)', fontweight='bold')
    ax.set_title('(d) Model Size vs Improvement\n(All Architectures)', fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

# ==================== 主函数 ====================
def main():
    """生成 Figure 1"""
    print("生成 Figure 1: MIM Core Effect...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plot_subplot_a(axes[0, 0])
    plot_subplot_b(axes[0, 1])
    plot_subplot_c(axes[1, 0])
    plot_subplot_d(axes[1, 1])
    
    plt.tight_layout()
    
    # 保存
    output_png = OUTPUT_DIR / 'figure1_mim_core_effect_v2.png'
    output_pdf = OUTPUT_DIR / 'figure1_mim_core_effect_v2.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 已保存: {output_png}")
    print(f"✅ 已保存: {output_pdf}")
    
    plt.close()

if __name__ == '__main__':
    main()
