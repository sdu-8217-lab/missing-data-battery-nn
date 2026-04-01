"""
Figure 3: 插补质量梯度效应（论文核心发现）
展示 MIM 效果随插补质量提升而递减的梯度效应：Zero > Mean > Iterative

数据来源: results/plotting_data_v0.5.csv
作者: Kimi
日期: 2026-03-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ==================== 配置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# 配色方案
COLORS = {
    'zero': '#e74c3c',      # 红 - 低质量
    'mean': '#f39c12',      # 橙 - 中质量
    'iterative': '#3498db', # 蓝 - 高质量
}

OUTPUT_DIR = Path(__file__).parent.parent

# ==================== 数据加载 ====================
print("加载数据...")
df = pd.read_csv('results/plotting_data_v0.5.csv')

# 使用 2C MLP 作为主数据
data = df[(df['batch'] == '2C') & (df['architecture'] == 'mlp')].copy()
baseline_data = data[data['config_type'] == 'baseline']
mim_data = data[data['config_type'] == 'mim']

print(f"数据加载完成")

# ==================== 计算改进率 ====================
def calculate_improvement(baseline_val, mim_val):
    """计算百分比改进率"""
    if pd.isna(baseline_val) or pd.isna(mim_val) or baseline_val == 0:
        return 0
    return ((baseline_val - mim_val) / baseline_val * 100)

# ==================== 子图 (a): 改进率梯度曲线 ====================
def plot_subplot_a(ax):
    """三种插补的改进率 vs MR 曲线叠加"""
    missing_rates = sorted(data['missing_rate'].unique())
    
    for imp in ['zero', 'mean', 'iterative']:
        improvements = []
        
        for mr in missing_rates:
            b = baseline_data[
                (baseline_data['imputation_method'] == imp) & 
                (baseline_data['missing_rate'] == mr)
            ]['rmse'].mean()
            m = mim_data[
                (mim_data['imputation_method'] == imp) & 
                (mim_data['missing_rate'] == mr)
            ]['rmse'].mean()
            
            imp_val = calculate_improvement(b, m)
            improvements.append(imp_val)
        
        mr_pct = [mr * 100 for mr in missing_rates]
        
        # 确定线宽和样式
        if imp == 'zero':
            linewidth = 3.0
            label = 'Zero (Low Quality)'
        elif imp == 'mean':
            linewidth = 2.5
            label = 'Mean (Medium Quality)'
        else:
            linewidth = 2.0
            label = 'Iterative (High Quality)'
        
        ax.plot(mr_pct, improvements, '-', color=COLORS[imp], 
                linewidth=linewidth, label=label, marker='o', markersize=6)
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 添加标注
    ax.annotate('Higher improvement\nwith simpler imputation', 
                xy=(50, 70), xytext=(25, 78),
                fontsize=10, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
    
    ax.annotate('Lower improvement\nwith better imputation', 
                xy=(55, 45), xytext=(70, 30),
                fontsize=10, color='#2980b9', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))
    
    ax.set_xlabel('Missing Rate (%)', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('(a) MIM Effectiveness Gradient\n(Zero > Mean > Iterative)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-5, 85)

# ==================== 子图 (b): 固定 MR=40% 的对比 ====================
def plot_subplot_b(ax):
    """固定 MR=40%，三种插补在三个缺失机制下的改进率"""
    imputations = ['zero', 'mean', 'iterative']
    modes = ['MCAR', 'MAR', 'MNAR']
    
    x = np.arange(len(modes))
    width = 0.25
    
    for i, imp in enumerate(imputations):
        improvements = []
        for mode in modes:
            b = baseline_data[
                (baseline_data['imputation_method'] == imp) & 
                (baseline_data['missing_mode'] == mode) &
                (baseline_data['missing_rate'] == 0.4)
            ]['rmse'].mean()
            m = mim_data[
                (mim_data['imputation_method'] == imp) & 
                (mim_data['missing_mode'] == mode) &
                (mim_data['missing_rate'] == 0.4)
            ]['rmse'].mean()
            
            imp_val = calculate_improvement(b, m)
            improvements.append(imp_val)
        
        bars = ax.bar(x + i*width, improvements, width, 
                      label=imp.title(), color=COLORS[imp], alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # 添加数值标注
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=9,
                   fontweight='bold')
    
    ax.set_xlabel('Missing Mechanism', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('(b) Improvement by Mechanism\n(at MR=40%)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(modes)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 80)

# ==================== 子图 (c): 插补质量 × 缺失率 热力图 ====================
def plot_subplot_c(ax):
    """二维热力图展示插补质量和缺失率的交互效应"""
    imputations = ['zero', 'mean', 'iterative']
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
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
    
    # 使用暖色调 colormap，越暖=改进越大
    im = ax.imshow(improvement_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=75)
    
    # 设置刻度
    ax.set_xticks(range(len(missing_rates)))
    ax.set_xticklabels([f'{int(mr*100)}' for mr in missing_rates])
    ax.set_yticks(range(len(imputations)))
    ax.set_yticklabels(['Zero\n(Low)', 'Mean\n(Medium)', 'Iterative\n(High)'])
    
    # 添加数值标注
    for i in range(len(imputations)):
        for j in range(len(missing_rates)):
            val = improvement_matrix[i, j]
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                   fontsize=11, color=text_color, fontweight='bold')
    
    ax.set_xlabel('Missing Rate (%)', fontweight='bold')
    ax.set_ylabel('Imputation Quality', fontweight='bold')
    ax.set_title('(c) Quality × Missing Rate Interaction\n(Warmer = Better MIM Effect)', 
                 fontweight='bold')
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=18, fontweight='bold')

# ==================== 主函数 ====================
def main():
    """生成 Figure 3"""
    print("生成 Figure 3: Imputation Quality Gradient Effect...")
    
    fig = plt.figure(figsize=(16, 5))
    
    # 创建三个子图
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    plot_subplot_a(ax1)
    plot_subplot_b(ax2)
    plot_subplot_c(ax3)
    

    
    # 保存
    output_png = OUTPUT_DIR / 'imputation_gradient.png'
    output_pdf = OUTPUT_DIR / 'imputation_gradient.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 已保存: {output_png}")
    print(f"✅ 已保存: {output_pdf}")
    
    plt.close()

if __name__ == '__main__':
    main()
