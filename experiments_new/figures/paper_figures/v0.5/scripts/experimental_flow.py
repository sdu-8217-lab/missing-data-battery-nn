"""
Figure 6: 实验流程概览图
展示完整的 9 层实验架构（L1-L9），帮助读者理解实验设计

作者: Kimi
日期: 2026-03-31
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ==================== 配置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path(__file__).parent.parent

# ==================== 颜色配置 ====================
COLORS = {
    'seed': '#3498db',      # 蓝
    'batch': '#9b59b6',     # 紫
    'arch': '#e74c3c',      # 红
    'level': '#e67e22',     # 橙
    'mim': '#2ecc71',       # 绿
    'test': '#1abc9c',      # 青
}

# ==================== 主函数 ====================
def main():
    """生成实验流程图"""
    print("生成 Figure 6: Experimental Flow Diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ==================== 标题 ====================
    ax.text(8, 9.5, 'Figure 6: Complete Experimental Architecture (L1-L9)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # ==================== 训练阶段 ====================
    ax.text(4, 8.8, 'TRAINING PHASE', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    
    # L1: Seeds
    y_pos = 7.5
    for i, seed in enumerate([0, 42, 123]):
        x = 2 + i * 2.5
        box = FancyBboxPatch((x-0.4, y_pos-0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.05", 
                             facecolor=COLORS['seed'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, f'Seed\n{seed}', ha='center', va='center', 
               fontsize=8, color='white', fontweight='bold')
    
    ax.text(5.5, y_pos+0.8, 'L1: Random Seeds (10 total)', ha='center', fontsize=9, 
           fontweight='bold', color=COLORS['seed'])
    
    # L2: Batches
    y_pos = 6.0
    batches = ['2C', '3C', 'R2.5']
    for i, batch in enumerate(batches):
        x = 2 + i * 3
        box = FancyBboxPatch((x-0.5, y_pos-0.3), 1, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=COLORS['batch'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, batch, ha='center', va='center', 
               fontsize=10, color='white', fontweight='bold')
    
    ax.text(5.5, y_pos+0.8, 'L2: Battery Batches (3 datasets)', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['batch'])
    
    # L3: Architectures
    y_pos = 4.5
    archs = ['MLP', 'CNN', 'LSTM']
    for i, arch in enumerate(archs):
        x = 2 + i * 3
        box = FancyBboxPatch((x-0.5, y_pos-0.3), 1, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=COLORS['arch'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, arch, ha='center', va='center', 
               fontsize=10, color='white', fontweight='bold')
    
    ax.text(5.5, y_pos+0.8, 'L3: Neural Architectures (3 types)', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['arch'])
    
    # L4: Levels
    y_pos = 3.0
    levels = ['L1\n(~6K)', 'L2\n(~12K)', 'L3\n(~21K)', 'L4\n(~45K)']
    for i, level in enumerate(levels):
        x = 1.5 + i * 2
        box = FancyBboxPatch((x-0.5, y_pos-0.35), 1, 0.7,
                             boxstyle="round,pad=0.05",
                             facecolor=COLORS['level'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, level, ha='center', va='center', 
               fontsize=8, color='white', fontweight='bold')
    
    ax.text(4.5, y_pos+0.9, 'L4: Model Levels (4 scales)', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['level'])
    
    # L5: MIM
    y_pos = 1.5
    configs = ['Baseline\n(No MIM)', 'MIM\n(With Indicator)']
    for i, config in enumerate(configs):
        x = 3 + i * 4
        color = '#95a5a6' if i == 0 else COLORS['mim']
        box = FancyBboxPatch((x-0.8, y_pos-0.4), 1.6, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y_pos, config, ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
    
    ax.text(5.5, y_pos+0.9, 'L5: MIM Configuration (2 variants)', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['mim'])
    
    # ==================== 分隔线 ====================
    ax.plot([8, 8], [0.5, 8.5], 'k--', linewidth=2, alpha=0.5)
    ax.text(8, 8.7, '→', ha='center', fontsize=20, fontweight='bold')
    
    # ==================== 测试阶段 ====================
    ax.text(12, 8.8, 'TESTING PHASE', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    
    # L6: Missing Mechanisms
    y_pos = 7.5
    mechanisms = ['MCAR', 'MAR', 'MNAR']
    for i, mech in enumerate(mechanisms):
        x = 10 + i * 2
        box = FancyBboxPatch((x-0.6, y_pos-0.3), 1.2, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=COLORS['test'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, mech, ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
    
    ax.text(12, y_pos+0.8, 'L6: Missing Mechanisms (3 types)', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['test'])
    
    # L7: Missing Rates
    y_pos = 6.0
    ax.text(12, y_pos, 'MR = 0%, 5%, 10%, ..., 95%\n(20 levels)', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='#d5dbdb', edgecolor='black', linewidth=1.5))
    ax.text(12, y_pos+0.8, 'L7: Missing Rate', ha='center', fontsize=9,
           fontweight='bold', color='#2c3e50')
    
    # L8: Imputation Methods
    y_pos = 4.5
    imputations = ['Zero', 'Mean', 'Iterative']
    for i, imp in enumerate(imputations):
        x = 10 + i * 2
        colors = ['#e74c3c', '#f39c12', '#3498db']
        box = FancyBboxPatch((x-0.6, y_pos-0.3), 1.2, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, imp, ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
    
    ax.text(12, y_pos+0.8, 'L8: Imputation Methods (3 types)', ha='center', fontsize=9,
           fontweight='bold', color='#2c3e50')
    
    # L9: Metrics
    y_pos = 3.0
    metrics = ['RMSE', 'MAE', 'R²']
    for i, metric in enumerate(metrics):
        x = 10.5 + i * 1.5
        box = FancyBboxPatch((x-0.5, y_pos-0.25), 1, 0.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#34495e', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_pos, metric, ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
    
    ax.text(12, y_pos+0.7, 'L9: Evaluation Metrics', ha='center', fontsize=9,
           fontweight='bold', color='#34495e')
    
    # ==================== 统计信息 ====================
    y_pos = 1.2
    total_models = 10 * 3 * 3 * 4 * 2  # seeds × batches × archs × levels × configs
    total_tests = total_models * 3 * 20 * 3  # × mechs × MRs × imputations
    
    stats_text = f'Total Models Trained: {total_models:,}\nTotal Test Conditions: {total_tests:,}'
    ax.text(12, y_pos, stats_text, ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2),
           fontweight='bold', color='#c0392b')
    
    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(color=COLORS['seed'], label='L1: Seeds'),
        mpatches.Patch(color=COLORS['batch'], label='L2: Batches'),
        mpatches.Patch(color=COLORS['arch'], label='L3: Architectures'),
        mpatches.Patch(color=COLORS['level'], label='L4: Model Levels'),
        mpatches.Patch(color=COLORS['mim'], label='L5: MIM Config'),
        mpatches.Patch(color=COLORS['test'], label='L6-L9: Test Config'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)
    
    # ==================== 保存 ====================
    output_png = OUTPUT_DIR / 'experimental_flow.png'
    output_pdf = OUTPUT_DIR / 'experimental_flow.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 已保存: {output_png}")
    print(f"✅ 已保存: {output_pdf}")
    
    plt.close()

if __name__ == '__main__':
    main()
