#!/usr/bin/env python3
"""
生成图12：基于缺失率的模型选择建议
修正数值，使用表6中的实际改进率
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_model_selection_diagram(output_path):
    """
    创建模型选择决策图
    使用表6中的实际改进率数据
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色定义
    color_low = '#2ECC71'      # 绿色 - 低缺失率
    color_mid = '#F39C12'      # 橙色 - 中等缺失率
    color_high = '#E74C3C'     # 红色 - 高缺失率
    
    # 标题
    ax.text(5, 9.5, 'Model Selection Guide Based on Missing Rate', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 9.0, 'Using Actual Improvement Rates from Table 6', 
            fontsize=12, ha='center', style='italic', color='gray')
    
    # ========== 低缺失率区 (MR ≤ 0.3) ==========
    box_low = FancyBboxPatch((0.5, 6.5), 2.5, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_low, alpha=0.3, edgecolor=color_low, linewidth=2)
    ax.add_patch(box_low)
    ax.text(1.75, 8.0, 'Low MR', fontsize=14, fontweight='bold', ha='center', color=color_low)
    ax.text(1.75, 7.6, 'MR ≤ 0.3', fontsize=12, ha='center')
    ax.text(1.75, 7.2, 'Baseline OK', fontsize=11, ha='center')
    ax.text(1.75, 6.85, 'MLP-Baseline', fontsize=10, ha='center', color='darkgreen')
    
    # 低缺失率MIM改进
    ax.text(1.75, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(1.75, 6.2, 'MLP: 44.4% | 1D-CNN: 45.2%', fontsize=8, ha='center', color='gray')
    
    # ========== 中等缺失率区 (0.3 < MR ≤ 0.6) ==========
    box_mid = FancyBboxPatch((3.8, 6.5), 2.5, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_mid, alpha=0.3, edgecolor=color_mid, linewidth=2)
    ax.add_patch(box_mid)
    ax.text(5.05, 8.0, 'Medium MR', fontsize=14, fontweight='bold', ha='center', color=color_mid)
    ax.text(5.05, 7.6, '0.3 < MR ≤ 0.6', fontsize=12, ha='center')
    ax.text(5.05, 7.2, 'Recommend MIM', fontsize=11, ha='center')
    ax.text(5.05, 6.85, '1D-CNN-MIM / MLP-MIM', fontsize=10, ha='center', color='darkorange')
    
    # 中等缺失率MIM改进
    ax.text(5.05, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(5.05, 6.2, 'MLP: 46.8% | 1D-CNN: 55.3%', fontsize=8, ha='center', color='gray')
    
    # ========== 高缺失率区 (MR ≥ 0.7) ==========
    box_high = FancyBboxPatch((7.1, 6.5), 2.5, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_high, alpha=0.3, edgecolor=color_high, linewidth=2)
    ax.add_patch(box_high)
    ax.text(8.35, 8.0, 'High MR', fontsize=14, fontweight='bold', ha='center', color=color_high)
    ax.text(8.35, 7.6, 'MR ≥ 0.7', fontsize=12, ha='center')
    ax.text(8.35, 7.2, 'Strongly Recommend MIM', fontsize=11, ha='center')
    ax.text(8.35, 6.85, '1D-CNN-MIM / LSTM-MIM', fontsize=10, ha='center', color='darkred')
    
    # 高缺失率MIM改进
    ax.text(8.35, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(8.35, 6.2, '1D-CNN: 57.0% | LSTM: 38.0%', fontsize=8, ha='center', color='gray')
    
    # ========== 详细模型对比表 ==========
    table_data = [
        ['Model', 'MR=0.3', 'MR=0.5', 'MR=0.7', 'MR=0.9', 'Avg'],
        ['MLP-MIM', '46.5%', '46.8%', '42.4%', '27.9%', '42.5%'],
        ['LSTM-MIM', '17.8%', '29.0%', '38.0%', '35.3%', '26.5%'],
        ['GRU-MIM', '20.6%', '30.4%', '37.4%', '31.4%', '26.9%'],
        ['1D-CNN-MIM', '51.1%', '55.3%', '57.0%', '51.0%', '52.6%'],
    ]
    
    # 绘制表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.15, 0.15, 0.7, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置第一列样式
    for i in range(1, 5):
        table[(i, 0)].set_facecolor('#ECF0F1')
        table[(i, 0)].set_text_props(weight='bold')
    
    # 高亮最高改进率
    for i in range(1, 5):
        for j in range(1, 6):
            val = float(table_data[i][j].rstrip('%'))
            if val >= 50:
                table[(i, j)].set_facecolor('#D5F4E6')
    
    ax.text(5, 3.8, 'MIM Improvement Rates by Model and Missing Rate', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 3.4, '(Green cells: improvement ≥ 50%)', 
            fontsize=9, ha='center', color='gray', style='italic')
    
    # 添加决策箭头
    arrow1 = FancyArrowPatch((3.2, 7.5), (3.6, 7.5), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='gray')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((6.5, 7.5), (6.9, 7.5), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='gray')
    ax.add_patch(arrow2)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    output_dir = 'my_figures'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'fig12_model_selection_v2.png')
    create_model_selection_diagram(output_path)
    print("\nDone!")

if __name__ == '__main__':
    main()
