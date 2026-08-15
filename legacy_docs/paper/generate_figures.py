# -*- coding: utf-8 -*-
"""
生成论文所需的图表
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 输出目录
output_dir = 'figures'

# =============================================================================
# 图1：工程问题与缺失数据场景示意图
# =============================================================================
def fig1_problem_scenario():
    """图1：工程问题与缺失数据场景示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：BMS架构与传感器
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('(a) 电池管理系统监测链路', fontsize=14, fontweight='bold', pad=10)
    
    # 电池包
    battery = FancyBboxPatch((1, 4), 2.5, 3, boxstyle="square,pad=0.1", 
                              facecolor='#E8F4F8', edgecolor='#2E86AB', linewidth=2)
    ax1.add_patch(battery)
    ax1.text(2.25, 6.5, '电池包', ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(2.25, 5.8, 'Cell 1~N', ha='center', va='center', fontsize=9, color='#555')
    
    # 传感器
    sensors = [
        (5, 7, '电压传感器', '#FFE4B5'),
        (5, 5.5, '电流传感器', '#E6E6FA'),
        (5, 4, '温度传感器', '#FFE4E1'),
    ]
    for x, y, name, color in sensors:
        rect = FancyBboxPatch((x-1, y-0.4), 2, 0.8, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x, y, name, ha='center', va='center', fontsize=9)
        # 箭头
        ax1.annotate('', xy=(x-1, y), xytext=(3.5, 5.5),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    # BMS
    bms = FancyBboxPatch((7.5, 4.5), 2, 2.5, boxstyle="round,pad=0.1",
                          facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax1.add_patch(bms)
    ax1.text(8.5, 6.2, 'BMS', ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(8.5, 5.5, '数据采集', ha='center', va='center', fontsize=9)
    ax1.text(8.5, 5, '与处理', ha='center', va='center', fontsize=9)
    
    # 箭头
    for y in [7, 5.5, 4]:
        ax1.annotate('', xy=(7.5, 5.75), xytext=(6, y),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    # 故障标注
    ax1.text(5, 2.5, '潜在故障：传感器失效 | 通信中断 | 数据丢包', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='#FFC107', linewidth=1.5))
    
    # 右图：特征矩阵与缺失
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) 循环级特征矩阵与MCAR缺失', fontsize=14, fontweight='bold', pad=10)
    
    # 特征矩阵
    n_rows, n_cols = 8, 6
    cell_size = 0.7
    start_x, start_y = 2, 7
    
    # 列标签
    features = ['v_mean', 'v_std', 'c_mean', 'c_std', 'T_max', '...']
    for j, feat in enumerate(features):
        ax2.text(start_x + j*cell_size + cell_size/2, start_y + 0.5, feat, 
                ha='center', va='center', fontsize=8, rotation=45)
    
    # 行标签
    for i in range(n_rows):
        ax2.text(start_x - 0.8, start_y - i*cell_size - cell_size/2, f'Cycle {i+1}', 
                ha='center', va='center', fontsize=8)
    
    # 矩阵格子
    np.random.seed(42)
    for i in range(n_rows):
        for j in range(n_cols):
            # 模拟MCAR缺失（约30%缺失）
            is_missing = np.random.random() < 0.3
            if is_missing:
                color = '#D3D3D3'  # 灰色表示缺失
                text = 'NA'
                text_color = '#666'
            else:
                color = '#E8F4F8'  # 浅蓝表示有值
                text = f'{np.random.uniform(3,4):.1f}'
                text_color = '#333'
            
            rect = Rectangle((start_x + j*cell_size, start_y - (i+1)*cell_size), 
                            cell_size-0.05, cell_size-0.05,
                            facecolor=color, edgecolor='#666', linewidth=0.5)
            ax2.add_patch(rect)
            ax2.text(start_x + j*cell_size + cell_size/2, 
                    start_y - i*cell_size - cell_size/2, 
                    text, ha='center', va='center', fontsize=7, color=text_color)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#E8F4F8', edgecolor='#666', label='观测值'),
        mpatches.Patch(facecolor='#D3D3D3', edgecolor='#666', label='MCAR缺失'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # 说明文字
    ax2.text(5, 1.5, '16维特征向量 × N个循环', ha='center', va='center', fontsize=10)
    ax2.text(5, 0.8, '缺失率 MR = 缺失元素数 / 总元素数', ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#6C757D'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_problem_scenario.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('图1已保存: fig1_problem_scenario.png')


# =============================================================================
# 图2：Baseline vs MIM输入对比
# =============================================================================
def fig2_baseline_vs_mim():
    """图2：Baseline vs MIM输入对比示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('(a) Baseline：仅插补', fontsize=14, fontweight='bold', pad=10)
    
    # 输入矩阵 X̂
    draw_matrix(ax1, 2, 6, 6, 1.2, 'X̂ (插补后特征)', '#E8F4F8', 
                'n×16', '插补值填充缺失')
    
    # 箭头
    ax1.annotate('', xy=(5, 4.5), xytext=(5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 模型
    model_box = FancyBboxPatch((3, 2.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(5, 3.5, 'f(·)', ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.text(5, 2.9, '神经网络模型', ha='center', va='center', fontsize=10)
    
    # 箭头
    ax1.annotate('', xy=(5, 1.8), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 输出
    draw_matrix(ax1, 3.5, 0.5, 3, 0.8, 'ŷ (SOH预测)', '#FFE4B5', 'n×1', '')
    
    # 维度标注
    ax1.text(8.5, 6.6, '输入维度: 16', ha='left', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='#FFC107'))
    
    # MIM
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) MIM：插补 + 缺失指示器', fontsize=14, fontweight='bold', pad=10)
    
    # 输入矩阵 X̂
    draw_matrix(ax2, 1, 6.5, 3.5, 1, 'X̂ (插补后)', '#E8F4F8', 'n×16', '')
    
    # 缺失指示器 M
    draw_matrix(ax2, 5, 6.5, 3.5, 1, 'M (指示器)', '#FFE4E1', 'n×16', '0=观测,1=缺失')
    
    # 拼接箭头
    ax2.annotate('', xy=(5, 7), xytext=(4.5, 7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax2.text(4.75, 7.3, '拼接', ha='center', va='center', fontsize=9)
    
    # 合并矩阵 Z
    draw_matrix(ax2, 2.5, 4.5, 5, 1, 'Z = [X̂ | M]', '#E6E6FA', 'n×32', '扩展特征')
    
    # 箭头
    ax2.annotate('', xy=(5, 3.3), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 模型
    model_box2 = FancyBboxPatch((3, 1.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax2.add_patch(model_box2)
    ax2.text(5, 2.5, 'f(·)', ha='center', va='center', fontsize=20, fontweight='bold')
    ax2.text(5, 1.9, '神经网络模型', ha='center', va='center', fontsize=10)
    
    # 箭头
    ax2.annotate('', xy=(5, 0.8), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 输出
    draw_matrix(ax2, 3.5, -0.3, 3, 0.8, 'ŷ (SOH预测)', '#FFE4B5', 'n×1', '')
    
    # 维度标注
    ax2.text(8.5, 6.8, '输入维度: 32', ha='left', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='#FFC107'))
    ax2.text(8.5, 4.8, '特征翻倍', ha='left', va='center', fontsize=9, color='#28A745')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_baseline_vs_mim.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图2已保存: fig2_baseline_vs_mim.png')


def draw_matrix(ax, x, y, w, h, label, color, dim, note):
    """绘制矩阵框"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(x + w/2, y + h/2 - 0.25, dim, ha='center', va='center', fontsize=9, color='#666')
    if note:
        ax.text(x + w/2, y - 0.3, note, ha='center', va='center', fontsize=8, color='#666')


# =============================================================================
# 图3：四种模型架构对比
# =============================================================================
def fig3_model_architectures():
    """图3：四种模型架构对比图"""
    fig = plt.figure(figsize=(16, 10))
    
    models = [
        ('MLP', [
            ('Input', 16, '#E8F4F8'),
            ('FC(192)', 192, '#E6E6FA'),
            ('FC(96)', 96, '#E6E6FA'),
            ('FC(48)', 48, '#E6E6FA'),
            ('FC(24)', 24, '#E6E6FA'),
            ('Output', 1, '#FFE4B5'),
        ], '(a) MLP: 全连接网络'),
        ('LSTM', [
            ('Input\n(seq=5)', '16×5', '#E8F4F8'),
            ('LSTM(96)', 96, '#E6E6FA'),
            ('LSTM(48)', 48, '#E6E6FA'),
            ('FC', 24, '#D4EDDA'),
            ('Output', 1, '#FFE4B5'),
        ], '(b) LSTM: 长短期记忆网络'),
        ('GRU', [
            ('Input\n(seq=5)', '16×5', '#E8F4F8'),
            ('GRU(88)', 88, '#E6E6FA'),
            ('GRU(44)', 44, '#E6E6FA'),
            ('FC', 24, '#D4EDDA'),
            ('Output', 1, '#FFE4B5'),
        ], '(c) GRU: 门控循环单元'),
        ('1D-CNN', [
            ('Input', 16, '#E8F4F8'),
            ('Conv1d\n(16→72)', 'k=7', '#E6E6FA'),
            ('Conv1d\n(72→32)', 'k=3', '#E6E6FA'),
            ('Adaptive\nAvgPool', '-', '#D4EDDA'),
            ('FC', 24, '#D4EDDA'),
            ('Output', 1, '#FFE4B5'),
        ], '(d) 1D-CNN: 一维卷积网络'),
    ]
    
    for idx, (name, layers, title) in enumerate(models):
        ax = plt.subplot(2, 2, idx+1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(layers) + 1)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        for i, (layer_name, size, color) in enumerate(layers):
            y = len(layers) - i
            # 层框
            width = 3 if 'Conv' in layer_name else 2.5
            rect = FancyBboxPatch((5-width/2, y-0.4), width, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='#333', linewidth=1.5)
            ax.add_patch(rect)
            
            # 层名
            ax.text(5, y+0.1, layer_name, ha='center', va='center',
                   fontsize=9, fontweight='bold')
            # 尺寸
            ax.text(5, y-0.2, str(size), ha='center', va='center',
                   fontsize=8, color='#666')
            
            # 连接箭头
            if i < len(layers) - 1:
                ax.annotate('', xy=(5, y-0.5), xytext=(5, y-0.9),
                           arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_model_architectures.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图3已保存: fig3_model_architectures.png')


# =============================================================================
# 图4：多缺失率联合训练流程
# =============================================================================
def fig4_joint_training():
    """图4：多缺失率联合训练流程"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('多缺失率联合训练流程 (Algorithm 2)', fontsize=16, fontweight='bold', pad=20)
    
    # 原始训练集
    draw_box(ax, 1, 7, 2.5, 1.2, '$D_{train}$\n(原始训练集)', '#E8F4F8')
    
    # 10个缺失率分支
    mr_values = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(mr_values)))
    
    for i, (mr, color) in enumerate(zip(mr_values, colors)):
        x = 4.5 + (i % 5) * 1.8
        y = 7.5 if i < 5 else 5.5
        
        # 分支框
        rect = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y+0.15, f'MR={mr}', ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, y-0.2, 'MIM构造', ha='center', va='center', fontsize=7)
        
        # 从原始数据集到分支的箭头
        if i < 5:
            ax.annotate('', xy=(x-0.7, y), xytext=(3.5, 7.6),
                       arrowprops=dict(arrowstyle='->', color='#666', lw=1, 
                                      connectionstyle='arc3,rad=0.1'))
        else:
            ax.annotate('', xy=(x-0.7, y), xytext=(3.5, 7.4),
                       arrowprops=dict(arrowstyle='->', color='#666', lw=1,
                                      connectionstyle='arc3,rad=-0.1'))
    
    # 标注
    ax.text(9, 6.8, '10个缺失率副本', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='#FFC107'))
    
    # 合并
    draw_box(ax, 6, 3.5, 3, 1, '$D_{augmented}$\n(合并数据集)', '#E6E6FA')
    
    # 合并箭头
    for i in range(5):
        x = 4.5 + i * 1.8
        ax.annotate('', xy=(7.5, 4.5), xytext=(x, 5),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1))
    for i in range(5):
        x = 4.5 + i * 1.8
        ax.annotate('', xy=(7.5, 4.5), xytext=(x, 5.5),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1))
    
    ax.text(5, 4.8, '合并', ha='center', va='center', fontsize=9, color='#666')
    
    # 统一模型
    draw_box(ax, 10.5, 3.5, 3, 1, '$f_{MIM}$\n(统一MIM模型)', '#D4EDDA')
    ax.annotate('', xy=(10.5, 4), xytext=(9, 4),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 输出示例
    ax.text(12, 2.5, '输入: 任意MR的数据', ha='center', va='center', fontsize=9)
    ax.text(12, 2, '输出: SOH预测', ha='center', va='center', fontsize=9)
    
    # 优势说明
    advantages = [
        '✓ 单模型处理所有缺失率',
        '✓ 减少模型存储开销', 
        '✓ 提升泛化能力',
        '✓ 简化部署流程'
    ]
    for i, adv in enumerate(advantages):
        ax.text(1, 2.5 - i*0.5, adv, ha='left', va='center', fontsize=9,
               color='#28A745', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_joint_training.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图4已保存: fig4_joint_training.png')


def draw_box(ax, x, y, w, h, text, color):
    """绘制带文字的框"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')


# =============================================================================
# 图5：重绘MAE热力图（带数值标注）
# =============================================================================
def fig5_heatmap():
    """图5：MAE热力图（高分辨率+数值标注）"""
    # 构造数据
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 从表7提取的数据（MAE均值）
    data = {
        'MLP-Baseline': [0.089, 0.120, 0.145, 0.171, 0.189, 0.218, 0.256, 0.297, 0.354],
        'MLP-MIM': [0.036, 0.036, 0.034, 0.031, 0.029, 0.026, 0.024, 0.022, 0.020],
        'LSTM-Baseline': [0.060, 0.082, 0.108, 0.122, 0.137, 0.154, 0.176, 0.196, 0.216],
        'LSTM-MIM': [0.019, 0.018, 0.017, 0.016, 0.015, 0.015, 0.016, 0.016, 0.017],
        'GRU-Baseline': [0.044, 0.063, 0.079, 0.089, 0.098, 0.109, 0.121, 0.128, 0.131],
        'GRU-MIM': [0.018, 0.017, 0.016, 0.015, 0.014, 0.014, 0.016, 0.017, 0.017],
        '1D-CNN-Baseline': [0.151, 0.225, 0.287, 0.323, 0.357, 0.380, 0.389, 0.381, 0.371],
        '1D-CNN-MIM': [0.022, 0.018, 0.016, 0.014, 0.014, 0.014, 0.015, 0.017, 0.020],
    }
    
    df = pd.DataFrame(data, index=missing_rates)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'MAE (越低越好)'}, 
                linewidths=0.5, ax=ax, vmin=0, vmax=0.4)
    
    ax.set_xlabel('缺失率 (MR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('模型', fontsize=12, fontweight='bold')
    ax.set_title('不同缺失率下各模型的MAE热力图', fontsize=14, fontweight='bold', pad=15)
    
    # 旋转y轴标签
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_mae_heatmap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图5已保存: fig5_mae_heatmap.png')


# =============================================================================
# 图6：模型选择决策图
# =============================================================================
def fig6_model_selection():
    """图6：基于MR的模型选择决策图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('基于缺失率的模型选择建议', fontsize=16, fontweight='bold', pad=20)
    
    # MR轴
    ax.annotate('', xy=(13, 1), xytext=(1, 1),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(13.2, 1, 'MR', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # MR刻度
    for mr in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = 1 + mr * 11
        ax.plot([x, x], [0.9, 1.1], 'k-', lw=1)
        ax.text(x, 0.6, f'{mr:.1f}', ha='center', va='center', fontsize=9)
    
    # 区域1：低缺失率
    rect1 = FancyBboxPatch((1, 2.5), 3.3, 2, boxstyle="round,pad=0.1",
                           facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.65, 3.8, 'MR ≤ 0.3', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.65, 3.3, '基线模型足够', ha='center', va='center', fontsize=10)
    ax.text(2.65, 2.9, '推荐: 简单MLP', ha='center', va='center', fontsize=9, color='#666')
    
    # 区域2：中等缺失率
    rect2 = FancyBboxPatch((4.3, 2.5), 3.3, 2, boxstyle="round,pad=0.1",
                           facecolor='#FFF3CD', edgecolor='#FFC107', linewidth=2)
    ax.add_patch(rect2)
    ax.text(6, 3.8, '0.3 < MR ≤ 0.6', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 3.3, '推荐MIM方案', ha='center', va='center', fontsize=10)
    ax.text(6, 2.9, 'MLP-MIM / GRU-MIM', ha='center', va='center', fontsize=9, color='#666')
    ax.text(6, 2.5, '改进率: 75%-86%', ha='center', va='center', fontsize=8, color='#28A745')
    
    # 区域3：高缺失率
    rect3 = FancyBboxPatch((7.6, 2.5), 4.4, 2, boxstyle="round,pad=0.1",
                           facecolor='#F8D7DA', edgecolor='#DC3545', linewidth=2)
    ax.add_patch(rect3)
    ax.text(9.8, 3.8, 'MR ≥ 0.7', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9.8, 3.3, '强烈推荐1D-CNN-MIM', ha='center', va='center', fontsize=10)
    ax.text(9.8, 2.9, '最佳性能-参数量权衡', ha='center', va='center', fontsize=9, color='#666')
    ax.text(9.8, 2.5, '改进率 > 90%', ha='center', va='center', fontsize=8, color='#28A745')
    
    # 关键数据点
    key_points = [
        (1 + 0.5*11, 5, 'MR=0.5\n1D-CNN-MIM\nMAE: 0.014'),
        (1 + 0.9*11, 5, 'MR=0.9\n1D-CNN-MIM\nMAE: 0.020'),
    ]
    for x, y, text in key_points:
        ax.annotate(text, xy=(x, 3.5), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.5),
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333'))
    
    # 底部说明
    ax.text(7, 0.2, '注：基于100随机种子实验结果，95%置信区间', 
           ha='center', va='center', fontsize=9, color='#666', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_model_selection.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图6已保存: fig6_model_selection.png')


# =============================================================================
# 图7：性能-参数量权衡散点图
# =============================================================================
def fig7_performance_params():
    """图7：性能-参数量权衡散点图"""
    # 数据：MR=0.5时的MAE和参数量
    models_data = [
        ('MLP-Baseline', 15000, 0.189, '#1f77b4'),
        ('MLP-MIM', 30000, 0.029, '#1f77b4'),
        ('LSTM-Baseline', 42000, 0.137, '#ff7f0e'),
        ('LSTM-MIM', 44000, 0.015, '#ff7f0e'),
        ('GRU-Baseline', 40000, 0.098, '#2ca02c'),
        ('GRU-MIM', 44000, 0.014, '#2ca02c'),
        ('1D-CNN-Baseline', 16700, 0.357, '#d62728'),
        ('1D-CNN-MIM', 21300, 0.014, '#d62728'),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制散点
    for name, params, mae, color in models_data:
        marker = 'o' if 'MIM' in name else 'x'
        size = 200 if 'MIM' in name else 150
        edge_color = 'white' if 'MIM' in name else color
        linewidth = 2 if 'MIM' in name else 1.5
        
        ax.scatter(params, mae, s=size, c=color, marker=marker, 
                  edgecolors=edge_color, linewidths=linewidth, alpha=0.8, zorder=5)
        
        # 添加标签
        offset_y = 0.015 if 'MIM' in name else -0.02
        ax.annotate(name.replace('-', '\n'), (params, mae), 
                   textcoords='offset points', xytext=(0, 10 if 'MIM' in name else -15),
                   ha='center', va='bottom' if 'MIM' in name else 'top', fontsize=8)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label='MIM策略'),
        plt.Line2D([0], [0], marker='x', color='gray', markerfacecolor='gray',
                  markersize=10, label='Baseline'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4',
                  markersize=8, label='MLP'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e',
                  markersize=8, label='LSTM'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c',
                  markersize=8, label='GRU'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
                  markersize=8, label='1D-CNN'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    
    ax.set_xlabel('参数量', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (MR=0.5)', fontsize=12, fontweight='bold')
    ax.set_title('性能-参数量权衡分析 (MR=0.5)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(10000, 50000)
    ax.set_ylim(0, 0.4)
    ax.grid(True, alpha=0.3)
    
    # 添加帕累托前沿注释
    ax.annotate('帕累托前沿', xy=(21300, 0.014), xytext=(30000, 0.08),
               arrowprops=dict(arrowstyle='->', color='#28A745', lw=2),
               fontsize=10, color='#28A745', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_performance_params.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('图7已保存: fig7_performance_params.png')


# =============================================================================
# 主函数
# =============================================================================
if __name__ == '__main__':
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*60)
    print('开始生成论文图表...')
    print('='*60)
    
    fig1_problem_scenario()
    fig2_baseline_vs_mim()
    fig3_model_architectures()
    fig4_joint_training()
    fig5_heatmap()
    fig6_model_selection()
    fig7_performance_params()
    
    print('='*60)
    print('所有图表生成完成！')
    print(f'输出目录: {output_dir}/')
    print('='*60)
