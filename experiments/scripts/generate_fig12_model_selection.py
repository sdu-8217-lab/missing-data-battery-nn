# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚鍥?2锛氬熀浜庣己澶辩巼鐨勬ā鍨嬮€夋嫨寤鸿
淇鏁板€硷紝浣跨敤琛?涓殑瀹為檯鏀硅繘鐜?
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_model_selection_diagram(output_path):
    """
    鍒涘缓妯″瀷閫夋嫨鍐崇瓥鍥?
    浣跨敤琛?涓殑瀹為檯鏀硅繘鐜囨暟鎹?
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 棰滆壊瀹氫箟
    color_low = '#2ECC71'      # 缁胯壊 - 浣庣己澶辩巼
    color_mid = '#F39C12'      # 姗欒壊 - 涓瓑缂哄け鐜?
    color_high = '#E74C3C'     # 绾㈣壊 - 楂樼己澶辩巼
    
    # 鏍囬
    ax.text(5, 9.5, 'Model Selection Guide Based on Missing Rate', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 9.0, 'Using Actual Improvement Rates from Table 6', 
            fontsize=12, ha='center', style='italic', color='gray')
    
    # ========== 浣庣己澶辩巼鍖?(MR 鈮?0.3) ==========
    box_low = FancyBboxPatch((0.5, 6.5), 2.5, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_low, alpha=0.3, edgecolor=color_low, linewidth=2)
    ax.add_patch(box_low)
    ax.text(1.75, 8.0, 'Low MR', fontsize=14, fontweight='bold', ha='center', color=color_low)
    ax.text(1.75, 7.6, 'MR 鈮?0.3', fontsize=12, ha='center')
    ax.text(1.75, 7.2, 'Baseline OK', fontsize=11, ha='center')
    ax.text(1.75, 6.85, 'MLP-Baseline', fontsize=10, ha='center', color='darkgreen')
    
    # 浣庣己澶辩巼MIM鏀硅繘
    ax.text(1.75, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(1.75, 6.2, 'MLP: 44.4% | 1D-CNN: 45.2%', fontsize=8, ha='center', color='gray')
    
    # ========== 涓瓑缂哄け鐜囧尯 (0.3 < MR 鈮?0.6) ==========
    box_mid = FancyBboxPatch((3.8, 6.5), 2.5, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_mid, alpha=0.3, edgecolor=color_mid, linewidth=2)
    ax.add_patch(box_mid)
    ax.text(5.05, 8.0, 'Medium MR', fontsize=14, fontweight='bold', ha='center', color=color_mid)
    ax.text(5.05, 7.6, '0.3 < MR 鈮?0.6', fontsize=12, ha='center')
    ax.text(5.05, 7.2, 'Recommend MIM', fontsize=11, ha='center')
    ax.text(5.05, 6.85, '1D-CNN-MIM / MLP-MIM', fontsize=10, ha='center', color='darkorange')
    
    # 涓瓑缂哄け鐜嘙IM鏀硅繘
    ax.text(5.05, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(5.05, 6.2, 'MLP: 46.8% | 1D-CNN: 55.3%', fontsize=8, ha='center', color='gray')
    
    # ========== 楂樼己澶辩巼鍖?(MR 鈮?0.7) ==========
    box_high = FancyBboxPatch((7.1, 6.5), 2.5, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_high, alpha=0.3, edgecolor=color_high, linewidth=2)
    ax.add_patch(box_high)
    ax.text(8.35, 8.0, 'High MR', fontsize=14, fontweight='bold', ha='center', color=color_high)
    ax.text(8.35, 7.6, 'MR 鈮?0.7', fontsize=12, ha='center')
    ax.text(8.35, 7.2, 'Strongly Recommend MIM', fontsize=11, ha='center')
    ax.text(8.35, 6.85, '1D-CNN-MIM / LSTM-MIM', fontsize=10, ha='center', color='darkred')
    
    # 楂樼己澶辩巼MIM鏀硅繘
    ax.text(8.35, 6.5, 'MIM Improvement:', fontsize=9, ha='center', color='gray')
    ax.text(8.35, 6.2, '1D-CNN: 57.0% | LSTM: 38.0%', fontsize=8, ha='center', color='gray')
    
    # ========== 璇︾粏妯″瀷瀵规瘮琛?==========
    table_data = [
        ['Model', 'MR=0.3', 'MR=0.5', 'MR=0.7', 'MR=0.9', 'Avg'],
        ['MLP-MIM', '46.5%', '46.8%', '42.4%', '27.9%', '42.5%'],
        ['LSTM-MIM', '17.8%', '29.0%', '38.0%', '35.3%', '26.5%'],
        ['GRU-MIM', '20.6%', '30.4%', '37.4%', '31.4%', '26.9%'],
        ['1D-CNN-MIM', '51.1%', '55.3%', '57.0%', '51.0%', '52.6%'],
    ]
    
    # 缁樺埗琛ㄦ牸
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.15, 0.15, 0.7, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 璁剧疆琛ㄥご鏍峰紡
    for i in range(6):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 璁剧疆绗竴鍒楁牱寮?
    for i in range(1, 5):
        table[(i, 0)].set_facecolor('#ECF0F1')
        table[(i, 0)].set_text_props(weight='bold')
    
    # 楂樹寒鏈€楂樻敼杩涚巼
    for i in range(1, 5):
        for j in range(1, 6):
            val = float(table_data[i][j].rstrip('%'))
            if val >= 50:
                table[(i, j)].set_facecolor('#D5F4E6')
    
    ax.text(5, 3.8, 'MIM Improvement Rates by Model and Missing Rate', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 3.4, '(Green cells: improvement 鈮?50%)', 
            fontsize=9, ha='center', color='gray', style='italic')
    
    # 娣诲姞鍐崇瓥绠ご
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

