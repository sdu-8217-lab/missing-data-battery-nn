# -*- coding: utf-8 -*-
with open('paper/main_zh.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Get line 279
line = lines[278]

# The problematic part we want to replace
# "通过显式标记被插补的位置，将"被插补的位置"显式暴露给模型，从而部分纠正插补引入的统计偏移。这种"简单插补+显式指示"的组合策略，在实际BMS部署中兼具实现复杂度与预测精度的平衡。"
# Should become:
# "通过显式标记被插补的位置，将缺失模式作为额外输入信号暴露给模型，以减轻插补误差对预测的影响。"

# Try to find the suffix and replace
old_suffix = '显式暴露给模型，从而部分纠正插补引入的统计偏移。这种"简单插补+显式指示"的组合策略，在实际BMS部署中兼具实现复杂度与预测精度的平衡。'
new_suffix = '以减轻插补误差对预测的影响。'

if old_suffix in line:
    line = line.replace(old_suffix, new_suffix)
    lines[278] = line
    with open('paper/main_zh.tex', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('Fixed the suffix!')
else:
    print('Old suffix not found!')
    # Try shorter match
    if '从而部分纠正插补引入的统计偏移' in line:
        print('Found: 从而部分纠正插补引入的统计偏移')
    if '简单插补+显式指示' in line:
        print('Found: 简单插补+显式指示')
    if '显式暴露给模型' in line:
        print('Found: 显式暴露给模型')
