# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('main_zh.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 统计各种引号
left_quote = chr(0x201C)   # "
right_quote = chr(0x201D)  # "
eng_quote = chr(34)        # "

left_count = 0
right_count = 0
eng_count = 0

for i, line in enumerate(lines, 1):
    l = line.count(left_quote)
    r = line.count(right_quote)
    e = line.count(eng_quote)
    left_count += l
    right_count += r
    eng_count += e
    
    # 显示包含英文引号的行（排除LaTeX命令中的引号）
    if e > 0:
        # 检查是否是在文本内容中（简单判断：包含中文字符）
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in line)
        if has_chinese:
            print(f'Line {i}: {line.strip()[:100]}')

print(f'\n=== 统计 ===')
print(f'左引号（"）: {left_count}')
print(f'右引号（"）: {right_count}')
print(f'英文引号（"）: {eng_count}')

if left_count != right_count:
    print(f'\n警告：左右引号数量不匹配！')
if eng_count > 0:
    print(f'\n警告：存在 {eng_count} 个英文引号！')
