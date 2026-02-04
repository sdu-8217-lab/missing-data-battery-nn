# -*- coding: utf-8 -*-
import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

with open('main_zh.tex', 'r', encoding='utf-8') as f:
    content = f.read()

eng_quote = chr(34)        # "
left_quote = chr(0x201C)   # "
right_quote = chr(0x201D)  # "

def replace_quotes(match):
    """将英文引号替换为中文引号"""
    before = match.group(1)
    text = match.group(2)
    after = match.group(3)
    
    # 判断是左引号还是右引号
    # 如果前面是中文、空格或标点，且后面是中文内容，则是左引号
    # 如果前面是内容，后面是标点或中文，则是右引号
    
    # 简单规则：根据位置判断
    # 在LaTeX中，引号通常成对出现
    return left_quote + text + right_quote

# 找到所有英文引号包裹的内容
# 模式: 英文引号 + 非引号内容 + 英文引号
pattern = r'"([^"\\]*)"'

# 用于计数
matches = list(re.finditer(pattern, content))
print(f'找到 {len(matches)} 对英文引号')

# 替换策略：奇数位置的引号替换为左引号，偶数位置的替换为右引号
# 但这需要遍历整个文档

result = []
i = 0
quote_count = 0

while i < len(content):
    if content[i] == eng_quote:
        # 判断是左引号还是右引号
        # 简单规则：交替使用左、右引号
        if quote_count % 2 == 0:
            result.append(left_quote)
        else:
            result.append(right_quote)
        quote_count += 1
        i += 1
    else:
        result.append(content[i])
        i += 1

new_content = ''.join(result)

# 验证
left_count = new_content.count(left_quote)
right_count = new_content.count(right_quote)
eng_count = new_content.count(eng_quote)

print(f'替换后：左引号={left_count}, 右引号={right_count}, 英文引号={eng_count}')

with open('main_zh.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('修复完成！')
