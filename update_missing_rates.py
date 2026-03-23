#!/usr/bin/env python3
"""
批量修改缺失率设置脚本
从 0.0, 0.1, ..., 0.9 (10档) 改为 0.0, 0.05, ..., 0.95 (20档)
"""

import re
from pathlib import Path

# 旧的缺失率列表（多种格式）
OLD_PATTERNS = [
    r'\[0\.0, 0\.1, 0\.2, 0\.3, 0\.4, 0\.5, 0\.6, 0\.7, 0\.8, 0\.9\]',  # Python列表
    r'\[0\.0,0\.1,0\.2,0\.3,0\.4,0\.5,0\.6,0\.7,0\.8,0\.9\]',  # 无空格
    r'0\.0, 0\.1, 0\.2, 0\.3, 0\.4, 0\.5, 0\.6, 0\.7, 0\.8, 0\.9',  # 逗号分隔
    r'0\.0,0\.1,0\.2,0\.3,0\.4,0\.5,0\.6,0\.7,0\.8,0\.9',  # 无空格
]

# 新的缺失率列表（20档，步长0.05）
NEW_MR_LIST = '[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]'
NEW_MR_LIST_NO_SPACES = '[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]'
NEW_MR_COMMA = '0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95'

# 需要修改的文件列表
FILES_TO_UPDATE = [
    # 核心配置
    'src/config/pydantic_config.py',
    'configs/reference/100seeds_experiment.yaml',
    'configs/reference/default_mlp_2c.yaml',
    'configs/reference/default_mlp_2c_mim.yaml',
    
    # 实验运行
    'experiments/run_experiment.py',
    'experiments/run_batch_experiments_v2.py',
    
    # 可视化脚本
    'experiments/visualize_single_seed.py',
    'experiments/visualize_by_batch_detail.py',
    'experiments/visualize_100seeds_results.py',
    'experiments/visualize_complete_analysis.py',
    'experiments/visualize_zero_mim_comparison.py',
    'experiments/visualization_mim_analysis.py',
    
    # 其他源文件
    'src/trainers/neural_network_trainer.py',
    'src/main.py',
    'scripts/run_paper_experiment.py',
    
    # 文档
    'docs/EXPERIMENTS.md',
    'configs/reference/experiment-params.md',
]


def update_file(filepath):
    """更新单个文件"""
    path = Path(filepath)
    if not path.exists():
        print(f"  ⚠️  文件不存在: {filepath}")
        return False
    
    content = path.read_text(encoding='utf-8')
    original_content = content
    
    # 统计替换次数
    replacements = 0
    
    # 替换带空格的列表格式
    for pattern in OLD_PATTERNS:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            # 根据原格式选择新格式
            if pattern.startswith(r'\['):
                if ' ' in pattern:
                    new_val = NEW_MR_LIST
                else:
                    new_val = NEW_MR_LIST_NO_SPACES
            else:
                if ' ' in pattern:
                    new_val = NEW_MR_COMMA
                else:
                    new_val = NEW_MR_COMMA.replace(' ', '')
            
            content = re.sub(pattern, new_val, content)
            replacements += matches
    
    if content != original_content:
        path.write_text(content, encoding='utf-8')
        print(f"  ✅ 已更新: {filepath} ({replacements} 处替换)")
        return True
    else:
        print(f"  ⏭️  无变化: {filepath}")
        return False


def main():
    print("=" * 70)
    print("批量修改缺失率设置")
    print("=" * 70)
    print(f"\n旧设置: 10档 [0.0, 0.1, ..., 0.9]")
    print(f"新设置: 20档 [0.0, 0.05, ..., 0.95]")
    print()
    
    updated_count = 0
    for filepath in FILES_TO_UPDATE:
        if update_file(filepath):
            updated_count += 1
    
    print()
    print("=" * 70)
    print(f"完成! 更新了 {updated_count}/{len(FILES_TO_UPDATE)} 个文件")
    print("=" * 70)
    print("\n⚠️  注意: 以下文件需要手动检查:")
    print("  - meta.md (如果包含缺失率说明)")
    print("  - paper/*.tex (论文中的数学公式)")
    print("  - src/experiments/runner.py (字符串格式的缺失率)")
    print("  - src/missing_data/fixed_feature_missing.py (已经是20档，跳过)")


if __name__ == "__main__":
    main()
