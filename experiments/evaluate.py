#!/usr/bin/env python3
"""
评估入口脚本
用于评估训练好的模型并生成结果图表
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.model_evaluator import ModelEvaluator
from src.visualization.batch_plots import BatchPlotter


def main():
    parser = argparse.ArgumentParser(description='Evaluate MIM models')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='实验结果目录路径')
    parser.add_argument('--output-dir', type=str, default='results/images',
                        help='输出图像目录')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'both'],
                        help='输出图像格式')
    
    args = parser.parse_args()
    
    print(f"Evaluating results from: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # TODO: 实现评估逻辑
    # 1. 加载实验结果
    # 2. 计算评估指标
    # 3. 生成可视化图表
    
    print("Evaluation completed.")


if __name__ == '__main__':
    main()
