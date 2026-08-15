#!/usr/bin/env python3
"""
为单个小实验结果绘图
可独立运行
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.visualization.single_plots import SingleExperimentPlots


def main():
    parser = argparse.ArgumentParser(description='为单个小实验绘图')
    parser.add_argument('--input', type=str, required=True, help='结果CSV文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / "figures"
    
    print(f"读取结果: {input_path}")
    df = pd.read_csv(input_path)
    print(f"记录数: {len(df)}")
    
    # 获取种子信息
    seed = df['seed'].iloc[0] if 'seed' in df.columns else 'unknown'
    print(f"种子: {seed}")
    
    # 绘图
    print(f"生成图表到: {output_dir}")
    plotter = SingleExperimentPlots(output_dir)
    plotter.plot_all(df)
    
    print("完成!")


if __name__ == '__main__':
    main()
