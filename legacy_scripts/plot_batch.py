#!/usr/bin/env python3
"""
为大实验结果绘制统计性图表
可独立运行，用于中断后手动恢复
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.visualization.batch_plots import BatchExperimentPlots


def main():
    parser = argparse.ArgumentParser(description='为大实验绘制统计性图表')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入目录（包含种子子目录）或CSV文件')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 确定输入文件
    if input_path.is_dir():
        # 如果是目录，查找seeds/子目录或aggregate/results_all.csv
        if (input_path / "aggregate" / "results_all.csv").exists():
            csv_file = input_path / "aggregate" / "results_all.csv"
        elif (input_path / "results_all.csv").exists():
            csv_file = input_path / "results_all.csv"
        else:
            # 合并所有种子结果
            print("正在合并各种子结果...")
            all_results = []
            for seed_dir in input_path.glob("seed_*"):
                result_file = seed_dir / "results.csv"
                if result_file.exists():
                    all_results.append(pd.read_csv(result_file))
            
            if not all_results:
                print(f"错误: 在 {input_path} 中未找到结果文件")
                sys.exit(1)
            
            df = pd.concat(all_results, ignore_index=True)
            print(f"合并完成: {len(df)} 条记录")
    else:
        csv_file = input_path
        df = pd.read_csv(csv_file)
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        if input_path.is_dir():
            output_dir = input_path / "aggregate" / "figures"
        else:
            output_dir = input_path.parent / "figures"
    
    print(f"结果记录数: {len(df)}")
    print(f"种子数: {df['seed'].nunique() if 'seed' in df.columns else 'N/A'}")
    print(f"模型数: {df['model_name'].nunique() if 'model_name' in df.columns else 'N/A'}")
    
    # 绘图
    print(f"\n生成统计图表到: {output_dir}")
    plotter = BatchExperimentPlots(output_dir)
    plotter.plot_all(df)
    
    print("完成!")


if __name__ == '__main__':
    main()
