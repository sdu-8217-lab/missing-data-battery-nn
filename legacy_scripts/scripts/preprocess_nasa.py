"""预处理 NASA .mat 文件，生成 per-battery CSV 特征。"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nasa_feature_extractor import process_nasa_directory


def main():
    parser = argparse.ArgumentParser(description='预处理 NASA 电池数据集')
    parser.add_argument('--data_dir', type=str, default='./data/NASA data',
                       help='包含 .mat 文件的目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，默认 data_dir/processed')
    args = parser.parse_args()

    summary = process_nasa_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))
    print(f"\n共处理 {len(summary)} 个电池，总循环数 {summary['n_cycles'].sum()}")


if __name__ == '__main__':
    main()
