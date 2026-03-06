# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
涓哄崟涓皬瀹為獙缁撴灉缁樺浘
鍙嫭绔嬭繍琛?
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.visualization.single_plots import SingleExperimentPlots


def main():
    parser = argparse.ArgumentParser(description='涓哄崟涓皬瀹為獙缁樺浘')
    parser.add_argument('--input', type=str, required=True, help='缁撴灉CSV鏂囦欢璺緞')
    parser.add_argument('--output', type=str, default=None, help='杈撳嚭鐩綍')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"閿欒: 鏂囦欢涓嶅瓨鍦?{input_path}")
        sys.exit(1)
    
    # 纭畾杈撳嚭鐩綍
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / "figures"
    
    print(f"璇诲彇缁撴灉: {input_path}")
    df = pd.read_csv(input_path)
    print(f"璁板綍鏁? {len(df)}")
    
    # 鑾峰彇绉嶅瓙淇℃伅
    seed = df['seed'].iloc[0] if 'seed' in df.columns else 'unknown'
    print(f"绉嶅瓙: {seed}")
    
    # 缁樺浘
    print(f"鐢熸垚鍥捐〃鍒? {output_dir}")
    plotter = SingleExperimentPlots(output_dir)
    plotter.plot_all(df)
    
    print("瀹屾垚!")


if __name__ == '__main__':
    main()

