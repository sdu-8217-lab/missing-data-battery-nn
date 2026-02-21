# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
涓哄ぇ瀹為獙缁撴灉缁樺埗缁熻鎬у浘琛?
鍙嫭绔嬭繍琛岋紝鐢ㄤ簬涓柇鍚庢墜鍔ㄦ仮澶?
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.visualization.batch_plots import BatchExperimentPlots


def main():
    parser = argparse.ArgumentParser(description='涓哄ぇ瀹為獙缁樺埗缁熻鎬у浘琛?)
    parser.add_argument('--input', type=str, required=True, 
                       help='杈撳叆鐩綍锛堝寘鍚瀛愬瓙鐩綍锛夋垨CSV鏂囦欢')
    parser.add_argument('--output', type=str, default=None, help='杈撳嚭鐩綍')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 纭畾杈撳叆鏂囦欢
    if input_path.is_dir():
        # 濡傛灉鏄洰褰曪紝鏌ユ壘seeds/瀛愮洰褰曟垨aggregate/results_all.csv
        if (input_path / "aggregate" / "results_all.csv").exists():
            csv_file = input_path / "aggregate" / "results_all.csv"
        elif (input_path / "results_all.csv").exists():
            csv_file = input_path / "results_all.csv"
        else:
            # 鍚堝苟鎵€鏈夌瀛愮粨鏋?
            print("姝ｅ湪鍚堝苟鍚勭瀛愮粨鏋?..")
            all_results = []
            for seed_dir in input_path.glob("seed_*"):
                result_file = seed_dir / "results.csv"
                if result_file.exists():
                    all_results.append(pd.read_csv(result_file))
            
            if not all_results:
                print(f"閿欒: 鍦?{input_path} 涓湭鎵惧埌缁撴灉鏂囦欢")
                sys.exit(1)
            
            df = pd.concat(all_results, ignore_index=True)
            print(f"鍚堝苟瀹屾垚: {len(df)} 鏉¤褰?)
    else:
        csv_file = input_path
        df = pd.read_csv(csv_file)
    
    # 纭畾杈撳嚭鐩綍
    if args.output:
        output_dir = Path(args.output)
    else:
        if input_path.is_dir():
            output_dir = input_path / "aggregate" / "figures"
        else:
            output_dir = input_path.parent / "figures"
    
    print(f"缁撴灉璁板綍鏁? {len(df)}")
    print(f"绉嶅瓙鏁? {df['seed'].nunique() if 'seed' in df.columns else 'N/A'}")
    print(f"妯″瀷鏁? {df['model_name'].nunique() if 'model_name' in df.columns else 'N/A'}")
    
    # 缁樺浘
    print(f"\n鐢熸垚缁熻鍥捐〃鍒? {output_dir}")
    plotter = BatchExperimentPlots(output_dir)
    plotter.plot_all(df)
    
    print("瀹屾垚!")


if __name__ == '__main__':
    main()

