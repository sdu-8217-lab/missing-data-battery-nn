# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
缁嗙矑搴︽灦鏋勬悳绱㈠叆鍙ｈ剼鏈?

浣跨敤绀轰緥:
    # 鎼滅储鎵€鏈夋ā鍨嬬殑鏈€浣虫灦鏋?
    python run_fine_grained_search.py --model all --batch 3C
    
    # 鍙悳绱STM鐨勭粏绮掑害閰嶇疆
    python run_fine_grained_search.py --model lstm --batch 3C
    
    # 浠庢寚瀹氱储寮曟仮澶?
    python run_fine_grained_search.py --model gru --start-idx 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='缁嗙矑搴︽灦鏋勬悳绱?- 鍦ㄥ綋鍓嶆渶浣抽厤缃檮杩戝瘑闆嗘帰绱?,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
绀轰緥:
  # 鎼滅储鎵€鏈夋ā鍨嬶紙鎺ㄨ崘锛?
  python run_fine_grained_search.py --model all --batch 3C
  
  # 鍙悳绱STM
  python run_fine_grained_search.py --model lstm --batch 3C
  
  # 鎼滅储GRU骞朵粠涓柇澶勬仮澶?
  python run_fine_grained_search.py --model gru --start-idx 25
  
  # 鎼滅储CNN1D
  python run_fine_grained_search.py --model cnn1d --batch 3C
  
  # 鎼滅储XGBoost锛堢矖绮掑害琚壀鏋濓紝缁嗙矑搴﹀啀楠岃瘉锛?
  python run_fine_grained_search.py --model xgboost --batch 3C
        """
    )
    
    parser.add_argument('--model', choices=['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost', 'all'],
                       default='all',
                       help='瑕佹悳绱㈢殑妯″瀷绫诲瀷 (榛樿: all)')
    parser.add_argument('--batch', default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='鏁版嵁鎵规 (榛樿: 3C)')
    parser.add_argument('--seed', type=int, default=42,
                       help='闅忔満绉嶅瓙 (榛樿: 42)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='璧峰鏋舵瀯绱㈠紩锛岀敤浜庢仮澶?(榛樿: 0)')
    
    args = parser.parse_args()
    
    from src.experiments.fine_grained_search import (
        run_single_model_search, run_all_models_fine_grained
    )
    
    # 浼扮畻鎼滅储绌洪棿澶у皬
    search_space_sizes = {
        'mlp': '~120涓厤缃?,
        'lstm': '~80涓厤缃?,
        'gru': '~80涓厤缃?,
        'cnn1d': '~150涓厤缃?,
        'xgboost': '~150涓厤缃?,
        'all': '~580涓厤缃紙鎬昏€楁椂绾?-10灏忔椂锛?
    }
    
    print("=" * 70)
    print("缁嗙矑搴︽灦鏋勬悳绱?)
    print("=" * 70)
    print(f"鐩爣妯″瀷: {args.model.upper()}")
    print(f"鏁版嵁鎵规: {args.batch}")
    print(f"闅忔満绉嶅瓙: {args.seed}")
    print(f"鎼滅储绌洪棿: {search_space_sizes[args.model]}")
    print(f"閰嶇疆: 50 epochs, 缂哄け鐜?[0.1, 0.5, 0.9]")
    print("=" * 70)
    print()
    
    confirm = input("纭寮€濮嬫悳绱? (y/n): ")
    if confirm.lower() != 'y':
        print("宸插彇娑?)
        return
    
    if args.model == 'all':
        results = run_all_models_fine_grained(args.batch, args.seed)
        
        # 姹囨€绘渶浣崇粨鏋?
        print("\n" + "=" * 70)
        print("鍚勬ā鍨嬫渶浣虫灦鏋勬眹鎬?)
        print("=" * 70)
        for model_type, df in results.items():
            if len(df) > 0:
                best = df.loc[df['mae_mean'].idxmin()]
                print(f"\n銆恵model_type.upper()}銆?)
                print(f"  鏈€浣矼AE: {best['mae_mean']:.4f}")
                print(f"  鍙傛暟閲? {best['param_count']:,}")
                print(f"  閰嶇疆: {best['config']}")
    else:
        df = run_single_model_search(args.model, args.batch, args.seed)
        
        if len(df) > 0:
            # 鏄剧ずTop 5
            print("\n" + "=" * 70)
            print(f"{args.model.upper()} Top 5 鏋舵瀯")
            print("=" * 70)
            top5 = df.nsmallest(5, 'mae_mean')[['config_name', 'mae_mean', 'param_count', 'config']]
            for idx, row in top5.iterrows():
                print(f"\n{row['config_name']}:")
                print(f"  MAE: {row['mae_mean']:.4f}")
                print(f"  鍙傛暟閲? {row['param_count']:,}")
                print(f"  閰嶇疆: {row['config']}")


if __name__ == '__main__':
    main()

