# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
妯″瀷鏋舵瀯鎼滅储瀹為獙鍏ュ彛鑴氭湰

浣跨敤绀轰緥:
    # 闃舵1: 绮楃矑搴︽悳绱?
    python run_architecture_search.py --stage coarse --batch 3C
    
    # 闃舵2: 缁嗙矑搴︽悳绱?
    python run_architecture_search.py --stage fine --batch 3C --coarse-results experiments_v2/arch_search_3C_coarse_*/results.csv
    
    # 鍒嗘瀽缁撴灉
    python run_architecture_search.py --analyze experiments_v2/arch_search_*/results.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='妯″瀷鏋舵瀯鎼滅储 - 瀵绘壘鏈€浣冲弬鏁伴噺涓庢灦鏋勯厤缃?,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
绀轰緥:
  # 杩愯绮楃矑搴︽悳绱紙蹇€熺瓫閫夛級
  python run_architecture_search.py --stage coarse --batch 3C --seed 42
  
  # 杩愯缁嗙矑搴︽悳绱紙绮剧‘楠岃瘉锛?
  python run_architecture_search.py --stage fine --batch 3C --coarse-results results.csv
  
  # 鍒嗘瀽宸叉湁缁撴灉
  python run_architecture_search.py --analyze results.csv --budget 50000
        """
    )
    
    parser.add_argument('--stage', choices=['coarse', 'fine'],
                       help='鎼滅储闃舵: coarse=绮楃矑搴?10epochs), fine=缁嗙矑搴?50epochs)')
    parser.add_argument('--batch', default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='鏁版嵁鎵规 (榛樿: 3C)')
    parser.add_argument('--seed', type=int, default=42,
                       help='闅忔満绉嶅瓙 (榛樿: 42)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='璧峰鏋舵瀯绱㈠紩锛岀敤浜庝粠涓柇澶勬仮澶?(榛樿: 0)')
    parser.add_argument('--coarse-results',
                       help='绮楃矑搴︾粨鏋滆矾寰勶紙缁嗙矑搴﹂樁娈典娇鐢級')
    parser.add_argument('--analyze',
                       help='鍒嗘瀽宸叉湁缁撴灉鏂囦欢')
    parser.add_argument('--budget', type=int, default=50000,
                       help='鍙傛暟閲忛绠楋紝鐢ㄤ簬鎺ㄨ崘鏈€浣虫灦鏋?(榛樿: 50000)')
    
    args = parser.parse_args()
    
    # 鍒嗘瀽妯″紡
    if args.analyze:
        from src.visualization.architecture_analysis import ArchitectureAnalyzer
        analyzer = ArchitectureAnalyzer(args.analyze)
        analyzer.analyze_all()
        analyzer.recommend_best_balance(args.budget)
        return
    
    # 妫€鏌ュ繀椤诲弬鏁?
    if not args.stage:
        parser.error("蹇呴』鎸囧畾 --stage (coarse 鎴?fine)")
    
    # 瀵煎叆骞惰繍琛屾悳绱?
    from src.experiments.architecture_search import (
        run_coarse_search, run_fine_search
    )
    
    if args.stage == 'coarse':
        print("=" * 70)
        print("妯″瀷鏋舵瀯鎼滅储 - 闃舵1: 绮楃矑搴︾瓫閫?)
        print("=" * 70)
        print(f"鎵规: {args.batch}")
        print(f"绉嶅瓙: {args.seed}")
        print(f"閰嶇疆: 10 epochs, 鍙祴缂哄け鐜?.5, 鑷姩鍓灊")
        print(f"璧峰绱㈠紩: {args.start_idx}")
        print("=" * 70)
        print()
        
        df = run_coarse_search(args.batch, args.seed, args.start_idx)
        
        print("\n绮楃矑搴︽悳绱㈠畬鎴?")
        print(f"娴嬭瘯鏋舵瀯鏁? {len(df) if df is not None else 0}")
        print("\n涓嬩竴姝ュ缓璁?")
        print("1. 鏌ョ湅鍒嗘瀽鍥捐〃: python run_architecture_search.py --analyze <results_path>")
        print("2. 杩愯缁嗙矑搴︽悳绱? python run_architecture_search.py --stage fine --coarse-results <results_path>")
        
    else:  # fine stage
        if not args.coarse_results:
            parser.error("缁嗙矑搴﹂樁娈甸渶瑕佹彁渚?--coarse-results 鍙傛暟")
        
        print("=" * 70)
        print("妯″瀷鏋舵瀯鎼滅储 - 闃舵2: 缁嗙矑搴﹂獙璇?)
        print("=" * 70)
        print(f"鎵规: {args.batch}")
        print(f"绉嶅瓙: {args.seed}")
        print(f"閰嶇疆: 50 epochs, 娴嬬己澶辩巼0.1/0.5/0.9, 3涓瀛?)
        print("=" * 70)
        print()
        
        df = run_fine_search(args.coarse_results, args.batch, args.seed)
        
        print("\n缁嗙矑搴︽悳绱㈠畬鎴?")
        print(f"娴嬭瘯鏋舵瀯鏁? {len(df) if df is not None else 0}")


if __name__ == '__main__':
    main()

