# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
妫€鏌ュ疄楠岀姸鎬?
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description='妫€鏌ュ疄楠岀姸鎬?)
    parser.add_argument('--experiment_dir', type=str, required=True, 
                       help='瀹為獙鐩綍璺緞')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    
    if not exp_dir.exists():
        print(f"閿欒: 鐩綍涓嶅瓨鍦?{exp_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("瀹為獙鐘舵€佹鏌?)
    print("=" * 70)
    print(f"鐩綍: {exp_dir}")
    print()
    
    # 鍔犺浇妫€鏌ョ偣
    checkpoint_file = exp_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        
        print("妫€鏌ョ偣淇℃伅:")
        print(f"  鎵规: {checkpoint.get('batch_name', 'N/A')}")
        print(f"  鏃堕棿鎴? {checkpoint.get('timestamp', 'N/A')}")
        print(f"  鐘舵€? {checkpoint.get('status', 'N/A')}")
        print(f"  鎬荤瀛愭暟: {checkpoint.get('total_seeds', 'N/A')}")
        print(f"  宸插畬鎴? {len(checkpoint.get('completed_seeds', []))}")
        print(f"  澶辫触: {len(checkpoint.get('failed_seeds', []))}")
        
        completed = checkpoint.get('completed_seeds', [])
        total = checkpoint.get('total_seeds', 1)
        percentage = len(completed) / total * 100
        
        print(f"  杩涘害: {percentage:.1f}%")
        print()
        
        # 鏄剧ず宸插畬鎴愬拰鏈畬鎴愮殑绉嶅瓙
        all_seeds = set(checkpoint.get('seeds', []))
        completed_set = set(completed)
        failed_set = set(checkpoint.get('failed_seeds', []))
        remaining = sorted(list(all_seeds - completed_set))
        
        if completed:
            print(f"宸插畬鎴愮瀛?({len(completed)}涓?:")
            print(f"  {completed[:20]}{'...' if len(completed) > 20 else ''}")
            print()
        
        if remaining:
            print(f"鏈畬鎴愮瀛?({len(remaining)}涓?:")
            print(f"  {remaining[:20]}{'...' if len(remaining) > 20 else ''}")
            print()
        
        if failed_set:
            print(f"澶辫触绉嶅瓙 ({len(failed_set)}涓?:")
            print(f"  {sorted(list(failed_set))}")
            print()
        
        # 寤鸿
        if checkpoint.get('status') == 'completed':
            print("鐘舵€? 瀹為獙宸插畬鎴?)
        elif remaining:
            print("寤鸿: 杩愯浠ヤ笅鍛戒护缁х画瀹為獙:")
            print(f"  python run_batch.py --batch {checkpoint.get('batch_name')} "
                  f"--n_repeats {total} --resume")
    else:
        print("鏈壘鍒版鏌ョ偣鏂囦欢")
        
        # 灏濊瘯缁熻绉嶅瓙鐩綍
        seed_dirs = list(exp_dir.glob("seed_*"))
        if seed_dirs:
            print(f"鍙戠幇 {len(seed_dirs)} 涓瀛愮洰褰?)
            for sd in seed_dirs[:5]:
                print(f"  - {sd.name}")
            if len(seed_dirs) > 5:
                print(f"  ... 杩樻湁 {len(seed_dirs) - 5} 涓?)


if __name__ == '__main__':
    main()

