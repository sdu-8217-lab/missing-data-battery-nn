#!/usr/bin/env python3
"""
检查实验状态
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description='检查实验状态')
    parser.add_argument('--experiment_dir', type=str, required=True, 
                       help='实验目录路径')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    
    if not exp_dir.exists():
        print(f"错误: 目录不存在 {exp_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("实验状态检查")
    print("=" * 70)
    print(f"目录: {exp_dir}")
    print()
    
    # 加载检查点
    checkpoint_file = exp_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        
        print("检查点信息:")
        print(f"  批次: {checkpoint.get('batch_name', 'N/A')}")
        print(f"  时间戳: {checkpoint.get('timestamp', 'N/A')}")
        print(f"  状态: {checkpoint.get('status', 'N/A')}")
        print(f"  总种子数: {checkpoint.get('total_seeds', 'N/A')}")
        print(f"  已完成: {len(checkpoint.get('completed_seeds', []))}")
        print(f"  失败: {len(checkpoint.get('failed_seeds', []))}")
        
        completed = checkpoint.get('completed_seeds', [])
        total = checkpoint.get('total_seeds', 1)
        percentage = len(completed) / total * 100
        
        print(f"  进度: {percentage:.1f}%")
        print()
        
        # 显示已完成和未完成的种子
        all_seeds = set(checkpoint.get('seeds', []))
        completed_set = set(completed)
        failed_set = set(checkpoint.get('failed_seeds', []))
        remaining = sorted(list(all_seeds - completed_set))
        
        if completed:
            print(f"已完成种子 ({len(completed)}个):")
            print(f"  {completed[:20]}{'...' if len(completed) > 20 else ''}")
            print()
        
        if remaining:
            print(f"未完成种子 ({len(remaining)}个):")
            print(f"  {remaining[:20]}{'...' if len(remaining) > 20 else ''}")
            print()
        
        if failed_set:
            print(f"失败种子 ({len(failed_set)}个):")
            print(f"  {sorted(list(failed_set))}")
            print()
        
        # 建议
        if checkpoint.get('status') == 'completed':
            print("状态: 实验已完成")
        elif remaining:
            print("建议: 运行以下命令继续实验:")
            print(f"  python run_batch.py --batch {checkpoint.get('batch_name')} "
                  f"--n_repeats {total} --resume")
    else:
        print("未找到检查点文件")
        
        # 尝试统计种子目录
        seed_dirs = list(exp_dir.glob("seed_*"))
        if seed_dirs:
            print(f"发现 {len(seed_dirs)} 个种子目录")
            for sd in seed_dirs[:5]:
                print(f"  - {sd.name}")
            if len(seed_dirs) > 5:
                print(f"  ... 还有 {len(seed_dirs) - 5} 个")


if __name__ == '__main__':
    main()
