#!/usr/bin/env python3
"""
批量评估脚本 - 训练完成后执行
扫描所有模型并批量评估
"""
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys


def evaluate_model_dir(model_dir: Path, workers: int = 1) -> bool:
    """评估单个模型目录"""
    try:
        cmd = [
            sys.executable, "scripts/evaluate_models.py",
            "--models-dir", str(model_dir),
            "--workers", str(workers),
            "--output-dir", str(model_dir.parent)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return result.returncode == 0
    except Exception as e:
        print(f"评估失败 {model_dir}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='批量评估脚本')
    parser.add_argument("--batch", required=True, help='批次名称')
    parser.add_argument("--workers", type=int, default=8, help='并行评估进程数')
    parser.add_argument("--output-dir", default='./results/10seeds_smart', help='输出目录')
    args = parser.parse_args()
    
    # 扫描所有模型目录
    batch_dir = Path(args.output_dir) / args.batch
    model_dirs = list(batch_dir.rglob("*/models"))
    
    if not model_dirs:
        print(f"未找到模型目录: {batch_dir}")
        return
    
    print(f"找到 {len(model_dirs)} 个模型目录")
    print(f"批量评估开始（{args.workers}并行）...")
    
    success_count = 0
    failed_count = 0
    
    # 并行评估
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_model_dir, d, 1): d for d in model_dirs}
        
        for i, future in enumerate(as_completed(futures), 1):
            model_dir = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"[{i}/{len(model_dirs)}] ✓ {model_dir.name}")
                else:
                    failed_count += 1
                    print(f"[{i}/{len(model_dirs)}] ✗ {model_dir.name}")
            except Exception as e:
                failed_count += 1
                print(f"[{i}/{len(model_dirs)}] ✗ {model_dir.name}: {e}")
    
    print(f"\n评估完成: 成功 {success_count}, 失败 {failed_count}")


if __name__ == "__main__":
    main()
