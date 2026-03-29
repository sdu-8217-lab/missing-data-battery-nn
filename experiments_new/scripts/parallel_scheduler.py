#!/usr/bin/env python3
"""
并行实验调度器 - 最终优化版 (v3.1)

特性:
- 极简实现 (~80行核心代码)
- 自动GPU显存检查
- 完整日志保存
- 错误处理和统计
"""

import os
import sys
import argparse
import time
from pathlib import Path
from multiprocessing import Pool
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gpu_memory(min_gb=2.0):
    """检查GPU显存是否充足"""
    if not torch.cuda.is_available():
        return True
    free_bytes = torch.cuda.mem_get_info()[0]
    free_gb = free_bytes / (1024**3)
    return free_gb >= min_gb


def run_single_experiment(args):
    """运行单个实验"""
    config_file, model_config, seed, output_dir = args
    
    # 确保输出目录存在
    log_dir = Path(output_dir) / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "parallel_training.log"
    
    # 构建命令
    cmd = (
        f"{sys.executable} scripts/train_models.py "
        f"--config {config_file} "
        f"--model-config {model_config} "
        f"--seeds {seed} "
        f"--output-dir {output_dir} "
        f">> {log_file} 2>&1"
    )
    
    start_time = time.time()
    exit_code = os.system(cmd)
    duration = time.time() - start_time
    
    return {
        'config': Path(config_file).stem,
        'seed': seed,
        'success': exit_code == 0,
        'duration': duration,
        'log': str(log_file)
    }


def main():
    parser = argparse.ArgumentParser(description='Parallel Experiment Scheduler')
    parser.add_argument('--batch', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--output-dir', type=str, default='results/10seeds_parallel')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--arch', type=str, default='mlp')
    parser.add_argument('--levels', type=str, nargs='+', 
                        default=['level_1', 'level_2', 'level_3', 'level_4'])
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Parallel Scheduler v3.1 (Optimized)")
    print(f"Batch: {args.batch}, Workers: {args.workers}")
    print(f"{'='*60}\n")
    
    # 生成任务
    tasks = []
    config_dir = "configs/experiments/batch_configs_standard"
    
    for level in args.levels:
        model_config = f"configs/models/{args.arch}_{level}.yaml"
        if not Path(model_config).exists():
            continue
            
        for config_suffix in ['mim', 'baseline']:
            config_file = f"{config_dir}/batch_{args.batch}_{args.arch}_{config_suffix}.yaml"
            if not Path(config_file).exists():
                continue
                
            output_subdir = f"{args.output_dir}/{args.batch}/{args.arch}_{level}_{config_suffix}"
            
            for seed in args.seeds:
                # 检查是否已完成
                model_file = Path(output_subdir) / f"seed_{seed}/models/best_model.pt"
                if not model_file.exists():
                    tasks.append((config_file, model_config, seed, output_subdir))
    
    if not tasks:
        print("All tasks already completed!")
        return 0
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Starting with {args.workers} workers...\n")
    
    # 检查GPU
    if not check_gpu_memory():
        print("Warning: GPU memory may be insufficient")
    
    start_time = time.time()
    completed = failed = 0
    
    # 并行执行
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_experiment, tasks), 1):
            status = "✓" if result['success'] else "✗"
            print(f"[{status}] {i}/{len(tasks)} {result['config']} s{result['seed']} "
                  f"({result['duration']:.1f}s)")
            
            if result['success']:
                completed += 1
            else:
                failed += 1
                print(f"    Error log: {result['log']}")
    
    # 统计
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{len(tasks)} | Failed: {failed}")
    print(f"Time: {total_time/60:.1f}min | Rate: {completed/(total_time/3600):.1f}/hour")
    
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
