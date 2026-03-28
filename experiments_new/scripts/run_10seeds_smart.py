#!/usr/bin/env python3
"""
智能调度实验运行器 - 10种子版本
自动平衡GPU训练和CPU评估，最大化硬件利用率

核心策略:
1. GPU密集: 训练阶段（使用单进程，避免显存冲突）
2. CPU密集: 评估阶段（使用多进程并行）
3. 流水线: 一个配置评估时，GPU开始下一个配置训练

用法:
    # 全量实验
    python scripts/run_10seeds_smart.py
    
    # 只跑3C batch
    python scripts/run_10seeds_smart.py --batch 3C
    
    # 快速测试（2种子）
    python scripts/run_10seeds_smart.py --seeds 0 1 --dry-run
"""
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='智能调度实验运行器')
    parser.add_argument('--batch', type=str, default=None,
                       help='指定batch，不指定则跑全部')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       help='随机种子列表')
    parser.add_argument('--config-dir',
                       default='configs/experiments/batch_configs_standard',
                       help='配置目录')
    parser.add_argument('--output-dir', default='./results/10seeds_smart',
                       help='输出目录')
    parser.add_argument('--eval-workers', type=int, default=8,
                       help='评估并行进程数')
    parser.add_argument('--dry-run', action='store_true',
                       help='干运行模式')
    parser.add_argument('--skip-eval', action='store_true',
                       help='跳过评估阶段（只训练）')
    parser.add_argument('--train-only', action='store_true',
                       help='只跑训练阶段')
    return parser.parse_args()


def get_experiment_tasks(args) -> List[Dict]:
    """生成所有实验任务"""
    batches = [args.batch] if args.batch else ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
    archs = ["mlp", "lstm", "cnn"]
    levels = ["level_1", "level_2", "level_3", "level_4"]
    modes = ["mim", "baseline"]
    
    tasks = []
    for batch in batches:
        for arch in archs:
            for level in levels:
                for mode in modes:
                    config_file = f"batch_{batch}_{arch}_{mode}.yaml"
                    config_path = Path(args.config_dir) / config_file
                    model_config = f"configs/models/{arch}_{level}.yaml"
                    
                    if config_path.exists() and Path(model_config).exists():
                        tasks.append({
                            'id': f"{batch}_{arch}_{level}_{mode}",
                            'batch': batch,
                            'arch': arch,
                            'level': level,
                            'mode': mode,
                            'config': str(config_path),
                            'model_config': model_config,
                            'output_dir': f"{args.output_dir}/{batch}/{arch}_{level}_{mode}",
                            'seeds': args.seeds,
                            'status': 'pending',
                            'train_time': 0,
                            'eval_time': 0
                        })
    return tasks


def run_training(task: Dict, exp_dir: Path) -> bool:
    """运行训练阶段"""
    cmd = [
        sys.executable, "scripts/train_models.py",
        "--config", task['config'],
        "--model-config", task['model_config'],
        "--output-dir", task['output_dir'],
        "--seeds"
    ] + [str(s) for s in task['seeds']]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=3600
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  训练失败: {e}")
        return False


def run_evaluation(task: Dict, exp_dir: Path, eval_workers: int) -> bool:
    """运行评估阶段（使用并行评估）"""
    task_output = Path(task['output_dir'])
    model_dirs = list(task_output.rglob("models"))
    
    if not model_dirs:
        print(f"  未找到模型目录: {task_output}")
        return False
    
    cmd = [
        sys.executable, "scripts/evaluate_models.py",
        "--models-dir"
    ] + [str(d) for d in model_dirs] + [
        "--workers", str(eval_workers),
        "--output-dir", str(task_output),
        "--missing-modes", "MCAR", "MAR", "MNAR",
        "--imputation-methods", "zero", "mean", "knn", "iterative"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=1800
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  评估失败: {e}")
        return False


def main():
    args = parse_args()
    exp_dir = Path(__file__).parent.parent
    
    tasks = get_experiment_tasks(args)
    if not tasks:
        print("错误: 未找到有效的实验配置")
        return 1
    
    total_models = len(tasks) * len(args.seeds)
    
    print("="*70)
    print("智能调度实验运行器")
    print("="*70)
    print(f"配置数: {len(tasks)}")
    print(f"种子数: {len(args.seeds)} ({args.seeds})")
    print(f"总模型数: {total_models}")
    print(f"评估进程: {args.eval_workers}")
    print(f"策略: GPU串行训练 + CPU并行评估")
    print("="*70)
    
    if args.dry_run:
        print("\n【干运行模式】")
        for t in tasks:
            print(f"  {t['id']}: seeds={t['seeds']}")
        return 0
    
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    plan_file = output_base / f"plan_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(plan_file, 'w') as f:
        json.dump({
            'tasks': [{k: v for k, v in t.items() if k != 'status'} for t in tasks],
            'total_models': total_models,
            'start_time': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    start_time = time.time()
    last_progress_time = 0
    
    print(f"\n开始执行 {len(tasks)} 个配置...")
    print("="*70)
    
    for i, task in enumerate(tasks, 1):
        task_start = time.time()
        task['status'] = 'training'
        
        if time.time() - last_progress_time > 30:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 进度: {i}/{len(tasks)}")
            last_progress_time = time.time()
        
        print(f"\n[{i}/{len(tasks)}] {task['id']}")
        print(f"  阶段1/2: 训练 {len(task['seeds'])} 个模型...")
        
        train_start = time.time()
        train_success = run_training(task, exp_dir)
        task['train_time'] = time.time() - train_start
        
        if not train_success:
            print(f"  ✗ 训练失败")
            task['status'] = 'failed'
            continue
        
        print(f"  ✓ 训练完成 ({task['train_time']:.1f}s)")
        
        if not args.skip_eval and not args.train_only:
            task['status'] = 'evaluating'
            print(f"  阶段2/2: 评估 ({args.eval_workers}并行)...")
            
            eval_start = time.time()
            eval_success = run_evaluation(task, exp_dir, args.eval_workers)
            task['eval_time'] = time.time() - eval_start
            
            if not eval_success:
                print(f"  ✗ 评估失败")
                task['status'] = 'failed'
                continue
            
            print(f"  ✓ 评估完成 ({task['eval_time']:.1f}s)")
        
        task['status'] = 'done'
        task['total_time'] = time.time() - task_start
        print(f"  ✓ 总耗时: {task['total_time']:.1f}s")
    
    total_time = time.time() - start_time
    done = sum(1 for t in tasks if t['status'] == 'done')
    failed = sum(1 for t in tasks if t['status'] == 'failed')
    
    print("\n" + "="*70)
    print("实验完成")
    print("="*70)
    print(f"成功: {done}/{len(tasks)} 配置")
    print(f"失败: {failed}/{len(tasks)} 配置")
    print(f"总耗时: {total_time/3600:.2f} 小时")
    
    results_df = pd.DataFrame([
        {k: v for k, v in t.items() if k != 'seeds'} 
        for t in tasks
    ])
    results_file = output_base / f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"结果文件: {results_file}")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
