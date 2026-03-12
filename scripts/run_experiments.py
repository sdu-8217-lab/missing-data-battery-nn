#!/usr/bin/env python
"""
Batch experiment runner for paper reproduction.

循环层级设计（从内到外重要性递增）：
- 最内层（核心）: method (baseline/mim) - 必须完整对比
- 中间层: model (mlp/lstm/gru/cnn1d)
- 最外层: seed (42-141) - 重复性

一个最小完整实验 = (seed, model, both_methods)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scheduler import ExperimentScheduler, SchedulerConfig


def generate_timestamp():
    """Generate timestamp for experiment run: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_structure(timestamp: str) -> Path:
    """Create complete experiment directory structure"""
    run_dir = Path(f"experiments/runs/{timestamp}")
    
    # Create subdirectories
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    
    # Create .gitkeep files for git tracking
    for subdir in ["logs", "results", "checkpoints", "figures"]:
        (run_dir / subdir / ".gitkeep").touch()
    
    return run_dir


def cmd_init(args):
    """Initialize experiment database.
    
    循环层级：seed(outer) → model(middle) → method(inner/core)
    自动生成时间戳，创建独立实验目录结构
    """
    # 自动生成时间戳
    timestamp = generate_timestamp()
    
    # 创建完整实验目录结构
    run_dir = create_experiment_structure(timestamp)
    
    # 创建/更新latest软链接
    latest_link = Path("experiments/latest")
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.relative_to(latest_link.parent), target_is_directory=True)
    
    print(f"Experiment timestamp: {timestamp}")
    print(f"Run directory: {run_dir}")
    
    # 使用新的数据库路径
    db_path = str(run_dir / "experiment_db.csv")
    scheduler = ExperimentScheduler(SchedulerConfig(db_path=db_path, log_dir=str(run_dir / "logs")))
    
    # 测试集缺失率：10档 [0.0, 0.1, ..., 0.9]
    eval_missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # MIM训练集缺失率：与测试集严格相同，10份合并训练
    mim_train_missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if args.all:
        # Full paper reproduction: 100 seeds × 4 models × 2 methods = 800 runs
        # 但最小完整实验单位 = (seed, model) 对，包含两种方法
        # 所以完整实验 = 100 seeds × 4 models = 400 个对比实验单元
        seeds = list(range(42, 142))  # 100 seeds
        models = ["mlp", "lstm", "gru", "cnn1d"]
        methods = ["baseline", "mim"]  # 必须完整
        
        total_runs = len(seeds) * len(models) * len(methods)
        total_comparison_units = len(seeds) * len(models)  # 400
        
        print(f"Initializing FULL experiment batch:")
        print(f"  Loop order: seed(outer) → model(middle) → method(inner)")
        print(f"  Seeds: {seeds[0]} to {seeds[-1]} ({len(seeds)}) [outer]")
        print(f"  Models: {models} [middle]")
        print(f"  Methods: {methods} [inner/core - must be complete]")
        print(f"  Total runs: {total_runs}")
        print(f"  Comparison units: {total_comparison_units} (each tests both methods)")
        print(f"  Eval MRs: {eval_missing_rates}")
        print(f"  MIM train MRs: {mim_train_missing_rates} (merged)")
        
        scheduler.initialize_experiments(
            seeds=seeds,
            models=models,
            methods=methods,
            eval_missing_rates=eval_missing_rates,
            mim_train_missing_rates=mim_train_missing_rates,
            timestamp=timestamp,
            run_dir=str(run_dir),
            clear_existing=args.clear
        )
    
    elif args.test:
        # Test: 3 seeds × 2 models × 2 methods = 12 runs
        seeds = [42, 43, 44]
        models = ["mlp", "lstm"]
        methods = ["baseline", "mim"]
        
        # 简化测试用3个MR
        test_eval_mrs = [0.0, 0.5, 0.9]
        test_train_mrs = [0.0, 0.5, 0.9]
        
        total_runs = len(seeds) * len(models) * len(methods)
        
        print(f"Initializing TEST batch:")
        print(f"  Seeds: {seeds}")
        print(f"  Models: {models}")
        print(f"  Methods: {methods}")
        print(f"  Total runs: {total_runs}")
        print(f"  Eval MRs: {test_eval_mrs}")
        
        scheduler.initialize_experiments(
            seeds=seeds,
            models=models,
            methods=methods,
            eval_missing_rates=test_eval_mrs,
            mim_train_missing_rates=test_train_mrs,
            timestamp=timestamp,
            run_dir=str(run_dir),
            clear_existing=True
        )
    
    elif args.mini:
        # Mini test: 2 seeds × 1 model × 2 methods = 4 runs
        # 最小完整实验：一个seed + 一个model + 两种方法
        seeds = [42, 43]
        models = ["mlp"]
        methods = ["baseline", "mim"]
        
        print(f"Initializing MINI batch (smallest complete experiment):")
        print(f"  Seeds: {seeds} [outer]")
        print(f"  Models: {models} [middle]")
        print(f"  Methods: {methods} [inner/core]")
        print(f"  Each (seed, model) tests both methods for fair comparison")
        
        scheduler.initialize_experiments(
            seeds=seeds,
            models=models,
            methods=methods,
            eval_missing_rates=eval_missing_rates,
            mim_train_missing_rates=mim_train_missing_rates,
            timestamp=timestamp,
            run_dir=str(run_dir),
            clear_existing=True
        )
    
    else:
        print("Use --all, --test, or --mini")
        return 1
    
    return 0


def get_latest_run_dir():
    """Get the latest run directory from symlink."""
    latest_link = Path("experiments/latest")
    if latest_link.exists() and latest_link.is_symlink():
        return latest_link.resolve()
    return None


def cmd_run(args):
    """Run experiments."""
    use_gpu = not args.no_gpu
    
    if use_gpu:
        gpu_workers = args.gpu_workers if args.gpu_workers is not None else 2
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else 2
    else:
        gpu_workers = 0
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else 4
    
    # 获取最新的运行目录
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("Error: No experiment run found. Run 'init' first.")
        return 1
    
    db_path = run_dir / "experiment_db.csv"
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    print(f"Using experiment run: {run_dir.name}")
    print(f"Database: {db_path}")
    
    config = SchedulerConfig(
        db_path=str(db_path),
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        use_gpu=use_gpu,
        report_interval_seconds=args.report_interval if args.report_interval is not None else 60
    )
    
    scheduler = ExperimentScheduler(config)
    
    if args.once:
        scheduler.run_batch()
    else:
        scheduler.run_all(continuous=True)
    
    return 0


def cmd_status(args):
    """Show current status."""
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("Error: No experiment run found. Run 'init' first.")
        return 1
    
    db_path = run_dir / "experiment_db.csv"
    config = SchedulerConfig(db_path=str(db_path))
    scheduler = ExperimentScheduler(config)
    status = scheduler.get_status()
    
    print(f"\n📊 Experiment Status ({run_dir.name})")
    print("=" * 40)
    print(f"Total runs: {status['total']:5d}")
    print(f"Pending:    {status['pending']:5d}")
    print(f"Running:    {status['running']:5d}")
    print(f"Completed:  {status['completed']:5d}")
    print(f"Failed:     {status['failed']:5d}")
    print("=" * 40)
    print(f"Progress:   {status['progress_percent']:.1f}%")
    
    # 计算完整对比单元
    total_units = status['total'] // 2 if status['total'] > 0 else 0
    completed_units = status['completed'] // 2 if status['completed'] > 0 else 0
    if total_units > 0:
        print(f"Comparison units: {completed_units}/{total_units}")
    
    if status['failed'] > 0:
        print(f"\n⚠️  {status['failed']} runs failed. Use 'retry' to re-run them.")
    
    return 0


def cmd_retry(args):
    """Retry failed experiments."""
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("Error: No experiment run found.")
        return 1
    
    db_path = run_dir / "experiment_db.csv"
    config = SchedulerConfig(db_path=str(db_path))
    scheduler = ExperimentScheduler(config)
    count = scheduler.retry_failed()
    
    if count > 0:
        print(f"✅ Reset {count} failed runs. Run 'run' to execute them.")
    else:
        print("No failed runs to retry.")
    
    return 0


def cmd_recover(args):
    """Recover from crash (reset running to pending)."""
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("Error: No experiment run found.")
        return 1
    
    db_path = run_dir / "experiment_db.csv"
    config = SchedulerConfig(db_path=str(db_path))
    scheduler = ExperimentScheduler(config)
    count = scheduler.recover()
    
    if count > 0:
        print(f"✅ Recovered {count} stuck runs.")
    else:
        print("No stuck runs found.")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner with proper nesting: seed→model→method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (4 runs = 2 seeds × 1 model × 2 methods)
  python scripts/run_experiments.py init --mini && python scripts/run_experiments.py run
  
  # Small test (12 runs = 3 seeds × 2 models × 2 methods)
  python scripts/run_experiments.py init --test && python scripts/run_experiments.py run
  
  # Full paper reproduction (800 runs = 100 seeds × 4 models × 2 methods)
  python scripts/run_experiments.py init --all
  python scripts/run_experiments.py run --gpu-workers 1 --cpu-workers 3
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize experiment database")
    init_group = init_parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--all", action="store_true", help="Full reproduction (800 runs)")
    init_group.add_argument("--test", action="store_true", help="Test batch (12 runs)")
    init_group.add_argument("--mini", action="store_true", help="Mini test (4 runs)")
    init_parser.add_argument("--clear", action="store_true", help="Clear existing database")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--workers", "-w", type=int, help="Total workers (legacy)")
    run_parser.add_argument("--gpu-workers", "-g", type=int, default=1, help="GPU workers (default: 1)")
    run_parser.add_argument("--cpu-workers", "-c", type=int, default=3, help="CPU workers (default: 3)")
    run_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    run_parser.add_argument("--once", action="store_true", help="Run single batch only")
    run_parser.add_argument("--report-interval", type=int, default=60, help="Status report interval (seconds)")
    
    # status command
    subparsers.add_parser("status", help="Show experiment status")
    
    # retry command
    subparsers.add_parser("retry", help="Retry failed experiments")
    
    # recover command
    subparsers.add_parser("recover", help="Recover from crash")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "status": cmd_status,
        "retry": cmd_retry,
        "recover": cmd_recover
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
