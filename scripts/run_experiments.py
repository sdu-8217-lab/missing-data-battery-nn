#!/usr/bin/env python
"""
Batch experiment runner for paper reproduction.

Usage:
    # Initialize all 7200 experiments
    python scripts/run_experiments.py init --all
    
    # Initialize small test batch (24 experiments)
    python scripts/run_experiments.py init --test
    
    # Run all pending experiments
    python scripts/run_experiments.py run
    
    # Run with custom workers
    python scripts/run_experiments.py run --gpu-workers 2 --cpu-workers 2
    
    # Check status
    python scripts/run_experiments.py status
    
    # Retry failed experiments
    python scripts/run_experiments.py retry
    
    # Recover from crash (reset running -> pending)
    python scripts/run_experiments.py recover
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scheduler import ExperimentScheduler, SchedulerConfig


def cmd_init(args):
    """Initialize experiment database."""
    scheduler = ExperimentScheduler()
    
    if args.all:
        # Full paper reproduction: 4 models × 2 methods × 9 MRs × 100 seeds
        models = ["mlp", "lstm", "gru", "cnn1d"]
        methods = ["baseline", "mim"]
        missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        seeds = list(range(42, 142))  # 100 seeds
        
        total = len(models) * len(methods) * len(missing_rates) * len(seeds)
        print(f"Initializing {total} experiments:")
        print(f"  Models: {models}")
        print(f"  Methods: {methods}")
        print(f"  Missing rates: {missing_rates}")
        print(f"  Seeds: {seeds[0]} to {seeds[-1]} ({len(seeds)} seeds)")
        
        scheduler.initialize_experiments(
            models, methods, missing_rates, seeds,
            clear_existing=args.clear
        )
    
    elif args.test:
        # Small test: 2 models × 2 methods × 2 MRs × 3 seeds = 24 experiments
        models = ["mlp", "lstm"]
        methods = ["baseline", "mim"]
        missing_rates = [0.1, 0.5]
        seeds = [42, 43, 44]
        
        print("Initializing TEST batch (24 experiments):")
        scheduler.initialize_experiments(
            models, methods, missing_rates, seeds,
            clear_existing=True
        )
    
    elif args.mini:
        # Mini test: 1 model × 2 methods × 1 MR × 2 seeds = 4 experiments
        models = ["mlp"]
        methods = ["baseline", "mim"]
        missing_rates = [0.5]
        seeds = [42, 43]
        
        print("Initializing MINI batch (4 experiments):")
        scheduler.initialize_experiments(
            models, methods, missing_rates, seeds,
            clear_existing=True
        )
    
    else:
        print("Use --all for full paper reproduction, --test for 24 experiments, or --mini for 4 experiments")
        return 1
    
    return 0


def cmd_run(args):
    """Run experiments."""
    # Determine GPU/CPU workers based on arguments
    use_gpu = not args.no_gpu
    
    if use_gpu:
        gpu_workers = args.gpu_workers if args.gpu_workers is not None else 2
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else 2
    else:
        # When --no-gpu is specified, force GPU workers to 0
        gpu_workers = 0
        cpu_workers = args.cpu_workers if args.cpu_workers is not None else 4
    
    config = SchedulerConfig(
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        use_gpu=use_gpu,
        report_interval_seconds=args.report_interval if args.report_interval is not None else 60
    )
    
    scheduler = ExperimentScheduler(config)
    
    if args.once:
        # Run single batch
        scheduler.run_batch()
    else:
        # Run continuously until done
        scheduler.run_all(continuous=True)
    
    return 0


def cmd_status(args):
    """Show current status."""
    scheduler = ExperimentScheduler()
    status = scheduler.get_status()
    
    print("\n📊 Experiment Status")
    print("=" * 40)
    print(f"Total:      {status['total']:5d}")
    print(f"Pending:    {status['pending']:5d}")
    print(f"Running:    {status['running']:5d}")
    print(f"Completed:  {status['completed']:5d}")
    print(f"Failed:     {status['failed']:5d}")
    print("=" * 40)
    print(f"Progress:   {status['progress_percent']:.1f}%")
    print(f"Remaining:  {status['estimated_remaining']} experiments")
    
    if status['failed'] > 0:
        print(f"\n⚠️  {status['failed']} experiments failed. Use 'retry' to re-run them.")
    
    return 0


def cmd_retry(args):
    """Retry failed experiments."""
    scheduler = ExperimentScheduler()
    count = scheduler.retry_failed()
    
    if count > 0:
        print(f"✅ Reset {count} failed experiments. Run 'run' to execute them.")
    else:
        print("No failed experiments to retry.")
    
    return 0


def cmd_recover(args):
    """Recover from crash (reset running to pending)."""
    scheduler = ExperimentScheduler()
    count = scheduler.recover()
    
    if count > 0:
        print(f"✅ Recovered {count} stuck experiments.")
    else:
        print("No stuck experiments found.")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for paper reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (4 experiments)
  python scripts/run_experiments.py init --mini && python scripts/run_experiments.py run
  
  # Small test (24 experiments)
  python scripts/run_experiments.py init --test && python scripts/run_experiments.py run
  
  # Full paper reproduction (7200 experiments)
  python scripts/run_experiments.py init --all
  python scripts/run_experiments.py run --gpu-workers 2 --cpu-workers 2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize experiment database")
    init_group = init_parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--all", action="store_true", help="Full paper reproduction (7200 experiments)")
    init_group.add_argument("--test", action="store_true", help="Small test (24 experiments)")
    init_group.add_argument("--mini", action="store_true", help="Mini test (4 experiments)")
    init_parser.add_argument("--clear", action="store_true", help="Clear existing database")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--workers", "-w", type=int, help="Total workers (legacy)")
    run_parser.add_argument("--gpu-workers", "-g", type=int, default=2, help="GPU workers (default: 2)")
    run_parser.add_argument("--cpu-workers", "-c", type=int, default=2, help="CPU workers (default: 2)")
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
