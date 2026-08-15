"""运行 BatteryCDE 复现的全部对比实验

用法:
    python scripts/run_all_experiments.py --config configs/nasa.yaml --seed 42
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all BatteryCDE reproduction experiments")
    parser.add_argument("--config", type=str, default="configs/nasa.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_cmd(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "train.py")
    base_cmd = [sys.executable, script_path, "--config", args.config, "--seed", str(args.seed)]

    # 1. 基线对比
    print("\n" + "=" * 70)
    print("PART 1: Baseline comparison")
    print("=" * 70)
    for model_type in ["lstm", "transformer", "neuralCDE"]:
        run_cmd(base_cmd + ["--model_type", model_type])

    # 2. 不同 horizon
    print("\n" + "=" * 70)
    print("PART 2: Horizon experiments")
    print("=" * 70)
    horizon_script = os.path.join(os.path.dirname(__file__), "run_horizon_experiments.py")
    run_cmd([sys.executable, horizon_script, "--config", args.config,
             "--model_type", "batteryCDE", "--seed", str(args.seed)])

    # 3. 缺失数据
    print("\n" + "=" * 70)
    print("PART 3: Missing data experiments")
    print("=" * 70)
    missing_script = os.path.join(os.path.dirname(__file__), "run_missing_experiments.py")
    run_cmd([sys.executable, missing_script, "--config", args.config,
             "--model_type", "batteryCDE", "--seed", str(args.seed)])

    # 4. 迁移学习
    print("\n" + "=" * 70)
    print("PART 4: Transfer experiments")
    print("=" * 70)
    transfer_script = os.path.join(os.path.dirname(__file__), "run_transfer_experiments.py")
    run_cmd([sys.executable, transfer_script, "--config", args.config,
             "--model_type", "batteryCDE", "--seed", str(args.seed)])

    # 5. 汇总与绘图
    print("\n" + "=" * 70)
    print("PART 5: Aggregate and plot")
    print("=" * 70)
    aggregate_script = os.path.join(os.path.dirname(__file__), "aggregate_results.py")
    plot_script = os.path.join(os.path.dirname(__file__), "plot_results.py")
    run_cmd([sys.executable, aggregate_script,
             "--results_dir", "../results",
             "--output", "../results/summary.csv"])
    run_cmd([sys.executable, plot_script,
             "--results_dir", "../results",
             "--output_dir", "../results/figures"])

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
