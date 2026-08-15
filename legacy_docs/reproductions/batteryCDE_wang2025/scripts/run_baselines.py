"""运行对比基线：LSTM / Transformer / Neural CDE

用法:
    python scripts/run_baselines.py --config configs/nasa.yaml
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


BASELINES = ["lstm", "transformer", "neuralCDE"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines for BatteryCDE")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    for model_type in BASELINES:
        print(f"\n{'='*60}")
        print(f"Training baseline: {model_type}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            script_path,
            "--config", args.config,
            "--model_type", model_type,
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
