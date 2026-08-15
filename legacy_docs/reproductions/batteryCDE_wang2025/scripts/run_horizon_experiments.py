"""运行不同预测步长（horizon）实验

用法:
    python scripts/run_horizon_experiments.py --config configs/nasa.yaml --model_type batteryCDE
"""
import argparse
import copy
import os
import subprocess
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run horizon experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="batteryCDE")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    horizons = [20, 50, 80, 100]

    for horizon in horizons:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["horizon"] = horizon
        cfg["experiment"]["output_dir"] = os.path.join(
            "./results/horizon",
            args.model_type,
            f"h{horizon}",
        )

        tmp_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", f"_tmp_horizon_h{horizon}.yaml"
        )
        tmp_config_path = os.path.abspath(tmp_config_path)
        with open(tmp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

        print(f"\n{'='*60}")
        print(f"Horizon experiment: h={horizon}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            script_path,
            "--config", tmp_config_path,
            "--model_type", args.model_type,
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=True)
        os.remove(tmp_config_path)


if __name__ == "__main__":
    main()
