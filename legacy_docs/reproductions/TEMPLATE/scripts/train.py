"""TEMPLATE 训练脚本

用法示例:
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import os
import sys
import yaml

# 将项目根目录加入路径，以便复用 src/utils 等
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="TEMPLATE training script")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    # TODO: 实现训练流程
    print(f"Config loaded: {cfg['experiment']['name']}")
    print("TODO: implement training loop.")


if __name__ == "__main__":
    main()
