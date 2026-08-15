"""TEMPLATE 评估脚本

用法示例:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint results/best_model.pth
"""
import argparse
import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="TEMPLATE evaluation script")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # TODO: 实现评估流程
    print(f"Evaluating {cfg['experiment']['name']} with checkpoint {args.checkpoint}")
    print("TODO: implement evaluation loop.")


if __name__ == "__main__":
    main()
