"""运行迁移实验

当前实现：同数据集跨电池迁移（NASA 内部）。
用法:
    python scripts/run_transfer_experiments.py --config configs/nasa.yaml --model_type batteryCDE
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
    parser = argparse.ArgumentParser(description="Run transfer experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="batteryCDE")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # 定义迁移场景：源域训练电池 -> 目标域测试电池
    scenarios = [
        {
            "name": "B5B6_to_B7",
            "train_batteries": ["B0005", "B0006"],
            "val_batteries": ["B0007"],
            "test_batteries": ["B0007"],
        },
        {
            "name": "B5B6_to_B18",
            "train_batteries": ["B0005", "B0006"],
            "val_batteries": ["B0018"],
            "test_batteries": ["B0018"],
        },
        {
            "name": "B7B18_to_B5B6",
            "train_batteries": ["B0007", "B0018"],
            "val_batteries": ["B0005"],
            "test_batteries": ["B0006"],
        },
    ]

    for scenario in scenarios:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["train_batteries"] = scenario["train_batteries"]
        cfg["data"]["val_batteries"] = scenario["val_batteries"]
        cfg["data"]["test_batteries"] = scenario["test_batteries"]
        cfg["experiment"]["output_dir"] = os.path.join(
            "./results/transfer",
            args.model_type,
            scenario["name"],
        )

        tmp_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", f"_tmp_transfer_{scenario['name']}.yaml"
        )
        tmp_config_path = os.path.abspath(tmp_config_path)
        with open(tmp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

        print(f"\n{'='*60}")
        print(f"Transfer scenario: {scenario['name']}")
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
