"""运行缺失数据实验

覆盖论文 Scenario 1/2：
- 原始轨迹随机缺失 30%、50%
- 原始轨迹连续缺失 30%、50%

用法:
    python scripts/run_missing_experiments.py --config configs/nasa.yaml --model_type batteryCDE
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
    parser = argparse.ArgumentParser(description="Run missing data experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="batteryCDE")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    # 读取基础配置
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    experiments = [
        {"missing_rate": 0.0, "missing_pattern": "random", "suffix": "clean"},
        {"missing_rate": 0.3, "missing_pattern": "random", "suffix": "random30"},
        {"missing_rate": 0.5, "missing_pattern": "random", "suffix": "random50"},
        {"missing_rate": 0.3, "missing_pattern": "block", "suffix": "block30"},
        {"missing_rate": 0.5, "missing_pattern": "block", "suffix": "block50"},
    ]

    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["missing_rate"] = exp["missing_rate"]
        cfg["data"]["missing_pattern"] = exp["missing_pattern"]
        cfg["experiment"]["output_dir"] = os.path.join(
            "./results/missing",
            args.model_type,
            exp["suffix"],
        )

        # 写入临时配置
        tmp_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", f"_tmp_missing_{exp['suffix']}.yaml"
        )
        tmp_config_path = os.path.abspath(tmp_config_path)
        with open(tmp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

        print(f"\n{'='*60}")
        print(f"Missing experiment: {exp['suffix']}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            script_path,
            "--config", tmp_config_path,
            "--model_type", args.model_type,
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=True)

        # 清理临时配置
        os.remove(tmp_config_path)


if __name__ == "__main__":
    main()
