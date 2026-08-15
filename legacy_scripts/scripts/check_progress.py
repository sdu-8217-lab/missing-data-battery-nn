#!/usr/bin/env python3
"""检查当前所有实验的进度。

用法：
    python scripts/check_progress.py
"""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("results")


def count_models_in_config(cfg_path: Path):
    """从 config.json 读取 n_repeats，默认 12 个模型。"""
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_repeats = cfg.get("n_repeats", 0)
    return 12 * n_repeats


def parse_progress_from_log(log_path: Path):
    """从日志中解析已完成多少次 single experiment。"""
    if not log_path.exists():
        return 0
    text = log_path.read_text(errors="ignore")
    return text.count("开始实验:")


def infer_total(log_path: Path):
    """从日志中推断总实验次数。"""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    m = re.search(r"重复实验次数[:：]\s*(\d+)", text)
    if m:
        return 12 * int(m.group(1))
    return None


def is_process_running(exp_dir: Path):
    """检查该实验是否仍有对应 python 进程在运行。"""
    log_files = list((exp_dir / "logs").glob("*.log")) if (exp_dir / "logs").exists() else []
    if not log_files:
        return False
    try:
        procs = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return False
    # 命令行里 --results_dir 指向的是父目录（如 ./results/stage1_bernoulli/xjtu_3c）
    # 实际实验子目录是 results_dir/timestamp_batch
    parent = exp_dir.parent
    for p in [exp_dir, parent]:
        for s in [str(p), str(p.resolve()), f"./{p}"]:
            if s in procs:
                return True
    return False


def check_experiment(exp_dir: Path):
    """检查单个实验目录的进度。"""
    cfg_path = exp_dir / "config.json"
    log_files = list((exp_dir / "logs").glob("*.log")) if (exp_dir / "logs").exists() else []
    csv_files = list((exp_dir / "results").glob("experiment_results_*.csv"))

    total = count_models_in_config(cfg_path)
    done = 0
    log_path = log_files[0] if log_files else None

    if csv_files:
        df = pd.read_csv(csv_files[0])
        done = df[["model", "seed"]].drop_duplicates().shape[0]
    elif log_path:
        done = parse_progress_from_log(log_path)
        if total is None:
            total = infer_total(log_path)

    running = is_process_running(exp_dir)
    status = "已完成" if csv_files and total and done >= total else ("运行中" if running else "已停止")
    pct = f"{done / total * 100:.1f}%" if total else "unknown"
    return {
        "path": str(exp_dir.relative_to(RESULTS_ROOT)),
        "status": status,
        "done": done,
        "total": total,
        "pct": pct,
    }


def find_experiment_dirs():
    """找到所有实际实验子目录（timestamp_batch 层）。"""
    dirs = []
    for pattern in ["stage1_bernoulli/*/*", "stage2_patterns/*/*"]:
        for p in RESULTS_ROOT.glob(pattern):
            # 只保留包含 config.json 或 logs 的 timestamp_batch 目录
            if p.is_dir() and ((p / "config.json").exists() or (p / "logs").exists()):
                dirs.append(p)
    return dirs


def main():
    if not RESULTS_ROOT.exists():
        print(f"{RESULTS_ROOT} 不存在")
        return

    exp_dirs = find_experiment_dirs()
    if not exp_dirs:
        print("未找到实验目录")
        return

    rows = [check_experiment(d) for d in exp_dirs]

    print(f"{'实验目录':<55} {'状态':<8} {'完成':>6} {'总计':>6} {'进度':>8}")
    print("-" * 90)
    for r in rows:
        total_str = str(r["total"]) if r["total"] is not None else "?"
        print(f"{r['path']:<55} {r['status']:<8} {r['done']:>6} {total_str:>6} {r['pct']:>8}")


if __name__ == "__main__":
    main()
