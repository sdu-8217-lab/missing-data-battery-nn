#!/usr/bin/env python3
"""
完整版大实验 - 批量执行脚本
规模: 4 datasets × 2 modes × 5 methods × 4 models × 100 seeds = 16,000 次训练
"""

import subprocess
import time
import json
import os
from datetime import datetime
from pathlib import Path

# 实验配置
CONFIG = {
    "datasets": ["xjtu", "tju", "hust", "mit"],
    "modes": ["mcar", "mar"],
    "methods": ["mim", "mean", "median", "knn", "zero"],
    "models": ["mlp", "lstm", "gru", "cnn1d"],
    "seeds": list(range(42, 142)),  # 100 seeds: 42-141
    "epochs": 100,
}

# 检查点文件
CHECKPOINT_FILE = "results/experiment_checkpoint.json"
LOG_FILE = "results/full_experiment.log"

def load_checkpoint():
    """加载检查点"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": [],
        "start_time": datetime.now().isoformat(),
        "total_experiments": (
            len(CONFIG["datasets"]) *
            len(CONFIG["modes"]) *
            len(CONFIG["methods"]) *
            len(CONFIG["models"])
        )
    }

def save_checkpoint(checkpoint):
    """保存检查点"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def log_message(msg):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")

def run_single_experiment(dataset, mode, method, model, seeds, epochs):
    """运行单次实验配置（跨所有种子）"""
    config_key = f"{dataset}_{mode}_{method}_{model}"
    
    # 构建命令
    seeds_str = ",".join(map(str, seeds))
    cmd = [
        "python", "src/main.py",
        f"data={dataset}",
        f"missing={mode}",
        f"method={method}",
        f"model={model}",
        f"experiment.seeds=[{seeds_str}]",
        f"training.epochs={epochs}",
        "wandb.enabled=false"
    ]
    
    log_message(f"Starting: {config_key} ({len(seeds)} seeds)")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4  # 4小时超时
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_message(f"✓ Completed: {config_key} in {elapsed/60:.1f} min")
            return True
        else:
            log_message(f"✗ Failed: {config_key} - {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        log_message(f"✗ Timeout: {config_key} (>4 hours)")
        return False
    except Exception as e:
        log_message(f"✗ Error: {config_key} - {str(e)}")
        return False

def main():
    """主函数"""
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])
    failed = set(checkpoint["failed"])
    
    log_message("=" * 60)
    log_message("完整版大实验启动")
    log_message(f"总配置数: {checkpoint['total_experiments']}")
    log_message(f"每配置种子数: {len(CONFIG['seeds'])}")
    log_message(f"已完成: {len(completed)}")
    log_message(f"已失败: {len(failed)}")
    log_message("=" * 60)
    
    # 生成所有实验配置
    experiments = []
    for dataset in CONFIG["datasets"]:
        for mode in CONFIG["modes"]:
            for method in CONFIG["methods"]:
                for model in CONFIG["models"]:
                    config_key = f"{dataset}_{mode}_{method}_{model}"
                    experiments.append((dataset, mode, method, model, config_key))
    
    # 执行实验
    total = len(experiments)
    for i, (dataset, mode, method, model, config_key) in enumerate(experiments, 1):
        if config_key in completed:
            log_message(f"[{i}/{total}] Skipping (completed): {config_key}")
            continue
            
        if config_key in failed:
            log_message(f"[{i}/{total}] Retrying (failed): {config_key}")
        else:
            log_message(f"[{i}/{total}] Running: {config_key}")
        
        # 运行实验
        success = run_single_experiment(
            dataset, mode, method, model,
            CONFIG["seeds"], CONFIG["epochs"]
        )
        
        if success:
            completed.add(config_key)
            checkpoint["completed"] = list(completed)
        else:
            failed.add(config_key)
            checkpoint["failed"] = list(failed)
        
        save_checkpoint(checkpoint)
        
        # 进度报告
        progress = len(completed) / total * 100
        log_message(f"Progress: {len(completed)}/{total} ({progress:.1f}%)")
        log_message("")
    
    # 完成报告
    log_message("=" * 60)
    log_message("实验完成!")
    log_message(f"成功: {len(completed)}/{total}")
    log_message(f"失败: {len(failed)}/{total}")
    log_message("=" * 60)

if __name__ == "__main__":
    main()
