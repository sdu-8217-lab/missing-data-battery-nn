#!/usr/bin/env python3
"""
批次工作脚本 - 执行单个批次的所有配置
独立运行或供调度器调用

用法:
    python scripts/batch_worker.py --batch R2.5 --seeds 0 1 2 ...
"""
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='批次工作脚本')
    parser.add_argument('--batch', type=str, required=True, help='批次名称')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='种子列表')
    parser.add_argument('--output-dir', default='./results/10seeds_smart', help='输出目录')
    parser.add_argument('--config-dir', default='configs/experiments/batch_configs_standard', 
                       help='配置目录')
    parser.add_argument('--model-config-dir', default='configs/models', help='模型配置目录')
    return parser.parse_args()


def get_tasks(args) -> List[dict]:
    """生成任务列表"""
    batch = args.batch
    archs = ["mlp", "lstm", "cnn"]
    levels = ["level_1", "level_2", "level_3", "level_4"]
    modes = ["mim", "baseline"]
    
    tasks = []
    config_dir = Path(args.config_dir)
    model_config_dir = Path(args.model_config_dir)
    
    for arch in archs:
        for level in levels:
            for mode in modes:
                config_file = config_dir / f"batch_{batch}_{arch}_{mode}.yaml"
                model_config = model_config_dir / f"{arch}_{level}.yaml"
                
                if config_file.exists() and model_config.exists():
                    tasks.append({
                        'id': f"{batch}_{arch}_{level}_{mode}",
                        'config': str(config_file),
                        'model_config': str(model_config),
                        'output_dir': f"{args.output_dir}/{batch}/{arch}_{level}_{mode}"
                    })
    return tasks


def run_training(task: dict, seeds: List[int]) -> bool:
    """运行训练任务"""
    cmd = [
        sys.executable, "scripts/train_models.py",
        "--config", task['config'],
        "--model-config", task['model_config'],
        "--output-dir", task['output_dir'],
        "--seeds"
    ] + [str(s) for s in seeds]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return result.returncode == 0


def main():
    args = parse_args()
    tasks = get_tasks(args)
    
    print(f"批次: {args.batch}")
    print(f"配置数: {len(tasks)}")
    print(f"种子数: {len(args.seeds)}")
    print("=" * 50)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['id']}")
        print("  训练中...")
        
        success = run_training(task, args.seeds)
        
        if success:
            print("  ✓ 完成")
        else:
            print("  ✗ 失败")
    
    print("\n批次完成")


if __name__ == "__main__":
    main()
