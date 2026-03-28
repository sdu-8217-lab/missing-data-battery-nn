#!/usr/bin/env python3
"""
完整实验运行脚本（单配置）
包含训练和评估两个阶段

用法:
    python scripts/run_full_experiment.py \
        --config configs/experiments/batch_configs/batch_3C_mlp_mim.yaml \
        --model-config configs/models/mlp_level_1.yaml \
        --output-dir ./results/full_runs
"""
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config_eval_params(config_path: Path) -> dict:
    """从配置文件加载评估参数"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_config = config.get('evaluation', {})
    
    # 提取评估参数，使用默认值
    missing_rates = eval_config.get('missing_rates', [i * 0.1 for i in range(10)])
    missing_modes = eval_config.get('missing_modes', ['MCAR'])
    imputation_methods = eval_config.get('imputation_methods', ['zero'])
    
    return {
        'missing_rates': missing_rates,
        'missing_modes': missing_modes,
        'imputation_methods': imputation_methods
    }


def run_command(cmd: list, cwd: Path = None, description: str = "") -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令失败 (return code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="完整实验运行")
    parser.add_argument("--config", required=True, help="实验配置文件")
    parser.add_argument("--model-config", required=True, help="模型架构配置文件")
    parser.add_argument("--output-dir", default="./results/full_runs", help="输出目录")
    parser.add_argument("--data-dir", default="../data/raw/XJTU", help="数据目录")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练阶段")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估阶段")
    parser.add_argument("--seeds", type=int, nargs="+", help="随机种子列表（覆盖配置）")
    
    args = parser.parse_args()
    
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / args.config
    model_config_path = exp_dir / args.model_config
    
    # 验证文件
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return 1
    if not model_config_path.exists():
        print(f"错误: 模型配置不存在: {model_config_path}")
        return 1
    
    # 加载评估参数
    eval_params = load_config_eval_params(config_path)
    print(f"评估配置: {len(eval_params['missing_modes'])} modes × "
          f"{len(eval_params['missing_rates'])} MRs × "
          f"{len(eval_params['imputation_methods'])} imputations = "
          f"{len(eval_params['missing_modes']) * len(eval_params['missing_rates']) * len(eval_params['imputation_methods'])} 组合")
    
    print("="*60)
    print("完整实验运行")
    print("="*60)
    print(f"实验配置: {config_path}")
    print(f"模型配置: {model_config_path}")
    print(f"工作目录: {exp_dir}")
    print("="*60)
    
    # 阶段1: 训练
    if not args.skip_train:
        train_cmd = [
            sys.executable, "scripts/train_models.py",
            "--config", str(config_path),
            "--model-config", str(model_config_path),
            "--output-dir", args.output_dir
        ]
        if args.seeds:
            train_cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
        
        if not run_command(train_cmd, cwd=exp_dir, description="阶段1: 模型训练"):
            print("训练阶段失败，中止")
            return 1
    else:
        print("\n跳过训练阶段")
    
    # 阶段2: 评估
    if not args.skip_eval:
        # 查找训练输出的模型目录
        results_base = exp_dir / args.output_dir
        
        # 找到最新的训练结果
        latest_models_dir = None
        
        for subdir in results_base.rglob("models"):
            if subdir.is_dir() and any(subdir.glob("model_*.pt")):
                latest_models_dir = subdir
                break
        
        if latest_models_dir is None:
            print("错误: 未找到训练好的模型")
            return 1
        
        print(f"找到模型目录: {latest_models_dir}")
        
        # 构建评估命令，使用配置文件中的参数
        eval_cmd = [
            sys.executable, "scripts/evaluate_models.py",
            "--models-dir", str(latest_models_dir),
            "--data-dir", args.data_dir,
            "--missing-modes", *eval_params['missing_modes'],
            "--imputation-methods", *eval_params['imputation_methods'],
            "--missing-rates", *[str(mr) for mr in eval_params['missing_rates']]
        ]
        
        if not run_command(eval_cmd, cwd=exp_dir, description="阶段2: 模型评估"):
            print("评估阶段失败")
            return 1
    else:
        print("\n跳过评估阶段")
    
    print("\n" + "="*60)
    print("完整实验完成")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
