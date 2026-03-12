#!/usr/bin/env python3
"""
验证论文配置文件是否正确

使用方法:
    python scripts/verify_paper_configs.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_file_exists(filepath: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description}: {filepath} (缺失)")
        return False


def check_yaml_parse(filepath: str) -> bool:
    """检查YAML是否能正确解析"""
    try:
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True
    except Exception as e:
        print(f"    ✗ YAML解析错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("论文配置文件验证")
    print("=" * 70)
    
    configs_dir = Path("configs/paper")
    
    # 检查文件结构
    print("\n[1/3] 检查文件结构...")
    files_to_check = [
        (configs_dir / "common.yaml", "通用配置"),
        (configs_dir / "run_mlp_baseline.yaml", "MLP Baseline快速运行配置"),
        (configs_dir / "run_mlp_mim.yaml", "MLP MIM快速运行配置"),
        (configs_dir / "README.md", "说明文档"),
        ("configs/model/paper_mlp.yaml", "MLP模型配置"),
        ("configs/model/paper_lstm.yaml", "LSTM模型配置"),
        ("configs/model/paper_gru.yaml", "GRU模型配置"),
        ("configs/model/paper_cnn1d.yaml", "CNN1D模型配置"),
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    if not all_exist:
        print("\n✗ 部分配置文件缺失！")
        return 1
    
    # 检查YAML格式
    print("\n[2/3] 检查YAML格式...")
    yaml_files = [
        configs_dir / "common.yaml",
        configs_dir / "run_mlp_baseline.yaml",
        configs_dir / "run_mlp_mim.yaml",
    ] + [
        Path("configs/model") / f"paper_{m}.yaml" 
        for m in ["mlp", "lstm", "gru", "cnn1d"]
    ]
    
    all_yaml_valid = True
    for yaml_file in yaml_files:
        if not check_yaml_parse(yaml_file):
            all_yaml_valid = False
    
    if all_yaml_valid:
        print("  ✓ 所有YAML文件格式正确")
    else:
        print("\n✗ 部分YAML文件格式错误！")
        return 1
    
    # 检查关键配置项
    print("\n[3/3] 检查关键配置项...")
    try:
        import yaml
        
        # 检查 common.yaml
        with open(configs_dir / "common.yaml", 'r', encoding='utf-8') as f:
            common = yaml.safe_load(f)
        
        checks = [
            ("seeds" in common and len(common.get("seeds", [])) == 100, "100个随机种子"),
            ("missing_rates" in common and len(common.get("missing_rates", [])) == 9, "9个缺失率"),
            ("training" in common and common["training"].get("learning_rate") == 0.001, "学习率0.001"),
            ("training" in common and common["training"].get("batch_size") == 64, "BatchSize 64"),
            ("training" in common and common["training"].get("max_epochs") == 200, "最大Epoch 200"),
        ]
        
        for check, description in checks:
            if check:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description}")
        
        # 检查模型配置
        for model_name in ["mlp", "lstm", "gru", "cnn1d"]:
            model_file = Path("configs/model") / f"paper_{model_name}.yaml"
            with open(model_file, 'r', encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)
            
            if model_cfg.get("name") == model_name:
                print(f"  ✓ {model_name.upper()}: model.name 正确")
            else:
                print(f"  ✗ {model_name.upper()}: model.name 错误")
        
    except Exception as e:
        print(f"  ✗ 配置项检查失败: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("✓ 所有论文配置文件验证通过！")
    print("=" * 70)
    print("\n使用示例:")
    print("  python src/main.py --config-name=paper/run_mlp_mim")
    print("  python src/main.py model=paper_mlp method=mim experiment.seeds=[42]")
    
    return 0


if __name__ == "__main__":
    exit(main())
