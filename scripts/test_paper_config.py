#!/usr/bin/env python3
"""
测试 Paper 配置是否能被正确加载

使用方法:
    python scripts/test_paper_config.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

# 手动加载配置测试
def test_config_loading():
    """测试配置加载"""
    print("=" * 70)
    print("测试 Paper 配置加载")
    print("=" * 70)
    
    # 加载各个配置文件
    configs_to_test = [
        ("configs/paper/common.yaml", "通用配置"),
        ("configs/paper/experiment_baseline.yaml", "Baseline实验"),
        ("configs/paper/experiment_mim.yaml", "MIM实验"),
        ("configs/paper/model/mlp_baseline.yaml", "MLP Baseline模型"),
        ("configs/paper/model/mlp_mim.yaml", "MLP MIM模型"),
    ]
    
    for config_path, description in configs_to_test:
        print(f"\n测试: {description}")
        print(f"  文件: {config_path}")
        try:
            cfg = OmegaConf.load(config_path)
            print(f"  ✓ 加载成功")
            
            # 显示关键字段
            if 'model' in cfg:
                model_cfg = cfg.model
                print(f"    Model: {model_cfg.get('name')} / {model_cfg.get('type')}")
                print(f"    Input dim: {model_cfg.get('input_dim')}")
                if 'hidden_dims' in model_cfg:
                    print(f"    Hidden dims: {model_cfg.hidden_dims}")
                if 'expected_params' in model_cfg:
                    print(f"    Expected params: {model_cfg.expected_params:,}")
            
            if 'training' in cfg:
                train_cfg = cfg.training
                print(f"    LR: {train_cfg.get('learning_rate')}")
                print(f"    Epochs: {train_cfg.get('max_epochs')}")
                
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            return False
    
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 70)
    print("测试模型创建")
    print("=" * 70)
    
    sys.path.insert(0, 'src')
    from models.mlp import MLP
    from models.lstm import LSTM
    from models.gru import GRU
    from models.cnn1d import CNN1D
    
    tests = [
        ("MLP Baseline", lambda: MLP(16, hidden_dims=[192, 96, 48, 24], dropout=0.15), 27649),
        ("MLP MIM", lambda: MLP(32, hidden_dims=[192, 96, 48, 24], dropout=0.15), 30721),
        ("LSTM Baseline", lambda: LSTM(16, hidden_size=48, num_layers=2, dropout=0.2), 31537),
        ("GRU Baseline", lambda: GRU(16, hidden_size=64, num_layers=2, dropout=0.2), 40769),
    ]
    
    for name, model_fn, expected_params in tests:
        print(f"\n测试: {name}")
        try:
            model = model_fn()
            actual_params = sum(p.numel() for p in model.parameters())
            print(f"  Expected: {expected_params:,}")
            print(f"  Actual:   {actual_params:,}")
            if actual_params == expected_params:
                print(f"  ✓ 匹配")
            else:
                print(f"  ✗ 不匹配 (差异: {actual_params - expected_params:+d})")
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
            return False
    
    return True


def main():
    """主函数"""
    success = True
    
    if not test_config_loading():
        success = False
    
    if not test_model_creation():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✓ 所有测试通过！")
        print("=" * 70)
        print("\n可以运行实际实验:")
        print("  python src/main.py --config-name=paper/experiment_baseline model=mlp experiment.seeds=[42]")
        return 0
    else:
        print("✗ 部分测试失败")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
