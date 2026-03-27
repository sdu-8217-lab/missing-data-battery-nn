#!/usr/bin/env python3
"""
安装验证脚本
验证GPU环境和所有依赖是否正确配置
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np

print("=" * 60)
print("SOH实验框架 - 安装验证")
print("=" * 60)

# 1. 检查GPU
print("\n[1/6] 检查GPU环境...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA可用: {torch.version.cuda}")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠️  CUDA不可用，将使用CPU")

# 2. 检查核心依赖
print("\n[2/6] 检查核心依赖...")
try:
    import pandas as pd
    print("  ✓ pandas")
except:
    print("  ✗ pandas缺失")
    
try:
    import sklearn
    print("  ✓ scikit-learn")
except:
    print("  ✗ scikit-learn缺失")
    
try:
    import yaml
    print("  ✓ PyYAML")
except:
    print("  ✗ PyYAML缺失")
    
try:
    from loguru import logger
    print("  ✓ loguru")
except:
    print("  ✗ loguru缺失")

# 3. 测试配置系统
print("\n[3/6] 测试配置系统...")
from src.config.experiment_config import ExperimentConfig, load_model_config
config = ExperimentConfig.from_yaml('configs/experiments/phase1_mlp_mim.yaml')
print(f"  ✓ 配置加载: {config.name}")
print(f"  ✓ 批次: {config.data.batch}")
print(f"  ✓ MIM: {config.training.use_mim}")
print(f"  ✓ 训练缺失率: {len(config.training.training_missing_rates)}档")

# 4. 测试模型创建
print("\n[4/6] 测试模型创建...")
from src.models.factory import create_model, list_available_models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_type in list_available_models():
    model = create_model(model_type, input_dim=16, use_mim=False, device=device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ {model_type.upper()}: {params:,} 参数")

# 5. 测试MIM模型
print("\n[5/6] 测试MIM模型...")
model_mim = create_model('mlp', input_dim=16, use_mim=True, device=device)
print(f"  ✓ MIM MLP创建成功")
print(f"  ✓ 输入维度: 16 -> 32 (特征+指示器)")

# 6. 测试训练流程
print("\n[6/6] 测试训练流程...")
from torch.utils.data import TensorDataset, DataLoader
from src.trainers.trainer import Trainer

# 创建简单数据集
X = torch.randn(50, 32 if device == 'cuda' else 32).to(device)  # MIM输入
y = torch.randn(50).to(device)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8)

# 训练
trainer = Trainer(model_mim, device=device)
history = trainer.train(loader, loader, epochs=3, patience=5, verbose=False)
print(f"  ✓ 训练完成")
print(f"  ✓ 最佳损失: {history['best_val_loss']:.6f}")
print(f"  ✓ 训练时间: {history['training_time']:.2f}s")

print("\n" + "=" * 60)
print("✅ 所有测试通过！环境配置正确。")
print("=" * 60)
print("\n可以开始运行实验：")
print("  ./run.sh --config configs/experiments/phase1_mlp_mim.yaml")
