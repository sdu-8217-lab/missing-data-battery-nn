"""
Debug script to check input dimensions for MIM vs Baseline
"""
import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf

# Load config files
cfg = OmegaConf.load("configs/config.yaml")
data_cfg = OmegaConf.load("configs/data/xjtu.yaml")
model_cfg = OmegaConf.load("configs/models/cnn1d.yaml")
missing_cfg = OmegaConf.load("configs/missing/mar.yaml")
exp_cfg = OmegaConf.load("configs/experiments/mim_mar_0.3.yaml")

# Merge properly
cfg = OmegaConf.merge(cfg, OmegaConf.create({
    "data": data_cfg,
    "models": model_cfg,
    "missing": missing_cfg,
    "experiments": exp_cfg
}))

from src.data.loader import load_dataset
from src.missing_data.mar import simulate_mar
from src.models.model_factory import create_model

# Load data
print("=" * 60)
print("Loading dataset...")
data = load_dataset(cfg)
X_train, y_train = data["X_train"], data["y_train"]
print(f"Original X_train shape: {X_train.shape}")  # Should be [N, 16]

# Test MIM
print("\n" + "=" * 60)
print("Testing MIM (use_mim=True)")
print("=" * 60)

X_imp, mask, mim_input = simulate_mar(X_train, y_train, missing_rate=0.3, beta=2.0, gamma=0.05, seed=42)
print(f"X_imputed shape: {X_imp.shape}")
print(f"mask shape: {mask.shape}")
print(f"mim_input shape: {mim_input.shape}")  # Should be [N, 32]

cfg.experiments.experiment.use_mim = True
model_mim = create_model(cfg)
print(f"Model input dim: {model_mim.backbone.input_dim}")

# Test Baseline
print("\n" + "=" * 60)
print("Testing Baseline (use_mim=False)")
print("=" * 60)

cfg.experiments.experiment.use_mim = False
model_baseline = create_model(cfg)
print(f"Model input dim: {model_baseline.backbone.input_dim}")

print("\n" + "=" * 60)
print("Summary:")
print(f"  MIM: model expects {model_mim.backbone.input_dim}, data is {mim_input.shape[1]}")
print(f"  Baseline: model expects {model_baseline.backbone.input_dim}, data is {X_imp.shape[1]}")
if model_mim.backbone.input_dim == mim_input.shape[1] and model_baseline.backbone.input_dim == X_imp.shape[1]:
    print("  ✓ Dimensions match!")
else:
    print("  ✗ DIMENSION MISMATCH!")
print("=" * 60)
