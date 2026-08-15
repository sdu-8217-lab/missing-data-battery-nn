"""BatteryCDE 评估脚本

用法:
    python scripts/evaluate.py --config configs/nasa.yaml --checkpoint results/nasa/best_model.pth
"""
import argparse
import os
import sys
import json
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from reproductions.batteryCDE_wang2025.src.models.battery_cde import BatteryCDEModel  # noqa: E402
from reproductions.batteryCDE_wang2025.src.data.battery_cde_dataset import (  # noqa: E402
    BatteryCDEDataset,
    load_nasa_trajectories,
)
from reproductions.batteryCDE_wang2025.src.trainers.battery_cde_trainer import BatteryCDETrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BatteryCDE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["experiment"].get("device", "auto")
    device = torch.device(
        "cuda" if device_str == "auto" and torch.cuda.is_available() else (device_str or "cpu")
    )

    model_cfg = cfg["model"]
    model = BatteryCDEModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        attention_dim=model_cfg["attention_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        method=model_cfg["method"],
        options=model_cfg.get("options"),
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 加载 scaler
    output_dir = os.path.dirname(args.checkpoint)
    scaler_path = os.path.join(output_dir, "scaler.npz")
    scaler_data = np.load(scaler_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    capacity_scale = float(scaler_data["capacity_scale"])

    data_cfg = cfg["data"]
    trajectory_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "nasa_trajectories")
    )

    split_batteries = {
        "train": data_cfg["train_batteries"],
        "val": data_cfg["val_batteries"],
        "test": data_cfg["test_batteries"],
    }[args.split]
    cycles = load_nasa_trajectories(
        trajectory_dir, split_batteries, capacity_scale=capacity_scale
    )

    dataset = BatteryCDEDataset(
        cycles,
        horizon=data_cfg["horizon"],
        history_len=data_cfg["history_len"],
        missing_rate=data_cfg.get("missing_rate", 0.0),
        missing_pattern=data_cfg.get("missing_pattern", "random"),
        scaler=scaler,
        fit_scaler=False,
        seed=cfg["experiment"]["seed"] + 10,
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    trainer = BatteryCDETrainer(
        model=model, optimizer=optimizer, device=device, output_dir=output_dir
    )
    metrics = trainer.evaluate(loader)
    print(f"{args.split} metrics: {metrics}")

    with open(os.path.join(output_dir, f"{args.split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
