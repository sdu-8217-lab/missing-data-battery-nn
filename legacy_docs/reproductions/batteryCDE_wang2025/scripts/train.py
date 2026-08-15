"""BatteryCDE 训练脚本（支持 BatteryCDE / LSTM / Transformer / Neural CDE）

用法:
    python scripts/train.py --config configs/nasa.yaml --model_type batteryCDE
    python scripts/train.py --config configs/nasa.yaml --model_type lstm
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
from reproductions.batteryCDE_wang2025.src.models.baselines import (  # noqa: E402
    LSTMBaseline,
    TransformerBaseline,
    NeuralCDEBaseline,
)
from reproductions.batteryCDE_wang2025.src.data.battery_cde_dataset import (  # noqa: E402
    BatteryCDEDataset,
    load_nasa_trajectories,
)
from reproductions.batteryCDE_wang2025.src.trainers.battery_cde_trainer import BatteryCDETrainer  # noqa: E402


MODEL_REGISTRY = {
    "batteryCDE": BatteryCDEModel,
    "lstm": LSTMBaseline,
    "transformer": TransformerBaseline,
    "neuralCDE": NeuralCDEBaseline,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train BatteryCDE or baselines")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_type", type=str, default="batteryCDE",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model type to train")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg, data_cfg, seed: int):
    """根据配置构造 DataLoader。"""
    trajectory_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "nasa_trajectories")
    )

    # 容量归一化：使用训练集最大容量作为 scale（得到 SOH-like 目标）
    train_cycles = load_nasa_trajectories(
        trajectory_dir, data_cfg["train_batteries"], capacity_scale=None
    )
    all_caps = [cyc["capacity"] for batt in train_cycles for cyc in batt]
    capacity_scale = float(max(all_caps)) if all_caps else 1.0
    print(f"Capacity scale (train max capacity): {capacity_scale:.4f}")

    # 重新加载并归一化
    train_cycles = load_nasa_trajectories(
        trajectory_dir, data_cfg["train_batteries"], capacity_scale=capacity_scale
    )
    val_cycles = load_nasa_trajectories(
        trajectory_dir, data_cfg["val_batteries"], capacity_scale=capacity_scale
    )
    test_cycles = load_nasa_trajectories(
        trajectory_dir, data_cfg["test_batteries"], capacity_scale=capacity_scale
    )

    train_dataset = BatteryCDEDataset(
        train_cycles,
        horizon=data_cfg["horizon"],
        history_len=data_cfg["history_len"],
        missing_rate=data_cfg.get("missing_rate", 0.0),
        missing_pattern=data_cfg.get("missing_pattern", "random"),
        fit_scaler=True,
        seed=seed,
    )
    scaler = train_dataset.scaler

    val_dataset = BatteryCDEDataset(
        val_cycles,
        horizon=data_cfg["horizon"],
        history_len=data_cfg["history_len"],
        missing_rate=data_cfg.get("missing_rate", 0.0),
        missing_pattern=data_cfg.get("missing_pattern", "random"),
        scaler=scaler,
        fit_scaler=False,
        seed=seed + 1,
    )
    test_dataset = BatteryCDEDataset(
        test_cycles,
        horizon=data_cfg["horizon"],
        history_len=data_cfg["history_len"],
        missing_rate=data_cfg.get("missing_rate", 0.0),
        missing_pattern=data_cfg.get("missing_pattern", "random"),
        scaler=scaler,
        fit_scaler=False,
        seed=seed + 2,
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, capacity_scale


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    model_type = args.model_type
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        base_dir = cfg["experiment"]["output_dir"]
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", base_dir, model_type)
        )
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    cfg["model_type"] = model_type
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    device_str = cfg["experiment"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, scaler, capacity_scale = build_dataloaders(
        cfg, cfg["data"], seed
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model_cfg = cfg["model"]
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        method=model_cfg.get("method", "rk4"),
        options=model_cfg.get("options"),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    trainer = BatteryCDETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        patience=cfg["training"]["patience"],
        output_dir=output_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
    )

    # 测试集评估
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest metrics: {test_metrics}")

    # 将 numpy 标量转为 Python 原生类型，便于 JSON 序列化
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    metrics_serializable = {k: float(v) for k, v in test_metrics.items()}

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2)
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2)

    # 保存 scaler 与 capacity_scale
    np.savez(
        os.path.join(output_dir, "scaler.npz"),
        mean=scaler.mean_,
        scale=scaler.scale_,
        capacity_scale=capacity_scale,
    )


if __name__ == "__main__":
    main()
