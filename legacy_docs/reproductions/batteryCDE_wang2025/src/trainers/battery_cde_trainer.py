"""BatteryCDE 训练器

实现训练、早停、验证、测试流程。尽量复用项目级工具，但保持自包含。
"""
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 复用项目级工具
import sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, _PROJECT_ROOT)
from src.evaluators.metrics import calculate_all_metrics  # noqa: E402


class BatteryCDETrainer:
    """BatteryCDE 训练器。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        patience: int = 15,
        output_dir: str = "./results",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.output_dir = output_dir
        self.criterion = nn.MSELoss()
        os.makedirs(output_dir, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds, targets = [], []
        for x, y in loader:
            x = x.to(self.device)
            pred = self.model(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        # 过滤 NaN/Inf 预测
        valid_mask = np.isfinite(preds)
        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0:
            print(f"Warning: {n_invalid}/{len(preds)} predictions are non-finite, filtering them out.")
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        if len(preds) == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

        metrics = calculate_all_metrics(targets, preds)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_mae": []}
        best_mae = float("inf")
        patience_counter = 0
        best_path = os.path.join(self.output_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mae = val_metrics["mae"]
            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_mae)

            epoch_time = time.time() - start
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.6f} | val_mae={val_mae:.6f} | "
                f"time={epoch_time:.2f}s"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                print(f"  -> Best model saved (val_mae={best_mae:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return history
