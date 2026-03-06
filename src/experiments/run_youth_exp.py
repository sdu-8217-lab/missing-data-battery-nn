"""
Youth Experiment Runner - Non-Hydra Version
直接使用 Python 参数运行实验
"""
import sys
sys.path.insert(0, '.')

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import omegaconf
from src.data.loader import load_dataset
from src.utils.seed_manager import set_seed
from src.utils.logger import setup_logger

# Import models
from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.cnn1d import CNN1D

# Import missing data mechanisms
from src.missing_data.mcar import simulate_mcar
from src.missing_data.mar import simulate_mar


def create_model(model_type: str, input_dim: int, batch_id: str):
    """Create model based on type"""
    arch_cfg = omegaconf.OmegaConf.load("configs/model_architectures.yaml")
    arch_key = f"{model_type}_small_best_{batch_id.lower()}"
    
    if arch_key not in arch_cfg.small_architectures:
        raise ValueError(f"Architecture {arch_key} not found")
    
    arch = arch_cfg.small_architectures[arch_key]
    
    if model_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dims=arch.hidden_dims,
            dropout=arch.get("dropout", 0.1),
        )
    elif model_type == "lstm":
        return LSTM(
            input_dim=input_dim,
            hidden_dim=arch.hidden_dim,
            num_layers=arch.num_layers,
            dropout=arch.get("dropout", 0.1),
        )
    elif model_type == "cnn":
        return CNN1D(
            input_dim=input_dim,
            channels=arch.channels,
            kernel_sizes=arch.kernel_sizes,
            fc_hidden_dims=arch.get("fc_hidden_dims", [32]),
            dropout=arch.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_missing_data_mim(X: torch.Tensor, y: torch.Tensor, missing_rate: float, seed: int):
    """Create multi-MR training data for MIM method"""
    rate_list = [i/10.0 for i in range(10)]
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(rate_list):
        mr_seed = seed + i * 100
        X_imp, mask, mim_input = simulate_mcar(X, mr, mr_seed)
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    combined_input = torch.cat(all_inputs, dim=0)
    combined_target = torch.cat(all_targets, dim=0)
    return combined_input, combined_target


def create_missing_data_eval(X: torch.Tensor, y: torch.Tensor, mode: str, 
                             rate: float, seed: int, beta=2.0, gamma=0.05):
    """Create evaluation data with specified missing mechanism"""
    if mode == "mar":
        X_imp, mask, mim_input = simulate_mar(X, y, rate, None, beta, gamma, seed)
    elif mode == "mcar":
        X_imp, mask, mim_input = simulate_mcar(X, rate, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X_imp, mask, mim_input


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
    """Train model"""
    device = torch.device("cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                val_loss += criterion(outputs, batch_y).item() * len(batch_x)
        
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    return {
        "test_mae": mean_absolute_error(targets, preds),
        "test_rmse": np.sqrt(mean_squared_error(targets, preds)),
        "test_r2": r2_score(targets, preds),
    }


def run_single_config(method: str, model_type: str, batch_id: str, 
                      missing_rate: float, seeds: list, output_csv: str):
    """Run a single experiment configuration"""
    
    logger = setup_logger("experiment", log_file=Path("logs/experiment.log"))
    logger.info(f"Starting: method={method}, model={model_type}, batch={batch_id}, mr={missing_rate}")
    
    results = []
    
    for seed in seeds:
        set_seed(seed)
        
        # Load data
        data_cfg = omegaconf.OmegaConf.create({
            "data": {
                "dataset": "xjtu",
                "data_dir": "data/raw/XJTU",
                "batch": batch_id,
                "features": [
                    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
                    "current mean", "current std", "current kurtosis", "current skewness",
                    "CC Q", "CC charge time", "CV Q", "CV charge time",
                    "voltage slope", "current slope", "voltage entropy", "current entropy"
                ],
                "target": "capacity",
                "preprocessing": {
                    "remove_outliers": True,
                    "outlier_method": "3sigma",
                    "standardize": True,
                    "fill_na": True,
                    "fill_method": "mean",
                },
                "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            }
        })
        
        data = load_dataset(data_cfg)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Create missing data
        if method == "mim":
            input_dim = 32
            train_input, train_target = create_missing_data_mim(X_train, y_train, missing_rate, seed)
            _, _, val_input = create_missing_data_eval(X_val, y_val, "mar", missing_rate, seed + 999)
            _, _, test_input = create_missing_data_eval(X_test, y_test, "mar", missing_rate, seed + 1000)
        else:
            input_dim = 16
            X_train_imp, _, _ = create_missing_data_eval(X_train, y_train, "mar", missing_rate, seed)
            X_val_imp, _, _ = create_missing_data_eval(X_val, y_val, "mar", missing_rate, seed + 999)
            X_test_imp, _, _ = create_missing_data_eval(X_test, y_test, "mar", missing_rate, seed + 1000)
            train_input, train_target = X_train_imp, y_train
            val_input = X_val_imp
            test_input = X_test_imp
        
        # Create model
        model = create_model(model_type, input_dim, batch_id)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_input, train_target),
            batch_size=64, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_input, y_val),
            batch_size=64, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_input, y_test),
            batch_size=64, shuffle=False
        )
        
        # Train and evaluate
        model = train_model(model, train_loader, val_loader)
        metrics = evaluate_model(model, test_loader)
        
        result = {
            "dataset": "xjtu",
            "batch_id": batch_id,
            "missing_mode": "mar",
            "missing_rate": missing_rate,
            "method": method,
            "model": model_type,
            "model_size": "small",
            "seed": seed,
            **metrics,
        }
        results.append(result)
        
        logger.info(f"Seed {seed}: MAE={metrics['test_mae']:.4f}, R2={metrics['test_r2']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    if Path(output_csv).exists():
        df_existing = pd.read_csv(output_csv)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"Completed {len(seeds)} experiments")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Run Youth Experiment")
    parser.add_argument("--method", choices=["baseline", "mim"], required=True)
    parser.add_argument("--model", choices=["mlp", "lstm", "cnn"], required=True)
    parser.add_argument("--batch", choices=["2c", "3c"], required=True)
    parser.add_argument("--mr", type=float, required=True, help="Missing rate (0.1-0.9)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of seeds")
    parser.add_argument("--output", type=str, help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_csv = args.output
    elif args.method == "mim":
        output_csv = "results/csv/youth_mar_mim.csv"
    else:
        output_csv = "results/csv/youth_mar_baseline.csv"
    
    print(f"Running: method={args.method}, model={args.model}, batch={args.batch}, mr={args.mr}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_csv}")
    
    count = run_single_config(
        method=args.method,
        model_type=args.model,
        batch_id=args.batch,
        missing_rate=args.mr,
        seeds=args.seeds,
        output_csv=output_csv
    )
    
    print(f"\nCompleted! Saved {count} results to {output_csv}")


if __name__ == "__main__":
    main()
