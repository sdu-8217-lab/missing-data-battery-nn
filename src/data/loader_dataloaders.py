"""DataLoader 创建模块."""
from typing import Dict, Tuple, List
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def apply_missing_and_impute(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rate: float,
    impute_method: str,
    mode: str,
    seed: int
) -> torch.Tensor:
    """Baseline: 施加缺失 + 传统插补"""
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    from ..missing_data.imputation import mean_imputation, median_imputation, knn_imputation, zero_imputation
    
    if mode == 'mar':
        _, missing_mask, _ = simulate_mar(X, y, missing_rate, seed=seed)
    else:
        _, missing_mask, _ = simulate_mcar(X, missing_rate, seed=seed)
    
    if impute_method == "mean":
        return mean_imputation(X, missing_mask)
    elif impute_method == "median":
        return median_imputation(X, missing_mask)
    elif impute_method == "knn":
        return knn_imputation(X, missing_mask)
    elif impute_method == "zero":
        return zero_imputation(X, missing_mask)
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")


def create_mim_training_data(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rates: List[float],
    cfg: DictConfig,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MIM 训练：混合多种缺失率"""
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(missing_rates):
        mr_seed = seed + i * 100
        
        if cfg.missing.mode == 'mar':
            _, _, mim_input = simulate_mar(X, y, mr, seed=mr_seed)
        else:
            _, _, mim_input = simulate_mcar(X, mr, seed=mr_seed)
        
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)


def create_dataloaders(
    data_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    mode: str = 'train',
    method: str = 'mim',
    missing_rate: float = 0.0,
) -> Tuple:
    """创建 DataLoader"""
    seq_len = cfg.data.get('seq_len', 5)
    batch_size = cfg.training.batch_size
    model_type = cfg.model.type
    
    def to_sequence(X):
        if X.dim() == 2 and model_type in ['lstm', 'gru', 'cnn1d']:
            X = X.unsqueeze(1).repeat(1, seq_len, 1)
        return X
    
    if mode == 'train':
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        if method == 'mim':
            mim_rates = cfg.missing.get('mim_train_rates', [i/10.0 for i in range(10)])
            X_train, y_train = create_mim_training_data(X_train, y_train, mim_rates, cfg)
        
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        if method == 'mim':
            mask_val = torch.zeros_like(X_val)
            X_val = torch.cat([X_val, mask_val], dim=1)
        
        X_train = to_sequence(X_train)
        X_val = to_sequence(X_val)
        
        return (
            DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        )
    
    elif mode == 'val':
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        if method == 'mim':
            mask_val = torch.zeros_like(X_val)
            X_val = torch.cat([X_val, mask_val], dim=1)
        
        X_val = to_sequence(X_val)
        return DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    else:  # eval
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        if method == 'mim':
            from ..missing_data.mcar import simulate_mcar
            from ..missing_data.mar import simulate_mar
            
            if cfg.missing.mode == 'mar':
                _, _, X_test = simulate_mar(X_test, y_test, missing_rate, seed=42)
            else:
                _, _, X_test = simulate_mcar(X_test, missing_rate, seed=42)
        else:
            X_test = apply_missing_and_impute(X_test, y_test, missing_rate, method, cfg.missing.mode, seed=42)
        
        X_test = to_sequence(X_test)
        return None, None, DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
