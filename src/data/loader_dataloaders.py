"""DataLoader 创建模块 - 支持滑动窗口序列."""
from typing import Dict, Tuple, List
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


def create_sliding_windows(
    X: torch.Tensor,
    y: torch.Tensor,
    seq_len: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用滑动窗口创建序列数据
    
    Args:
        X: 特征矩阵 [N, D]
        y: 目标值 [N]
        seq_len: 序列长度（窗口大小）
        
    Returns:
        X_seq: 序列特征 [N-seq_len+1, seq_len, D]
        y_seq: 目标值 [N-seq_len+1] (使用窗口最后一个时间步的目标)
        
    说明:
        - 输入数据 X 应该已经按时间顺序排列
        - 对于电池数据，假设行顺序就是循环顺序
        - 前 seq_len-1 个样本无法构成完整窗口，被丢弃
    """
    N, D = X.shape
    
    if N < seq_len:
        raise ValueError(f"Not enough samples ({N}) for sequence length ({seq_len})")
    
    # 创建滑动窗口
    X_seq = []
    y_seq = []
    
    for i in range(N - seq_len + 1):
        # 取从 i 到 i+seq_len 的序列
        window = X[i:i+seq_len]  # [seq_len, D]
        # 目标值使用窗口最后一个时间步
        target = y[i+seq_len-1]
        
        X_seq.append(window)
        y_seq.append(target)
    
    return torch.stack(X_seq), torch.tensor(y_seq)


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
    """
    MIM 训练：混合多种缺失率
    
    注意: 这里对每个样本独立施加缺失，不保持序列结构
    对于序列模型，应该在滑动窗口之后、序列构造之前施加缺失
    """
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
    """
    创建 DataLoader，支持滑动窗口序列
    
    注意: 当前实现中，MIM 训练和滑动窗口有冲突：
    - MIM 将每个样本复制 10 份（不同缺失率）
    - 滑动窗口需要时间连续的序列
    - 目前的折衷：先创建滑动窗口，再对每个窗口施加 MIM
    """
    seq_len = cfg.data.get('seq_len', 5)
    batch_size = cfg.training.batch_size
    model_type = cfg.model.type
    use_sequence = model_type in ['lstm', 'gru', 'cnn1d']
    
    def prepare_data(X, y, is_mim=False):
        """准备数据：可选滑动窗口 + MIM 处理"""
        
        if use_sequence:
            # 序列模型：先滑动窗口
            X_seq, y_seq = create_sliding_windows(X, y, seq_len)
            
            if is_mim:
                # 对每个时间步独立施加 MIM
                # 将 [batch, seq_len, features]  reshape 为 [batch*seq_len, features]
                batch, seq, feat = X_seq.shape
                X_flat = X_seq.reshape(-1, feat)
                y_flat = y_seq.unsqueeze(1).expand(-1, seq).reshape(-1)
                
                # 施加 MIM（这会扩展 10 倍）
                mim_rates = cfg.missing.get('mim_train_rates', [i/10.0 for i in range(10)])
                X_mim, y_mim = create_mim_training_data(X_flat, y_flat, mim_rates, cfg)
                
                # 还原为序列形状 [batch*10, seq_len, features*2]
                new_batch = X_mim.shape[0] // seq
                X_mim = X_mim.reshape(new_batch, seq, -1)
                # 取每个序列最后一个时间步的目标
                y_mim = y_mim[seq-1::seq]
                
                return X_mim, y_mim
            else:
                # Baseline：添加零掩码（16 -> 32）
                if method == 'mim':
                    mask = torch.zeros_like(X_seq)
                    X_seq = torch.cat([X_seq, mask], dim=-1)
                return X_seq, y_seq
        else:
            # MLP：不使用序列
            if is_mim:
                mim_rates = cfg.missing.get('mim_train_rates', [i/10.0 for i in range(10)])
                return create_mim_training_data(X, y, mim_rates, cfg)
            else:
                if method == 'mim':
                    mask = torch.zeros_like(X)
                    X = torch.cat([X, mask], dim=1)
                return X, y
    
    if mode == 'train':
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        if method == 'mim':
            X_train, y_train = prepare_data(X_train, y_train, is_mim=True)
        else:
            X_train, y_train = prepare_data(X_train, y_train, is_mim=False)
        
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        if method == 'mim':
            mask_val = torch.zeros_like(X_val)
            if use_sequence:
                X_val, y_val = create_sliding_windows(X_val, y_val, seq_len)
                mask_val = torch.zeros_like(X_val)
            X_val = torch.cat([X_val, mask_val], dim=-1)
        elif use_sequence:
            X_val, y_val = create_sliding_windows(X_val, y_val, seq_len)
        
        return (
            DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        )
    
    elif mode == 'val':
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        if method == 'mim':
            mask_val = torch.zeros_like(X_val)
            if use_sequence:
                X_val, y_val = create_sliding_windows(X_val, y_val, seq_len)
                mask_val = torch.zeros_like(X_val)
            X_val = torch.cat([X_val, mask_val], dim=-1)
        elif use_sequence:
            X_val, y_val = create_sliding_windows(X_val, y_val, seq_len)
        
        return DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    else:  # eval
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # 评估：先滑动窗口，再施加缺失
        if use_sequence:
            X_test, y_test = create_sliding_windows(X_test, y_test, seq_len)
        
        if method == 'mim':
            from ..missing_data.mcar import simulate_mcar
            from ..missing_data.mar import simulate_mar
            
            if cfg.missing.mode == 'mar':
                _, _, X_test = simulate_mar(X_test, y_test, missing_rate, seed=42)
            else:
                _, _, X_test = simulate_mcar(X_test, missing_rate, seed=42)
        else:
            X_test = apply_missing_and_impute(X_test, y_test, missing_rate, method, cfg.missing.mode, seed=42)
        
        return None, None, DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
