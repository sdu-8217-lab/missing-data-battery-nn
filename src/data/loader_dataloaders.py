"""DataLoader 创建模块 - 支持滑动窗口序列."""
from typing import Dict, Tuple, List
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


# DataLoader 性能优化参数
# 根据CPU核心数自动调整（i9-14900KF 32核）
import multiprocessing
_cpu_count = multiprocessing.cpu_count()
DEFAULT_NUM_WORKERS = min(12, _cpu_count // 2)  # 使用最多12个workers
DEFAULT_PIN_MEMORY = True  # 加速CPU->GPU传输
DEFAULT_PERSISTENT_WORKERS = True  # 避免worker进程重复创建

print(f"DataLoader config: num_workers={DEFAULT_NUM_WORKERS}, pin_memory={DEFAULT_PIN_MEMORY}")


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
    """Baseline: 施加缺失 + 传统插补（支持 2D 和 3D 序列数据）"""
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    from ..missing_data.mnar import simulate_mnar
    from ..missing_data.imputation import (
        mean_imputation, median_imputation, knn_imputation, 
        zero_imputation, forward_fill_imputation, iterative_imputation
    )
    
    # 处理 3D 序列数据: [batch, seq_len, features] -> [batch*seq_len, features]
    original_shape = X.shape
    is_3d = len(original_shape) == 3
    
    if is_3d:
        batch, seq, feat = original_shape
        X_flat = X.reshape(-1, feat)
        # y 需要扩展以匹配
        y_expanded = y.unsqueeze(1).expand(-1, seq).reshape(-1)
    else:
        X_flat = X
        y_expanded = y
    
    # 施加缺失（在展平后的数据上）
    if mode == 'mar':
        _, missing_mask, _ = simulate_mar(X_flat, y_expanded, missing_rate, seed=seed)
    elif mode == 'mnar':
        _, missing_mask, _ = simulate_mnar(X_flat, y_expanded, missing_rate, seed=seed)
    else:  # mcar
        _, missing_mask, _ = simulate_mcar(X_flat, missing_rate, seed=seed)
    
    # 在展平数据上进行插补
    if impute_method == "mean":
        X_imputed = mean_imputation(X_flat, missing_mask)
    elif impute_method == "median":
        X_imputed = median_imputation(X_flat, missing_mask)
    elif impute_method == "knn":
        X_imputed = knn_imputation(X_flat, missing_mask)
    elif impute_method == "zero":
        X_imputed = zero_imputation(X_flat, missing_mask)
    elif impute_method == "forward_fill":
        X_imputed = forward_fill_imputation(X_flat, missing_mask)
    elif impute_method == "iterative":
        X_imputed = iterative_imputation(X_flat, missing_mask, random_state=seed)
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")
    
    # 如果需要，恢复形状
    if is_3d:
        X_imputed = X_imputed.reshape(batch, seq, feat)
    
    return X_imputed


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
    from ..missing_data.mnar import simulate_mnar
    
    all_inputs = []
    all_targets = []
    
    # 获取插补方法配置（从 _group_ 或 method）
    group_cfg = cfg.get('_group_', {})
    method_cfg = cfg.get('method', {})
    
    if isinstance(group_cfg, dict) and 'imputation' in group_cfg:
        impute_method = group_cfg.get('imputation', 'mean')
    elif isinstance(method_cfg, dict):
        impute_method = method_cfg.get('imputation', 'mean')
    else:
        impute_method = 'mean'
    
    # 获取缺失模式
    missing_mode = cfg.get('missing', {}).get('mode', 'mcar')
    
    for i, mr in enumerate(missing_rates):
        mr_seed = seed + i * 100
        
        # 根据缺失模式选择模拟方法
        if missing_mode == 'mar':
            _, _, mim_input = simulate_mar(X, y, mr, seed=mr_seed, impute_method=impute_method)
        elif missing_mode == 'mnar':
            _, _, mim_input = simulate_mnar(X, y, mr, seed=mr_seed, impute_method=impute_method)
        else:  # mcar
            _, _, mim_input = simulate_mcar(X, mr, seed=mr_seed, impute_method=impute_method)
        
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)


def create_dataloaders(
    data_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    mode: str = 'train',
    method: str = 'mim',
    missing_rate: float = 0.0,
    missing_rates: List[float] = None,
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
    
    def prepare_data(X, y, is_mim=False, train_mrs=None):
        """准备数据：可选滑动窗口 + MIM 处理
        
        Args:
            train_mrs: MIM训练用的缺失率列表（合并训练），如 [0.0, 0.1, ..., 0.9]
        """
        # 确定使用哪些缺失率
        if train_mrs is None:
            # 默认10档
            train_mrs = [i/10.0 for i in range(10)]
        
        if use_sequence:
            # 序列模型：先滑动窗口
            X_seq, y_seq = create_sliding_windows(X, y, seq_len)
            
            if is_mim:
                # 对每个时间步独立施加 MIM
                # 将 [batch, seq_len, features]  reshape 为 [batch*seq_len, features]
                batch, seq, feat = X_seq.shape
                X_flat = X_seq.reshape(-1, feat)
                y_flat = y_seq.unsqueeze(1).expand(-1, seq).reshape(-1)
                
                # 施加 MIM（使用传入的缺失率列表）
                X_mim, y_mim = create_mim_training_data(X_flat, y_flat, train_mrs, cfg)
                
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
                return create_mim_training_data(X, y, train_mrs, cfg)
            else:
                if method == 'mim':
                    mask = torch.zeros_like(X)
                    X = torch.cat([X, mask], dim=1)
                return X, y
    
    if mode == 'train':
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        # 对于MIM，使用传入的missing_rates（10档合并）
        # 对于Baseline，忽略missing_rates，只在完整数据上训练
        if method == 'mim':
            X_train, y_train = prepare_data(X_train, y_train, is_mim=True, train_mrs=missing_rates)
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
        
        # 使用性能优化的 DataLoader 参数
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': DEFAULT_NUM_WORKERS,
            'pin_memory': DEFAULT_PIN_MEMORY,
            'persistent_workers': DEFAULT_PERSISTENT_WORKERS and DEFAULT_NUM_WORKERS > 0,
        }
        
        return (
            DataLoader(TensorDataset(X_train, y_train), shuffle=True, **loader_kwargs),
            DataLoader(TensorDataset(X_val, y_val), shuffle=False, **loader_kwargs)
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
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': DEFAULT_NUM_WORKERS,
            'pin_memory': DEFAULT_PIN_MEMORY,
            'persistent_workers': DEFAULT_PERSISTENT_WORKERS and DEFAULT_NUM_WORKERS > 0,
        }
        return DataLoader(TensorDataset(X_val, y_val), shuffle=False, **loader_kwargs)
    
    else:  # eval
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # 评估：先滑动窗口，再施加缺失
        if use_sequence:
            X_test, y_test = create_sliding_windows(X_test, y_test, seq_len)
        
        if method == 'mim':
            from ..missing_data.mcar import simulate_mcar
            from ..missing_data.mar import simulate_mar
            
            # 处理 3D 序列数据
            original_shape = X_test.shape
            is_3d = len(original_shape) == 3
            
            if is_3d:
                batch, seq, feat = original_shape
                X_flat = X_test.reshape(-1, feat)
                y_expanded = y_test.unsqueeze(1).expand(-1, seq).reshape(-1)
            else:
                X_flat = X_test
                y_expanded = y_test
            
            # 施加缺失（注意：simulate_mcar/simulate_mar 不需要 y 参数）
            if cfg.missing.mode == 'mar':
                # MAR 需要 SOH 值，使用展平后的 y_expanded
                _, _, X_processed = simulate_mar(X_flat, y_expanded, missing_rate, seed=42)
            else:
                # MCAR 不需要 y
                _, _, X_processed = simulate_mcar(X_flat, missing_rate, seed=42)
            
            # 恢复形状
            if is_3d:
                X_test = X_processed.reshape(batch, seq, -1)
            else:
                X_test = X_processed
        else:
            X_test = apply_missing_and_impute(X_test, y_test, missing_rate, method, cfg.missing.mode, seed=42)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': DEFAULT_NUM_WORKERS,
            'pin_memory': DEFAULT_PIN_MEMORY,
            'persistent_workers': DEFAULT_PERSISTENT_WORKERS and DEFAULT_NUM_WORKERS > 0,
        }
        return None, None, DataLoader(TensorDataset(X_test, y_test), shuffle=False, **loader_kwargs)
