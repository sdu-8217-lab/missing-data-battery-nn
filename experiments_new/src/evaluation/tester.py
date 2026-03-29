"""
模型测试器
支持完整的测试流程：3种缺失模式 × 20种MR × 4种插补 = 240组合
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, TensorDataset

from ..data.imputation import Imputer
from ..data.missing_patterns import generate_missing_mask
from .metrics import calculate_metrics


class ModelTester:
    """
    模型测试器
    
    执行META文档定义的测试流程：
    - 分界线以下：测试阶段遍历所有条件组合
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "auto",
        use_cache: bool = True,
        seq_len: int = 5,
        is_sequence_model: bool = False
    ):
        """
        参数:
            model: 训练好的模型
            device: 计算设备
            use_cache: 是否使用插补缓存
            seq_len: 序列长度（用于LSTM/CNN）
            is_sequence_model: 是否为序列模型
        """
        self.model = model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 插补缓存
        self.use_cache = use_cache
        self._cache = {}
        
        # 序列模型参数
        self.seq_len = seq_len
        self.is_sequence_model = is_sequence_model
    
    def test_all_conditions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_mim: bool = False,
        missing_modes: List[str] = ["MCAR"],
        missing_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        imputation_methods: List[str] = ["zero"],
        base_seed: int = 42,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        测试所有条件组合
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            use_mim: 模型是否使用MIM
            missing_modes: 缺失模式列表
            missing_rates: 缺失率列表
            imputation_methods: 插补方法列表
            base_seed: 基础随机种子
            batch_size: 批大小
        
        返回:
            测试结果DataFrame
        """
        results = []
        
        # 预拟合各种插补器（使用完整测试数据）
        imputers = {}
        for method in imputation_methods:
            if method != "zero":
                imputer = Imputer(method=method)
                imputer.fit(X_test)
                imputers[method] = imputer
        
        total_tests = len(missing_modes) * len(missing_rates) * len(imputation_methods)
        test_count = 0
        
        # 遍历所有组合
        for mode in missing_modes:
            for missing_rate in missing_rates:
                for imp_method in imputation_methods:
                    test_count += 1
                    
                    # 生成测试缺失（分界线以下 L7+L8）
                    seed = base_seed + hash(f"{mode}_{missing_rate}_{imp_method}") % 10000
                    mask = generate_missing_mask(X_test, missing_rate, mode, seed)
                    
                    # 应用插补（分界线以下 L9）
                    if imp_method == "zero":
                        X_imputed = np.where(mask == 0, X_test, 0.0)
                    else:
                        X_missing = X_test.copy()
                        X_missing[mask == 1] = np.nan
                        X_imputed = imputers[imp_method].transform(X_test, mask)
                    
                    # 准备模型输入
                    if use_mim:
                        # MIM：拼接插补后特征和掩码
                        missing_indicators = mask.astype(np.float32)
                        X_input = np.concatenate([X_imputed, missing_indicators], axis=1)
                    else:
                        # 非MIM：仅使用插补后特征
                        X_input = X_imputed
                    
                    # 如果是序列模型，构造序列数据
                    if self.is_sequence_model:
                        X_input = self._construct_sequences(X_input)
                        # 标签也需要相应调整（取序列最后一个时间步的标签）
                        y_test_seq = y_test[self.seq_len - 1:]
                    else:
                        y_test_seq = y_test
                    
                    # 预测
                    y_pred = self._predict(X_input, batch_size)
                    
                    # 确保标签长度与预测一致
                    y_test_seq = y_test_seq[:len(y_pred)]
                    
                    # 计算指标
                    metrics = calculate_metrics(y_test_seq, y_pred)
                    
                    # 记录结果
                    result = {
                        'missing_mode': mode,
                        'missing_rate': missing_rate,
                        'imputation_method': imp_method,
                        'use_mim': use_mim,
                        **metrics
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _construct_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        构造序列数据（用于LSTM/CNN）
        
        参数:
            X: 输入特征 [n_samples, n_features]
        
        返回:
            序列数据 [n_samples - seq_len + 1, seq_len, n_features]
        """
        n_samples, n_features = X.shape
        sequences = []
        
        for i in range(n_samples - self.seq_len + 1):
            seq = X[i:i + self.seq_len]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def _predict(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 输入特征
            batch_size: 批大小
        
        返回:
            预测结果
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(np.zeros(len(X)), dtype=torch.float32)  # 占位标签
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                # 展平输出确保为一维数组 [batch, 1] -> [batch]
                pred = outputs.cpu().numpy().flatten()
                predictions.append(pred)
        
        return np.concatenate(predictions)


def run_full_evaluation(
    model_path: str,
    data: Dict[str, np.ndarray],
    use_mim: bool,
    config: dict,
    device: str = "auto"
) -> pd.DataFrame:
    """
    运行完整评估
    
    参数:
        model_path: 模型文件路径
        data: 数据字典
        use_mim: 是否使用MIM
        config: 配置字典
        device: 计算设备
    
    返回:
        完整测试结果
    """
    # 加载模型
    model = torch.load(model_path, map_location=device)
    
    # 创建测试器
    tester = ModelTester(model, device=device)
    
    # 执行测试
    results = tester.test_all_conditions(
        X_test=data['X_test'],
        y_test=data['y_test'],
        use_mim=use_mim,
        missing_modes=config.get('missing_modes', ['MCAR']),
        missing_rates=config.get('missing_rates', [i * 0.05 for i in range(20)]),
        imputation_methods=config.get('imputation_methods', ['zero']),
        base_seed=config.get('seed', 42),
        batch_size=config.get('batch_size', 32)
    )
    
    return results
