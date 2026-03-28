"""
XJTU电池数据集加载器
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class XJTUDatasetLoader:
    """
    XJTU电池数据集加载器
    
    参数:
        data_dir: 数据目录路径
        batch: 批次名称，如 '3C', '2C', 'R2.5', 'R3', 'RW', 'Sim_satellite'
    """
    
    def __init__(self, data_dir: str, batch: str):
        self.data_dir = Path(data_dir)
        self.batch = batch
        self.feature_cols = None
        self.target_col = None
        self.scaler = StandardScaler()
    
    def load_battery(self, battery_id: int) -> pd.DataFrame:
        """
        加载单个电池数据
        
        参数:
            battery_id: 电池编号
        
        返回:
            电池数据DataFrame
        """
        filename = f"{self.batch}_battery-{battery_id}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"找不到电池数据文件: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # 处理无穷大值：替换为NaN然后前向/后向填充
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
        # 计算SOH（容量保持率）
        # 使用小写的 'capacity' 匹配数据文件
        capacity_col = 'capacity' if 'capacity' in df.columns else 'Capacity'
        initial_capacity = df[capacity_col].iloc[0]
        df['SOH'] = df[capacity_col] / initial_capacity * 100
        
        return df
    
    def load_all_batteries(self) -> Dict[int, pd.DataFrame]:
        """
        加载所有电池数据
        
        返回:
            电池ID到DataFrame的映射字典
        """
        batteries = {}
        pattern = f"{self.batch}_battery-*.csv"
        
        for filepath in self.data_dir.glob(pattern):
            # 提取电池ID
            battery_id = int(filepath.stem.split('-')[-1])
            batteries[battery_id] = self.load_battery(battery_id)
        
        return batteries
    
    def prepare_data(
        self,
        feature_cols: List[str],
        target_col: str = 'Capacity',
        test_size: float = 0.25,
        val_size: float = 0.25,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        准备训练和测试数据
        
        参数:
            feature_cols: 特征列名列表
            target_col: 目标列名
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子
        
        返回:
            包含X_train, X_val, X_test, y_train, y_val, y_test的字典
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # 加载所有电池
        batteries = self.load_all_batteries()
        
        if len(batteries) == 0:
            raise ValueError(f"未找到{self.batch}批次的电池数据")
        
        # 按电池划分训练/验证/测试集
        battery_ids = list(batteries.keys())
        
        # 首先划分出测试集
        train_val_ids, test_ids = train_test_split(
            battery_ids,
            test_size=test_size,
            random_state=random_seed
        )
        
        # 然后划分训练集和验证集
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size / (1 - test_size),
            random_state=random_seed
        )
        
        # 合并数据
        train_data = self._merge_battery_data(batteries, train_ids)
        val_data = self._merge_battery_data(batteries, val_ids)
        test_data = self._merge_battery_data(batteries, test_ids)
        
        # 提取特征和目标
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_val = val_data[feature_cols].values
        y_val = val_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        # 标准化
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids,
            'scaler': self.scaler
        }
    
    def _merge_battery_data(
        self,
        batteries: Dict[int, pd.DataFrame],
        battery_ids: List[int]
    ) -> pd.DataFrame:
        """合并指定电池的数据"""
        data_list = [batteries[bid] for bid in battery_ids if bid in batteries]
        return pd.concat(data_list, ignore_index=True)
    
    def get_feature_info(self) -> Dict[str, int]:
        """获取特征信息"""
        return {
            'n_features': len(self.feature_cols) if self.feature_cols else 0,
            'feature_names': self.feature_cols,
            'n_batteries': len(self.load_all_batteries())
        }


def prepare_datasets_for_training(
    data: Dict[str, np.ndarray],
    use_mim: bool = False,
    model_type: str = 'mlp',
    training_config: Optional[Dict] = None
):
    """
    准备训练用的Dataset（支持多MR验证）
    """
    from .dataset import (
        BatteryDataset, SequenceDataset,
        MIMDataset, SequenceMIMDataset
    )
    
    if training_config is None:
        training_config = {}
    
    is_sequence = model_type in ['lstm', 'gru', 'cnn1d', 'cnn']
    seq_len = training_config.get('seq_len', 5)
    seed = training_config.get('seed', 42)
    
    training_missing_rates = training_config.get(
        'training_missing_rates',
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    
    val_mr_subset = training_config.get('val_mr_subset', [-1])
    if val_mr_subset == [-1]:
        validation_missing_rates = training_missing_rates
    else:
        validation_missing_rates = val_mr_subset
    
    imputation_method = training_config.get('imputation_method', 'zero')
    
    if use_mim:
        if is_sequence:
            train_dataset = SequenceMIMDataset(
                data['X_train'], data['y_train'],
                seq_len=seq_len,
                missing_rates=training_missing_rates,
                use_mim=True,
                base_seed=seed,
                imputation_method=imputation_method,
                fit_imputer=True
            )
            val_datasets = {}
            for mr in validation_missing_rates:
                val_datasets[f"mr_{mr:.2f}"] = SequenceDataset(
                    data['X_val'], data['y_val'],
                    seq_len=seq_len,
                    missing_rate=mr,
                    use_mim=True,
                    seed=seed
                )
        else:
            train_dataset = MIMDataset(
                data['X_train'], data['y_train'],
                missing_rates=training_missing_rates,
                use_mim=True,
                base_seed=seed,
                imputation_method=imputation_method,
                fit_imputer=True
            )
            val_datasets = {}
            for mr in validation_missing_rates:
                val_datasets[f"mr_{mr:.2f}"] = BatteryDataset(
                    data['X_val'], data['y_val'],
                    missing_rate=mr,
                    use_mim=True,
                    seed=seed
                )
    else:
        if is_sequence:
            train_dataset = SequenceDataset(
                data['X_train'], data['y_train'],
                seq_len=seq_len,
                missing_rate=0.0,
                use_mim=False,
                seed=seed
            )
            val_datasets = {
                "full": SequenceDataset(
                    data['X_val'], data['y_val'],
                    seq_len=seq_len,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=seed
                )
            }
        else:
            train_dataset = BatteryDataset(
                data['X_train'], data['y_train'],
                missing_rate=0.0,
                use_mim=False,
                seed=seed
            )
            val_datasets = {
                "full": BatteryDataset(
                    data['X_val'], data['y_val'],
                    missing_rate=0.0,
                    use_mim=False,
                    seed=seed
                )
            }
    
    return train_dataset, val_datasets
