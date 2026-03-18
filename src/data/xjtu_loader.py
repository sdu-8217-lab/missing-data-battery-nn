"""
XJTU数据加载器 - 简化版

用于新架构的简单数据加载
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class XJTUDataLoader:
    """
    XJTU电池数据加载器
    
    Args:
        batch_id: 批次ID，如 '2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'
        data_dir: 数据目录路径
    """
    
    def __init__(self, batch_id: str, data_dir: str = "data/XJTU data"):
        self.batch_id = batch_id
        self.data_dir = Path(data_dir)
        
    def load_data(self) -> pd.DataFrame:
        """
        加载指定批次的所有电池数据
        
        Returns:
            DataFrame包含所有电池的数据，带有battery_id列
        """
        # 查找批次文件
        pattern = f"{self.batch_id}_battery-*.csv"
        battery_files = sorted(self.data_dir.glob(pattern))
        
        if not battery_files:
            raise FileNotFoundError(f"No battery files found for batch {self.batch_id} "
                                   f"with pattern {pattern} in {self.data_dir}")
        
        # 加载所有电池
        dfs = []
        for file_path in battery_files:
            battery_id = file_path.stem
            df = pd.read_csv(file_path)
            df['battery_id'] = battery_id
            dfs.append(df)
        
        # 合并
        combined_df = pd.concat(dfs, ignore_index=True)
        
        return combined_df
    
    def get_battery_ids(self) -> List[str]:
        """获取批次中所有电池ID"""
        pattern = f"{self.batch_id}_battery-*.csv"
        battery_files = sorted(self.data_dir.glob(pattern))
        return [f.stem for f in battery_files]


def load_xjtu_batch(batch_id: str, data_dir: str = "data/XJTU data") -> pd.DataFrame:
    """
    加载XJTU批次的便捷函数
    
    Args:
        batch_id: 批次ID
        data_dir: 数据目录
        
    Returns:
        DataFrame
    """
    loader = XJTUDataLoader(batch_id=batch_id, data_dir=data_dir)
    return loader.load_data()
