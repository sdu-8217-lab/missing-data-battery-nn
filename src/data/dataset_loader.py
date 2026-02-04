"""XJTU数据集加载器"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler


class XJTUDatasetLoader:
    """XJTU电池数据集加载器"""
    
    def __init__(self, data_dir: str, batch: str):
        """
        初始化加载器
        
        Args:
            data_dir: 数据目录路径
            batch: 批次名称 (2C, 3C, R2.5, R3, RW, Sim_satellite)
        """
        self.data_dir = Path(data_dir)
        self.batch = batch
        
    def get_battery_files(self) -> List[Path]:
        """获取该批次下所有电池文件"""
        pattern = f"{self.batch}_battery-*.csv"
        files = sorted(self.data_dir.glob(pattern))
        
        if len(files) == 0:
            raise FileNotFoundError(
                f"在目录 {self.data_dir} 中未找到匹配 '{pattern}' 的电池数据文件。"
                f"请确认数据文件存在且命名格式正确（如：{self.batch}_battery-1.csv）"
            )
        
        return files
    
    def get_battery_id_range(self) -> Dict[str, any]:
        """
        获取该批次电池ID的范围信息
        
        Returns:
            Dict: {
                'min_id': 最小电池ID,
                'max_id': 最大电池ID,
                'count': 电池总数,
                'ids': 所有电池ID列表
            }
        """
        files = self.get_battery_files()
        ids = []
        
        for file_path in files:
            try:
                # 从文件名提取电池ID (如: 2C_battery-5.csv -> 5)
                battery_id = int(file_path.stem.split('-')[-1])
                ids.append(battery_id)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"无法从文件名 '{file_path.name}' 提取电池ID。"
                    f"期望格式: '{{batch}}_battery-{{id}}.csv'"
                ) from e
        
        return {
            'min_id': min(ids),
            'max_id': max(ids),
            'count': len(ids),
            'ids': sorted(ids)
        }
    
    def load_battery(self, file_path: Path, feature_cols: List[str], 
                     target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个电池数据
        
        Args:
            file_path: 电池CSV文件路径
            feature_cols: 特征列名列表
            target_col: 目标列名
            
        Returns:
            X: 特征数组 [n_samples, n_features]
            y: SOH标签 [n_samples]
        """
        df = pd.read_csv(file_path)
        
        # 数据清洗：移除inf和NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols + [target_col])
        
        if len(df) == 0:
            raise ValueError(f"{file_path.name}: 清洗后无数据")
        
        # 异常值检测（3-sigma）
        for col in feature_cols + [target_col]:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                mask = (df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)
                df = df[mask]
        
        if len(df) == 0:
            raise ValueError(f"{file_path.name}: 异常值过滤后无数据")
        
        # 计算SOH
        initial_capacity = df[target_col].iloc[0]
        y = df[target_col].values / initial_capacity
        
        # 提取特征
        X = df[feature_cols].values
        
        return X, y
    
    def load_all_batteries(self, feature_cols: List[str], 
                          target_col: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        加载所有电池数据
        
        Returns:
            Dict: {battery_id: (X, y)}
        """
        files = self.get_battery_files()
        batteries = {}
        
        for file_path in files:
            # 从文件名提取电池ID
            battery_id = int(file_path.stem.split('-')[-1])
            X, y = self.load_battery(file_path, feature_cols, target_col)
            batteries[battery_id] = (X, y)
            
        return batteries
    
    def split_batteries(self, battery_ids: List[int], 
                       test_size: float = 0.25, 
                       val_size: float = 0.25,
                       random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        按电池划分训练/验证/测试集
        
        Args:
            battery_ids: 电池ID列表
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子
            
        Returns:
            train_ids, val_ids, test_ids
        """
        # 验证电池数量是否足够
        n_batteries = len(battery_ids)
        min_required = 4  # 至少4个电池才能进行合理划分
        
        if n_batteries < min_required:
            raise ValueError(
                f"电池数量({n_batteries})不足，无法进行训练/验证/测试集划分。"
                f"至少需要{min_required}个电池。"
                f"建议：收集更多电池数据或调整划分比例。"
            )
        
        # 计算划分数量，确保每个集合至少有1个电池
        n_test = max(1, int(n_batteries * test_size))
        n_val = max(1, int(n_batteries * val_size))
        n_train = n_batteries - n_test - n_val
        
        if n_train < 1:
            raise ValueError(
                f"划分后训练集为空。电池总数={n_batteries}, "
                f"测试集={n_test}, 验证集={n_val}。"
                f"请调整test_size和val_size参数。"
            )
        
        # 随机打乱并划分
        np.random.seed(random_seed)
        ids = np.array(sorted(battery_ids))
        np.random.shuffle(ids)
        
        test_ids = ids[:n_test].tolist()
        val_ids = ids[n_test:n_test+n_val].tolist()
        train_ids = ids[n_test+n_val:].tolist()
        
        return train_ids, val_ids, test_ids
    
    def prepare_data(self, feature_cols: List[str], target_col: str,
                    test_size: float = 0.25, val_size: float = 0.25,
                    random_seed: int = 42,
                    logger=None) -> Dict:
        """
        准备完整数据集
        
        Args:
            feature_cols: 特征列名列表
            target_col: 目标列名
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子
            logger: 日志记录器（可选）
        
        Returns:
            Dict包含：
                - X_train, y_train
                - X_val, y_val
                - X_test, y_test
                - scaler
                - battery_info
        """
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # 获取电池ID范围信息
        battery_range = self.get_battery_id_range()
        logger.info(f"批次 '{self.batch}' 检测到 {battery_range['count']} 个电池文件")
        logger.info(f"电池ID范围: {battery_range['min_id']} - {battery_range['max_id']}")
        logger.info(f"电池ID列表: {battery_range['ids']}")
        
        # 加载所有电池
        batteries = self.load_all_batteries(feature_cols, target_col)
        battery_ids = list(batteries.keys())
        
        logger.info(f"成功加载 {len(batteries)} 个电池数据")
        
        # 划分电池
        train_ids, val_ids, test_ids = self.split_batteries(
            battery_ids, test_size, val_size, random_seed
        )
        
        logger.info(f"数据集划分 (seed={random_seed}):")
        logger.info(f"  - 训练集: {len(train_ids)} 个电池 {train_ids}")
        logger.info(f"  - 验证集: {len(val_ids)} 个电池 {val_ids}")
        logger.info(f"  - 测试集: {len(test_ids)} 个电池 {test_ids}")
        
        # 合并数据
        def merge_batteries(ids):
            X_list = []
            y_list = []
            for bid in ids:
                if bid not in batteries:
                    raise ValueError(f"电池ID {bid} 不在数据中. 可用ID: {list(batteries.keys())}")
                X, y = batteries[bid]
                X_list.append(X)
                y_list.append(y)
            if len(X_list) == 0:
                raise ValueError("没有有效的电池数据")
            return np.vstack(X_list), np.concatenate(y_list)
        
        X_train, y_train = merge_batteries(train_ids)
        X_val, y_val = merge_batteries(val_ids)
        X_test, y_test = merge_batteries(test_ids)
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scaler': scaler,
            'battery_info': {
                'train': train_ids,
                'val': val_ids,
                'test': test_ids
            }
        }
