"""结果写入模块

提供将实验结果保存到 CSV 文件的功能。
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

# 配置日志记录器
logger = logging.getLogger(__name__)


def append_result_row(
    metrics: Dict[str, Any], 
    cfg: DictConfig,
    extra_fields: Optional[Dict[str, Any]] = None
) -> Path:
    """
    将实验结果追加到 CSV 文件
    
    如果文件不存在，自动创建并写入表头。支持自动创建目录。
    
    Args:
        metrics: 实验指标字典，包含如 mae, rmse, r2 等键值
        cfg: OmegaConf 配置对象，需要包含 experiment.output.result_csv 路径
        extra_fields: 可选的额外字段，将与 metrics 合并
        
    Returns:
        结果文件的 Path 对象
        
    Raises:
        ValueError: 当配置中缺少必要的输出路径时
        OSError: 当文件写入失败时
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({
        ...     "experiment": {"output": {"result_csv": "./results/exp.csv"}}
        ... })
        >>> metrics = {"mae": 0.01, "rmse": 0.02, "r2": 0.95}
        >>> path = append_result_row(metrics, cfg)
        >>> print(f"Results saved to: {path}")
    """
    try:
        # 获取结果文件路径
        result_csv = _get_result_path(cfg)
        
        # 确保目录存在
        result_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # 合并额外字段
        row_data = dict(metrics)
        if extra_fields:
            row_data.update(extra_fields)
        
        # 添加时间戳
        if "timestamp" not in row_data:
            row_data["timestamp"] = datetime.now().isoformat()
        
        # 确保所有值都是可序列化的
        row_data = _sanitize_dict(row_data)
        
        # 确定写入模式
        file_exists = result_csv.exists() and result_csv.stat().st_size > 0
        mode = 'a' if file_exists else 'w'
        
        # 写入文件
        with open(result_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            if not file_exists:
                writer.writeheader()
                logger.debug(f"Created new results file with headers: {result_csv}")
            
            writer.writerow(row_data)
        
        logger.info(f"Result saved to {result_csv}")
        return result_csv
        
    except Exception as e:
        logger.error(f"保存结果时发生错误: {e}")
        raise


def save_config_backup(cfg: DictConfig, output_dir: Optional[Path] = None) -> Path:
    """
    保存配置备份到 JSON 文件
    
    Args:
        cfg: OmegaConf 配置对象
        output_dir: 输出目录，默认使用 cfg.experiment.output.result_csv 的父目录
        
    Returns:
        配置文件的路径
    """
    try:
        if output_dir is None:
            result_csv = _get_result_path(cfg)
            output_dir = result_csv.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = output_dir / "config_backup.yaml"
        OmegaConf.save(cfg, config_path)
        
        logger.info(f"Config backup saved to {config_path}")
        return config_path
        
    except Exception as e:
        logger.error(f"保存配置备份时发生错误: {e}")
        raise


def load_results(result_csv: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    从 CSV 文件加载实验结果
    
    Args:
        result_csv: 结果文件路径
        
    Returns:
        结果字典列表
        
    Raises:
        FileNotFoundError: 当文件不存在时
    """
    result_csv = Path(result_csv)
    
    if not result_csv.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_csv}")
    
    try:
        with open(result_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]
        
        logger.info(f"Loaded {len(results)} rows from {result_csv}")
        return results
        
    except Exception as e:
        logger.error(f"加载结果时发生错误: {e}")
        raise


def _get_result_path(cfg: DictConfig) -> Path:
    """
    从配置中提取结果文件路径
    
    Args:
        cfg: OmegaConf 配置对象
        
    Returns:
        结果文件的 Path 对象
        
    Raises:
        ValueError: 当配置中缺少必要的输出路径时
    """
    # 尝试从多种可能的配置结构中获取路径
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'output'):
        if hasattr(cfg.experiment.output, 'result_csv'):
            return Path(cfg.experiment.output.result_csv)
    
    # 备选路径配置
    if hasattr(cfg, 'results_dir'):
        return Path(cfg.results_dir) / "results.csv"
    
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'results_dir'):
        return Path(cfg.experiment.results_dir) / "results.csv"
    
    # 默认路径
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'name'):
        exp_name = cfg.experiment.name
    else:
        exp_name = "experiment"
    
    default_path = Path("./results") / exp_name / "results.csv"
    logger.warning(
        f"配置中未找到结果输出路径，使用默认值: {default_path}"
    )
    return default_path


def _sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理字典中的值，确保可 CSV 序列化
    
    将复杂类型转换为字符串表示。
    
    Args:
        data: 原始字典
        
    Returns:
        清理后的字典
    """
    sanitized = {}
    
    for key, value in data.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            # 将列表和字典序列化为 JSON 字符串
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                sanitized[key] = str(value)
        elif hasattr(value, 'item'):  # NumPy 标量
            sanitized[key] = value.item()
        else:
            sanitized[key] = str(value)
    
    return sanitized
