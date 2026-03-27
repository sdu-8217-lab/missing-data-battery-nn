"""
日志工具
使用Loguru进行日志记录
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logger(
    name: str = "experiment",
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB"
):
    """
    设置日志记录器
    
    参数:
        name: 日志名称
        log_file: 日志文件路径
        level: 日志级别
        rotation: 日志文件轮转大小
    
    返回:
        配置好的logger实例
    """
    # 移除默认处理器
    loguru_logger.remove()
    
    # 添加控制台处理器
    loguru_logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - <level>{message}</level>",
        filter=lambda record: record["extra"].get("name") == name
    )
    
    # 添加文件处理器
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            log_file,
            level=level,
            rotation=rotation,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
            filter=lambda record: record["extra"].get("name") == name
        )
    
    # 创建带名称的logger
    named_logger = loguru_logger.bind(name=name)
    
    return named_logger


# 默认logger
logger = setup_logger("default")
