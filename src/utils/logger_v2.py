"""日志配置模块 - 使用Loguru"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(name: str = None, log_file: str = None, level: str = "INFO"):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称（在loguru中不直接使用，保留参数用于兼容性）
        log_file: 日志文件路径
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        loguru.Logger: 配置好的日志记录器
    """
    # 移除默认的stderr处理器
    logger.remove()
    
    # 检测是否在支持Unicode的终端
    import os
    force_ascii = os.environ.get('PYTHONIOENCODING', '').lower() == 'ascii'
    
    # 添加stdout处理器，使用友好的格式
    # 如果控制台不支持Unicode，使用ASCII-safe格式
    if force_ascii:
        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}"
    else:
        fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " \
              "<level>{level: <8}</level> | " \
              "<cyan>{name}</cyan>: " \
              "<level>{message}</level>"
    
    try:
        logger.add(
            sys.stdout,
            level=level,
            format=fmt,
            filter=lambda record: record["level"].no >= logger.level(level).no,
            enqueue=True,
            colorize=not force_ascii
        )
    except Exception:
        # 如果stdout有问题，只使用文件日志
        pass
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            encoding="utf-8",
            rotation="100 MB",  # 自动轮转
            retention="30 days",  # 保留30天
            compression="gz",  # 压缩旧日志
            enqueue=True,
        )
    
    # 如果指定了name，添加绑定
    if name:
        return logger.bind(name=name)
    
    return logger


# 保持向后兼容的快捷函数
def get_logger(name: str = None):
    """获取已配置的logger实例"""
    if name:
        return logger.bind(name=name)
    return logger
