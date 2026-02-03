"""工具模块"""
from .seed_manager import set_seeds, generate_seeds
from .logger import setup_logger

__all__ = ['set_seeds', 'generate_seeds', 'setup_logger']
