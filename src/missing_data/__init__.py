"""
缺失数据模拟模块

提供 MCAR (Missing Completely At Random) 和 MAR (Missing At Random) 
两种缺失模式的模拟实现。
"""

from .mcar import simulate_mcar
from .mar import simulate_mar

__all__ = ["simulate_mcar", "simulate_mar"]
