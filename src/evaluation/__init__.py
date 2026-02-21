"""评估模块

提供模型评估指标和结果记录功能。
"""

from .metrics import compute_metrics
from .result_writer import append_result_row

__all__ = ["compute_metrics", "append_result_row"]
