#!/usr/bin/env python3
"""
并行实验启动脚本 - 简洁版
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

if __name__ == "__main__":
    # 直接转发到简化版scheduler
    from parallel_scheduler import main
    sys.exit(main())
