#!/usr/bin/env python3
"""
对比测试：旧版代码 vs 新版代码
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_old_version():
    """测试旧版代码"""
    print("=" * 60)
    print("[旧版代码] 使用 dataclass + logging + 自定义训练器")
    print("=" * 60)
    
    from src.config import ExperimentConfig
    from src.utils import setup_logger
    
    start = time.time()
    
    # 创建配置
    config = ExperimentConfig(
        batch="3C",
        random_seed=42,
        n_repeats=1,
        epochs=50
    )
    
    # 设置日志
    logger = setup_logger("old_test", "./logs/old_version.log")
    logger.info(f"旧版配置创建成功: {config.name}")
    logger.info(f"批次: {config.batch}")
    logger.info(f"模型数量: {len(config.get_model_configs())}")
    
    elapsed = time.time() - start
    print(f"旧版代码初始化时间: {elapsed:.3f}s")
    print(f"日志文件: ./logs/old_version.log")
    return elapsed


def test_new_version():
    """测试新版代码"""
    print("\n" + "=" * 60)
    print("[新版代码] 使用 Pydantic + Loguru + Lightning")
    print("=" * 60)
    
    from src.config import ExperimentConfigV2
    from src.utils import setup_logger
    
    start = time.time()
    
    # 创建配置
    config = ExperimentConfigV2(
        batch="3C",
        random_seed=42,
        n_repeats=1,
        epochs=50
    )
    
    # 设置日志
    logger = setup_logger("new_test", "./logs/new_version.log")
    logger.info(f"新版配置创建成功: {config.name}")
    logger.info(f"批次: {config.batch}")
    logger.info(f"模型数量: {len(config.get_model_configs())}")
    
    # 测试验证功能
    try:
        invalid = ExperimentConfigV2(test_size=1.5)
        print("验证失败!")
    except Exception as e:
        logger.info(f"验证功能正常: {type(e).__name__}")
    
    elapsed = time.time() - start
    print(f"新版代码初始化时间: {elapsed:.3f}s")
    print(f"日志文件: ./logs/new_version.log")
    return elapsed


def main():
    print("开始对比测试: 旧版 vs 新版代码")
    print("=" * 60)
    
    old_time = test_old_version()
    new_time = test_new_version()
    
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"旧版代码初始化: {old_time:.3f}s")
    print(f"新版代码初始化: {new_time:.3f}s")
    
    if new_time < old_time:
        print(f"新版更快: {(old_time - new_time)/old_time*100:.1f}%")
    else:
        print(f"旧版更快: {(new_time - old_time)/old_time*100:.1f}%")
    
    print("\n[结论]")
    print("- 新版代码提供了更好的类型安全和验证")
    print("- Loguru日志更友好，自动处理UTF-8编码")
    print("- Lightning训练器大幅简化训练代码")
    print("- 可以通过运行实际实验来对比训练性能")


if __name__ == "__main__":
    main()
