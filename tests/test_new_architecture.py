#!/usr/bin/env python3
"""
测试新架构
运行一个小规模测试（2个种子，2个模型）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.single_experiment import SingleExperimentRunner, SingleExperimentConfig
from src.experiments.batch_experiment import BatchExperimentRunner


def test_single_experiment():
    """测试单个小实验"""
    print("=" * 70)
    print("测试单个小实验")
    print("=" * 70)
    
    config = SingleExperimentConfig(
        batch_name='3C',
        seed=42,
        epochs=5,  # 少量轮数用于测试
        device='cpu'
    )
    
    runner = SingleExperimentRunner(config)
    results = runner.run()
    
    print(f"\n测试通过! 结果数: {len(results)}")
    print(f"结果目录: {runner.seed_dir}")
    return runner.seed_dir


def test_batch_experiment():
    """测试大实验"""
    print("\n" + "=" * 70)
    print("测试大实验 (2个种子)")
    print("=" * 70)
    
    runner = BatchExperimentRunner(
        batch_name='3C',
        n_repeats=2,  # 只有2个种子用于测试
        epochs=5,
        device='cpu',
        resume=False
    )
    
    exp_dir = runner.run()
    
    print(f"\n测试通过! 实验目录: {exp_dir}")
    return exp_dir


def main():
    print("新架构测试")
    print("这将在小规模数据上运行测试以验证架构正确性\n")
    
    # 测试单个小实验
    try:
        single_dir = test_single_experiment()
    except Exception as e:
        print(f"单实验测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试大实验
    try:
        batch_dir = test_batch_experiment()
    except Exception as e:
        print(f"大实验测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
    print(f"单实验结果: {single_dir}")
    print(f"大实验结果: {batch_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
