"""缺失模式生成器单元测试（不依赖 pytest）"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.data.missing_patterns import (
    create_missing_pattern,
    MISSING_PATTERN_REGISTRY,
)


def sample_data(seed=42, n_samples=500, n_features=16):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.linspace(1.0, 0.5, n_samples).astype(np.float32)
    return X, y


def test_missing_rate_accuracy():
    X, y = sample_data()
    for pattern_name in MISSING_PATTERN_REGISTRY.keys():
        for missing_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            generator = create_missing_pattern(pattern_name)
            mask = generator.generate(X, y, missing_rate, seed=42)
            actual = 1.0 - mask.mean()
            assert abs(actual - missing_rate) < 0.02, (
                f"{pattern_name} @ MR={missing_rate}: actual={actual:.4f}"
            )
    print("[OK] missing_rate_accuracy")


def test_reproducibility():
    X, y = sample_data()
    for pattern_name in MISSING_PATTERN_REGISTRY.keys():
        generator = create_missing_pattern(pattern_name)
        mask1 = generator.generate(X, y, 0.3, seed=123)
        mask2 = generator.generate(X, y, 0.3, seed=123)
        assert np.array_equal(mask1, mask2), f"{pattern_name} 不可复现"
    print("[OK] reproducibility")


def test_state_dependent_correlation():
    X, y = sample_data()
    generator = create_missing_pattern("state_dependent", alpha=1.0, use_target_for_state=True)
    mask = generator.generate(X, y, 0.3, seed=42)

    median_soh = np.median(y)
    low_missing = 1.0 - mask[y <= median_soh].mean()
    high_missing = 1.0 - mask[y > median_soh].mean()
    assert low_missing > high_missing, (
        f"低 SOH 缺失率 {low_missing:.4f} 应高于高 SOH 缺失率 {high_missing:.4f}"
    )
    print("[OK] state_dependent_correlation")


def test_channel_pattern_structure():
    X, y = sample_data()
    generator = create_missing_pattern("channel")
    mask = generator.generate(X, y, 0.3, seed=42)

    col_missing = 1.0 - mask.mean(axis=0)
    n_empty = np.sum(col_missing < 0.01)
    n_affected = np.sum(col_missing > 0.1)
    assert n_empty >= 1 and n_affected >= 1, "channel 模式应同时存在未缺失列与高缺失列"
    print("[OK] channel_pattern_structure")


def test_block_pattern_temporal_structure():
    X, y = sample_data()
    block_gen = create_missing_pattern("block")
    bern_gen = create_missing_pattern("bernoulli")
    block_mask = block_gen.generate(X, y, 0.3, seed=42)
    bern_mask = bern_gen.generate(X, y, 0.3, seed=42)

    col_block = block_mask[:, 0].astype(int)
    col_bern = bern_mask[:, 0].astype(int)
    block_transitions = np.sum(np.abs(np.diff(col_block)))
    bern_transitions = np.sum(np.abs(np.diff(col_bern)))
    assert block_transitions < bern_transitions, (
        f"块缺失转换次数 {block_transitions} 应低于伯努利 {bern_transitions}"
    )
    print("[OK] block_pattern_temporal_structure")


def test_mixed_pattern_composition():
    X, y = sample_data()
    generator = create_missing_pattern("mixed")
    mask = generator.generate(X, y, 0.5, seed=42)
    assert mask.shape == X.shape
    actual = 1.0 - mask.mean()
    assert abs(actual - 0.5) < 0.02, f"mixed 模式缺失率偏差过大: {actual:.4f}"
    print("[OK] mixed_pattern_composition")


if __name__ == "__main__":
    test_missing_rate_accuracy()
    test_reproducibility()
    test_state_dependent_correlation()
    test_channel_pattern_structure()
    test_block_pattern_temporal_structure()
    test_mixed_pattern_composition()
    print("\n所有测试通过！")
