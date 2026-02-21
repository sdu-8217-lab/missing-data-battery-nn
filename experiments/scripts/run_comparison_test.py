# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
瀵规瘮娴嬭瘯锛氭棫鐗堜唬鐮?vs 鏂扮増浠ｇ爜
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_old_version():
    """娴嬭瘯鏃х増浠ｇ爜"""
    print("=" * 60)
    print("[鏃х増浠ｇ爜] 浣跨敤 dataclass + logging + 鑷畾涔夎缁冨櫒")
    print("=" * 60)
    
    from src.config import ExperimentConfig
    from src.utils import setup_logger
    
    start = time.time()
    
    # 鍒涘缓閰嶇疆
    config = ExperimentConfig(
        batch="3C",
        random_seed=42,
        n_repeats=1,
        epochs=50
    )
    
    # 璁剧疆鏃ュ織
    logger = setup_logger("old_test", "./logs/old_version.log")
    logger.info(f"鏃х増閰嶇疆鍒涘缓鎴愬姛: {config.name}")
    logger.info(f"鎵规: {config.batch}")
    logger.info(f"妯″瀷鏁伴噺: {len(config.get_model_configs())}")
    
    elapsed = time.time() - start
    print(f"鏃х増浠ｇ爜鍒濆鍖栨椂闂? {elapsed:.3f}s")
    print(f"鏃ュ織鏂囦欢: ./logs/old_version.log")
    return elapsed


def test_new_version():
    """娴嬭瘯鏂扮増浠ｇ爜"""
    print("\n" + "=" * 60)
    print("[鏂扮増浠ｇ爜] 浣跨敤 Pydantic + Loguru + Lightning")
    print("=" * 60)
    
    from src.config import ExperimentConfigV2
    from src.utils import setup_logger
    
    start = time.time()
    
    # 鍒涘缓閰嶇疆
    config = ExperimentConfigV2(
        batch="3C",
        random_seed=42,
        n_repeats=1,
        epochs=50
    )
    
    # 璁剧疆鏃ュ織
    logger = setup_logger("new_test", "./logs/new_version.log")
    logger.info(f"鏂扮増閰嶇疆鍒涘缓鎴愬姛: {config.name}")
    logger.info(f"鎵规: {config.batch}")
    logger.info(f"妯″瀷鏁伴噺: {len(config.get_model_configs())}")
    
    # 娴嬭瘯楠岃瘉鍔熻兘
    try:
        invalid = ExperimentConfigV2(test_size=1.5)
        print("楠岃瘉澶辫触!")
    except Exception as e:
        logger.info(f"楠岃瘉鍔熻兘姝ｅ父: {type(e).__name__}")
    
    elapsed = time.time() - start
    print(f"鏂扮増浠ｇ爜鍒濆鍖栨椂闂? {elapsed:.3f}s")
    print(f"鏃ュ織鏂囦欢: ./logs/new_version.log")
    return elapsed


def main():
    print("寮€濮嬪姣旀祴璇? 鏃х増 vs 鏂扮増浠ｇ爜")
    print("=" * 60)
    
    old_time = test_old_version()
    new_time = test_new_version()
    
    print("\n" + "=" * 60)
    print("瀵规瘮缁撴灉")
    print("=" * 60)
    print(f"鏃х増浠ｇ爜鍒濆鍖? {old_time:.3f}s")
    print(f"鏂扮増浠ｇ爜鍒濆鍖? {new_time:.3f}s")
    
    if new_time < old_time:
        print(f"鏂扮増鏇村揩: {(old_time - new_time)/old_time*100:.1f}%")
    else:
        print(f"鏃х増鏇村揩: {(new_time - old_time)/old_time*100:.1f}%")
    
    print("\n[缁撹]")
    print("- 鏂扮増浠ｇ爜鎻愪緵浜嗘洿濂界殑绫诲瀷瀹夊叏鍜岄獙璇?)
    print("- Loguru鏃ュ織鏇村弸濂斤紝鑷姩澶勭悊UTF-8缂栫爜")
    print("- Lightning璁粌鍣ㄥぇ骞呯畝鍖栬缁冧唬鐮?)
    print("- 鍙互閫氳繃杩愯瀹為檯瀹為獙鏉ュ姣旇缁冩€ц兘")


if __name__ == "__main__":
    main()

