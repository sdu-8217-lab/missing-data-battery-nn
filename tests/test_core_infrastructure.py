"""
测试核心基础设施

验证接口定义和注册中心是否工作正常。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_imports():
    """测试导入是否正常"""
    print("测试导入...")
    from src.core import (
        IImputer, IMissingGenerator, IModel, ITrainer, IDataLoader,
        BatteryDataset, TrainingConfig, ExperimentConfig,
        Registry, IMPUTERS, MISSING_GENERATORS, MODELS
    )
    print("✅ 所有导入成功")


def test_battery_dataset():
    """测试 BatteryDataset"""
    print("\n测试 BatteryDataset...")
    from src.core import BatteryDataset
    
    # 正常创建
    ds = BatteryDataset(
        features=np.random.randn(100, 16),
        labels=np.random.randn(100),
        battery_ids=np.array([f"B{i}" for i in range(100)]),
        metadata={"batch": "2C"}
    )
    assert ds.features.shape == (100, 16)
    print("✅ BatteryDataset 创建成功")
    
    # 测试维度不匹配会报错
    try:
        ds_bad = BatteryDataset(
            features=np.random.randn(100, 16),
            labels=np.random.randn(50),  # 不匹配
            battery_ids=np.array([f"B{i}" for i in range(100)])
        )
        assert False, "应该抛出异常"
    except AssertionError as e:
        if "labels" in str(e):
            print("✅ 维度检查正常工作")


def test_registry_basic():
    """测试注册中心基本功能"""
    print("\n测试 Registry 基本功能...")
    from src.core import Registry
    
    # 创建测试注册中心
    test_reg = Registry("test", base_class=None)
    
    # 注册组件
    @test_reg.register("component_a")
    class ComponentA:
        def __init__(self, value=1):
            self.value = value
    
    @test_reg.register("component_b")
    class ComponentB:
        def __init__(self, value=2):
            self.value = value
    
    # 测试创建实例
    a = test_reg.create("component_a", value=10)
    assert a.value == 10
    print(f"✅ 创建实例成功: ComponentA(value={a.value})")
    
    # 测试列出可用组件
    available = test_reg.list_available()
    assert "component_a" in available
    assert "component_b" in available
    print(f"✅ 列出组件成功: {available}")
    
    # 测试获取不存在的组件
    try:
        test_reg.get("nonexistent")
        assert False, "应该抛出 KeyError"
    except KeyError as e:
        print(f"✅ 错误处理正常: {e}")
    
    # 测试重复注册
    try:
        @test_reg.register("component_a")
        class ComponentADuplicate:
            pass
        assert False, "应该抛出 KeyError"
    except KeyError:
        print("✅ 重复注册检查正常")
    
    # 清理
    test_reg.unregister("component_a")
    test_reg.unregister("component_b")


def test_registry_with_base_class():
    """测试带基类约束的注册中心"""
    print("\n测试带基类约束的 Registry...")
    from src.core import Registry
    
    class BaseClass:
        pass
    
    class ValidSubclass(BaseClass):
        pass
    
    class InvalidClass:
        pass
    
    reg = Registry("test_base", base_class=BaseClass)
    
    # 有效的子类可以注册
    @reg.register("valid")
    class RegisteredValid(ValidSubclass):
        pass
    print("✅ 有效子类注册成功")
    
    # 非子类应该报错
    try:
        @reg.register("invalid")
        class RegisteredInvalid(InvalidClass):
            pass
        assert False, "应该抛出 TypeError"
    except TypeError as e:
        print(f"✅ 基类约束检查正常: {e}")
    
    reg.unregister("valid")


def test_global_registries():
    """测试全局注册中心"""
    print("\n测试全局注册中心...")
    from src.core import (
        IMPUTERS, MISSING_GENERATORS, MODELS, TRAINERS, DATA_LOADERS,
        list_all_components
    )
    
    # 初始时应该为空
    print(f"IMPUTERS: {IMPUTERS}")
    print(f"MISSING_GENERATORS: {MISSING_GENERATORS}")
    
    # 列出所有组件
    all_components = list_all_components()
    print(f"✅ 所有注册中心初始化成功")
    print(f"   当前组件: {all_components}")


def test_experiment_config():
    """测试 ExperimentConfig"""
    print("\n测试 ExperimentConfig...")
    from src.core import ExperimentConfig, TrainingConfig
    
    # 默认配置
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.batch_id == "2C"
    assert len(config.missing_rates) == 20  # 20个缺失率
    assert config.training_config.epochs == 200
    print(f"✅ 默认配置正常: seed={config.seed}, MRs={len(config.missing_rates)}")
    
    # 自定义配置
    config2 = ExperimentConfig(
        seed=123,
        batch_id="3C",
        model_type="lstm",
        use_mim=True,
        train_imputation="mean"
    )
    assert config2.seed == 123
    assert config2.model_type == "lstm"
    assert config2.use_mim == True
    print(f"✅ 自定义配置正常: {config2}")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("核心基础设施测试")
    print("=" * 60)
    
    try:
        test_imports()
        test_battery_dataset()
        test_registry_basic()
        test_registry_with_base_class()
        test_global_registries()
        test_experiment_config()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！基础设施工作正常。")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
