"""
测试插补器组件
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_zero_imputer():
    """测试零值插补器"""
    print("\n测试 ZeroImputer...")
    from src.missing.imputers import ZeroImputer
    
    # 创建测试数据
    X = np.array([[1.0, np.nan, 3.0],
                  [4.0, 5.0, np.nan]])
    
    # 创建插补器
    imputer = ZeroImputer()
    X_filled = imputer.fit(X).transform(X)
    
    # 验证
    assert X_filled[0, 1] == 0.0
    assert X_filled[1, 2] == 0.0
    assert not np.isnan(X_filled).any()
    print("✅ ZeroImputer 工作正常")


def test_mean_imputer():
    """测试均值插补器"""
    print("\n测试 MeanImputer...")
    from src.missing.imputers import MeanImputer
    
    # 训练数据（完整）
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]])
    
    # 测试数据（有缺失）
    X_test = np.array([[np.nan, 2.0],
                       [3.0, np.nan]])
    
    # 期望的均值: col0=(1+3+5)/3=3, col1=(2+4+6)/3=4
    imputer = MeanImputer()
    imputer.fit(X_train)
    
    # 验证学习到的均值
    assert np.allclose(imputer.means, [3.0, 4.0])
    
    # 填充
    X_filled = imputer.transform(X_test)
    assert X_filled[0, 0] == 3.0  # 用均值填充
    assert X_filled[1, 1] == 4.0
    
    print("✅ MeanImputer 工作正常")


def test_knn_imputer():
    """测试 KNN 插补器"""
    print("\n测试 KNNImputer...")
    
    try:
        from src.missing.imputers import KNNImputer
        
        # 训练数据
        X_train = np.array([[1.0, 2.0],
                            [2.0, 3.0],
                            [10.0, 20.0]])
        
        # 测试数据
        X_test = np.array([[1.5, np.nan]])  # 应该更接近 [1,2] 和 [2,3]
        
        imputer = KNNImputer(n_neighbors=2)
        X_filled = imputer.fit(X_train).transform(X_test)
        
        # 验证填充值在合理范围内
        assert 2.0 < X_filled[0, 1] < 3.0  # 应该在 2 和 3 之间
        print(f"✅ KNNImputer 工作正常 (filled value: {X_filled[0, 1]:.2f})")
        
    except ImportError as e:
        print(f"⚠️ 跳过 KNNImputer 测试: {e}")


def test_iterative_imputer():
    """测试迭代插补器"""
    print("\n测试 IterativeImputer...")
    
    try:
        from src.missing.imputers import IterativeImputer
        
        # 简单测试：验证功能可用
        X_train = np.random.randn(50, 4)
        X_test = X_train.copy()
        X_test[0, 0] = np.nan
        
        imputer = IterativeImputer(max_iter=5, random_state=42)
        X_filled = imputer.fit(X_train).transform(X_test)
        
        # 验证没有缺失值
        assert not np.isnan(X_filled).any()
        
        print(f"✅ IterativeImputer 工作正常")
        print(f"   成功填充 {np.isnan(X_test).sum()} 个缺失值")
        
    except ImportError as e:
        print(f"⚠️ 跳过 IterativeImputer 测试: {e}")


def test_registry_integration():
    """测试注册中心集成"""
    print("\n测试注册中心集成...")
    from src.core import IMPUTERS, create_imputer
    
    # 列出所有可用插补器
    available = IMPUTERS.list_available()
    print(f"   可用插补器: {available}")
    
    expected = ["zero", "mean", "knn", "iterative"]
    for name in expected:
        assert name in available, f"{name} 未注册"
    
    # 测试通过注册中心创建
    imputer = create_imputer("zero")
    from src.missing.imputers import ZeroImputer
    assert isinstance(imputer, ZeroImputer)
    
    print("✅ 注册中心集成正常")


def test_fit_transform_convenience():
    """测试 fit_transform 便捷方法"""
    print("\n测试 fit_transform 便捷方法...")
    from src.missing.imputers import MeanImputer
    
    X = np.array([[1.0, np.nan],
                  [3.0, 4.0]])
    
    imputer = MeanImputer()
    X_filled = imputer.fit_transform(X)  # 使用便捷方法
    
    assert X_filled[0, 0] == 1.0  # 未缺失不变
    assert X_filled[0, 1] == 4.0  # 填充均值 (4.0 是唯一非缺失值)
    
    print("✅ fit_transform 便捷方法正常")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("插补器组件测试")
    print("=" * 60)
    
    try:
        test_zero_imputer()
        test_mean_imputer()
        test_knn_imputer()
        test_iterative_imputer()
        test_registry_integration()
        test_fit_transform_convenience()
        
        print("\n" + "=" * 60)
        print("✅ 所有插补器测试通过！")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
