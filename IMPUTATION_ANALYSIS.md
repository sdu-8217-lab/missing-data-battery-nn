# Baseline插补方法分析报告

## 1. 当前实现方式

### 1.1 代码位置
- **数据标准化**: `src/data/dataset_loader.py` 第249-252行
- **缺失填充**: `src/data/datasets.py` 第33行、79行、139行等

### 1.2 实现流程

```
原始数据 → StandardScaler标准化 → 0填充缺失位置 → 输入模型
```

**关键代码**:
```python
# 1. 标准化 (dataset_loader.py)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 均值为0，方差为1

# 2. 缺失填充 (datasets.py)
X_missing = np.where(mask, X, 0.0)  # 缺失位置填0
```

## 2. 合理性分析

### 2.1 ✅ 合理的方面

| 方面 | 说明 |
|------|------|
| **数学等价性** | 对于标准化数据，0填充 ≈ 均值插补 |
| **简单高效** | 无需计算特征均值，计算复杂度O(1) |
| **可复现性** | 不依赖训练集的统计量，避免数据泄漏 |
| **与MIM兼容** | 配合缺失指示器，模型可以区分"真0"和"缺失0" |

### 2.2 ⚠️ 潜在问题

| 问题 | 影响 | 建议 |
|------|------|------|
| **硬编码0填充** | 如果取消标准化，插补将失效 | 添加插补方法配置选项 |
| **无时序信息利用** | 电池数据是时序的，未利用前后循环信息 | 考虑前向/后向填充 |
| **单一插补方法** | 无法进行插补方法敏感性分析 | 支持多种插补方法 |

## 3. 与其他插补方法的对比

| 插补方法 | 实现复杂度 | 对标准化数据的适用性 | 是否需MIM |
|----------|-----------|---------------------|----------|
| **0填充 (当前)** | ⭐ | ✅ 好 (等价于均值) | 推荐配合 |
| 均值填充 | ⭐⭐ | ⚠️ 冗余 (标准化后均值为0) | 推荐配合 |
| 前向填充 | ⭐⭐⭐ | ✅ 适合时序数据 | 推荐配合 |
| 线性插值 | ⭐⭐⭐⭐ | ✅ 适合时序数据 | 推荐配合 |
| 多重插补 | ⭐⭐⭐⭐⭐ | ✅ 统计严谨 | 必须配合 |

## 4. 实验设计层面的验证

### 4.1 论文要求检查

根据 `docs/10-实验设计与计划.md`:
> "imputation: "mean"  # 标准化后的0填充"

✅ **当前实现符合论文设计** - 明确指定了标准化后的0填充

### 4.2 RQ3: 插补方法敏感性

如果论文需要分析插补方法的影响，当前代码**不支持**灵活切换插补方法。

## 5. 改进建议

### 5.1 短期（保持现状）

当前实现对于论文的主要研究问题（MIM效果）是足够的：
- 所有模型使用相同的插补方法，对比公平
- 标准化后的0填充是合理的基线

### 5.2 中期（建议优化）

添加插补方法配置选项：

```python
# src/data/datasets.py
class BatteryDataset(Dataset):
    def __init__(self, ..., imputation_method: str = 'zero'):
        self.imputation_method = imputation_method
        
        if imputation_method == 'zero':
            self.X_missing = np.where(mask, X, 0.0)
        elif imputation_method == 'mean':
            # 使用特征均值填充
            feature_means = X.mean(axis=0)
            self.X_missing = np.where(mask, X, feature_means)
        elif imputation_method == 'forward':
            # 前向填充（需按电池分组）
            ...
```

### 5.3 长期（完整方案）

实现插补策略抽象基类：

```python
from abc import ABC, abstractmethod

class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass

class ZeroImputation(ImputationStrategy):
    def impute(self, X, mask):
        return np.where(mask, X, 0.0)

class ForwardFillImputation(ImputationStrategy):
    def impute(self, X, mask):
        # 实现前向填充
        ...
```

## 6. 结论

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确。标准化后的0填充等价于均值插补 |
| **合理性** | ✅ 合理。与论文设计一致，适合MIM对比实验 |
| **灵活性** | ⚠️ 不足。硬编码实现，不利于扩展 |
| **可维护性** | ⚠️ 一般。缺少插补方法的显式配置 |

### 最终建议

**对于当前论文**: 保持现状，当前实现足够支撑主要研究问题

**对于代码复用**: 建议添加插补方法配置，提高灵活性
