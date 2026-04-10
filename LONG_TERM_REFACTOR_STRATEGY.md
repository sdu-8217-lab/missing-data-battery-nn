# 长期主义重构战略

> **目标**：构建电池健康状态预测领域的标准开源框架  
> **愿景**：成为缺失数据研究的基础工具，支撑未来10年的学术发展  
> **时间跨度**：3个月重构期 + 长期维护

---

## 一、长期主义核心原则

### 1.1 开源项目成功公式

```
成功 = 清晰的问题定义 × 优秀的架构设计 × 完善的开发者体验 × 活跃的社区
```

**当前状态评估**：
| 要素 | 当前 | 目标 | 差距 |
|------|------|------|------|
| 问题定义 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 已清晰 |
| 架构设计 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 需对齐实现 |
| 开发者体验 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 需大量工作 |
| 社区基础 | ⭐ | ⭐⭐⭐⭐ | 从零开始 |

### 1.2 长期主义技术债务观

**技术债务的三种处理方式**：

| 方式 | 短期成本 | 长期成本 | 适用场景 |
|------|---------|---------|---------|
| **忽视** | 低 | 极高（指数增长）| ❌ 不适用 |
| **迁移** | 中 | 高（持续维护旧代码）| 过渡方案 |
| **偿还** | 高 | 低（线性维护）| ✅ 长期主义选择 |

**决策**：彻底偿还技术债务，不保留任何妥协方案。

### 1.3 开源质量标准

以 **PyTorch Lightning** 和 **Hugging Face Transformers** 为标杆：

- ✅ **文档全面**：API文档、教程、示例
- ✅ **测试严格**：>90%覆盖率，CI/CD自动化
- ✅ **向后兼容**：语义化版本，清晰的弃用策略
- ✅ **社区友好**：清晰的贡献流程，响应及时
- ✅ **可扩展性**：插件系统，模块化设计

---

## 二、重构战略框架

### 2.1 三阶段重构路线

```
Phase 1: 基础重建 (4周)
    └── 彻底清理，建立正确架构
    └── 输出：可运行的最小核心
    
Phase 2: 能力建设 (4周)
    └── 完善功能，建立质量门禁
    └── 输出：功能完整的Alpha版本
    
Phase 3: 开源准备 (4周)
    └── 文档、测试、社区准备
    └── 输出：可发布的开源项目
```

### 2.2 重构的核心准则

```python
"""
长期主义重构准则
"""

class RefactorPrinciples:
    # 1. 宁可慢，不可错
    RUSH_TO_COMPLETE = False
    DO_IT_RIGHT = True
    
    # 2. 删除优于保留
    WHEN_IN_DOUBT = "DELETE"
    
    # 3. 简单优于复杂
    PREFER = "SIMPLE"
    OVER = "CLEVER"
    
    # 4. 显式优于隐式
    EXPLICITNESS = "MANDATORY"
    
    # 5. 现在投资未来
    SHORT_TERM_PAIN = True
    LONG_TERM_GAIN = True
```

---

## 三、Phase 1: 基础重建（第1-4周）

### 3.1 战略删除

**删除一切非必要代码，从零开始重建**

```bash
# 第1周：清理周

# 1. 删除所有临时/实验代码
rm -rf experiments_new/
rm -rf .rework/
rm -rf src/missing_data/fixed_feature_missing.py
rm -f src/models/*_simple.py
rm -f src/models/model_factory.py
rm -f src/experiments/simple_runner.py

# 2. 删除重复实现，只保留最好的一个
# 保留：src/models/factory.py（最完整）
rm -f src/models/factory_simple.py

# 3. 删除未使用的功能
# 保留核心：MLP/LSTM/CNN，删除预留的GRU等
rm -f src/models/gru.py

# 4. 删除旧架构残余
rm -f src/main.py  # 旧Hydra入口
rm -f src/trainers/neural_network_trainer.py  # 重复
rm -f src/trainers/xgboost_trainer.py  # 预留

# 5. 清理测试
rm -f tests/test_simplified.py
rm -f tests/test_p1_mock.py
rm -f tests/test_data_loader_smoke.py

# 预计删除：~5000行代码，释放认知负担
```

### 3.2 核心架构重建

**严格遵循meta.md的9层架构，代码即文档**

```python
# src/battery_soh/  (新包名，更清晰)
"""
Battery SOH Prediction Framework

A long-term open-source framework for battery state-of-health prediction
under missing data scenarios.

Architecture: 9-Level Experimental Framework
See: docs/ARCHITECTURE.md
"""

# 目录结构（最终目标）
battery_soh/
├── core/               # 核心抽象（接口、类型、常量）
│   ├── types.py       # 类型别名、常量
│   ├── interfaces.py  # Protocol定义
│   └── constants.py   # 物理常量、枚举
├── data/              # 数据层
│   ├── loaders.py     # 数据加载统一接口
│   ├── transforms.py  # 特征工程
│   └── splits.py      # 数据划分策略
├── models/            # 模型层
│   ├── architectures/ # 网络架构
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   └── cnn.py
│   └── factory.py     # 模型创建
├── missing/           # 缺失数据处理
│   ├── generators/    # 缺失模式生成
│   │   ├── base.py
│   │   ├── mcar.py
│   │   ├── mar.py
│   │   └── mnar.py
│   └── imputers/      # 插补方法
│       ├── base.py
│       ├── mean.py
│       ├── knn.py
│       └── iterative.py
├── training/          # 训练层
│   ├── trainer.py     # 训练器抽象
│   ├── lightning.py   # Lightning实现
│   └── callbacks.py   # 回调函数
├── evaluation/        # 评估层
│   ├── metrics.py     # 评估指标
│   └── evaluator.py   # 评估器
├── experiments/       # 实验执行
│   ├── runner.py      # 实验运行器
│   └── config.py      # 实验配置
└── utils/             # 工具
    ├── seed.py        # 随机种子管理
    └── logging.py     # 日志工具
```

### 3.3 第1周详细计划：删除与重建

**Day 1-2: 删除周**
```bash
# 创建删除分支
git checkout -b refactor/cleanup

# 执行战略删除
# ... 删除命令 ...

# 提交
git add .
git commit -m "refactor!: strategic deletion for long-term health

BREAKING CHANGE: Remove all temporary and duplicate code

Removed:
- experiments_new/ (temporary parallel implementation)
- .rework/ (legacy configs)
- *simple*.py (transitional files)
- Unused model architectures (GRU, XGBoost)
- Old Hydra entry points

This is the foundation for a clean, maintainable architecture.
"
```

**Day 3-4: 重建核心**
```python
# 从最核心的抽象开始

# 1. core/types.py
"""Core type definitions."""
from typing import NewType, Literal
import numpy as np
from numpy.typing import NDArray

# 类型别名 - 自文档化
Seed = NewType("Seed", int)
MissingRate = NewType("MissingRate", float)
BatchId = Literal["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
ModelType = Literal["mlp", "lstm", "cnn"]
MissingMode = Literal["MCAR", "MAR", "MNAR"]
ImputationMethod = Literal["mean", "knn", "iterative", "zero"]

# 数组类型
Features = NDArray[np.float32]
Labels = NDArray[np.float32]
Mask = NDArray[np.bool_]
BatteryIds = NDArray[np.str_]

# 常量
N_FEATURES = 16
N_FEATURES_WITH_MIM = 32
TRAIN_TEST_SPLIT_RATIO = 0.25
VALIDATION_SPLIT_RATIO = 0.25
```

**Day 5: 验证**
```bash
# 确保删除后项目仍能运行（通过测试）
python -c "import battery_soh; print('Core import OK')"

# 提交
git commit -m "feat(core): establish clean type system"
```

---

## 四、Phase 2: 能力建设（第5-8周）

### 4.1 逐步实现功能

**第5-6周：数据层**
```python
# data/loaders.py
from battery_soh.core.types import BatchId, Features, Labels, BatteryIds
from battery_soh.core.interfaces import DataLoader

class XJTULoader(DataLoader):
    """XJTU dataset loader.
    
    Loads battery data from the XJTU dataset.
    
    Args:
        data_dir: Path to the XJTU data directory
        
    Example:
        >>> loader = XJTULoader()
        >>> X, y, battery_ids = loader.load_batch("2C")
    """
    
    BATTERIES: list[BatchId] = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
    
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/XJTU data")
    
    def load_batch(
        self, 
        batch_id: BatchId
    ) -> tuple[Features, Labels, BatteryIds]:
        """Load a specific batch.
        
        Args:
            batch_id: One of the 6 XJTU battery batches
            
        Returns:
            Tuple of (features, labels, battery_ids)
            
        Raises:
            FileNotFoundError: If batch data not found
            ValueError: If batch_id is invalid
        """
        ...
```

**第7周：模型层**
```python
# models/architectures/mlp.py
import torch.nn as nn
from battery_soh.core.types import ModelType

class MLP(nn.Module):
    """Multi-Layer Perceptron for SOH prediction.
    
    Architecture constraints (per meta.md):
    - Parameter count: 16,384 ~ 32,768
    - Input dim: 16 (no MIM) or 32 (with MIM)
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15
    ):
        super().__init__()
        
        # Default configs per meta.md
        if hidden_dims is None:
            hidden_dims = [128, 96] if input_dim == 32 else [192, 96]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Predictions of shape [batch_size]
        """
        return self.network(x).squeeze(-1)
```

**第8周：缺失数据处理**
```python
# missing/generators/mcar.py
import numpy as np
from battery_soh.core.types import Features, Mask, MissingRate, Seed
from battery_soh.core.interfaces import MissingGenerator

class MCARGenerator(MissingGenerator):
    """MCAR (Missing Completely At Random) generator.
    
    Missing mechanism: Missing probability is independent of observed
    and unobserved values.
    
    Reference:
        Rubin, D. B. (1976). Inference and missing data. Biometrika.
    """
    
    def generate(
        self,
        X: Features,
        missing_rate: MissingRate,
        seed: Seed
    ) -> tuple[Features, Mask]:
        """Generate MCAR missing pattern.
        
        Args:
            X: Original feature matrix [N, D]
            missing_rate: Target missing rate [0, 1]
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (X_with_nan, mask)
            - X_with_nan: Feature matrix with NaN for missing values
            - mask: Boolean mask, True = observed, False = missing
        """
        rng = np.random.default_rng(seed)
        mask = rng.random(X.shape) > missing_rate
        
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        return X_missing, mask
```

### 4.2 建立质量门禁

**自动化检查（第8周开始实施）**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Type check
        run: mypy battery_soh/ --strict
      
      - name: Lint
        run: ruff check battery_soh/
      
      - name: Test with coverage
        run: pytest tests/ --cov=battery_soh --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**质量指标**

| 指标 | 目标 | 检查频率 |
|------|------|---------|
| 测试覆盖率 | >90% | 每次提交 |
| 类型检查 | mypy strict 0错误 | 每次提交 |
| 代码风格 | ruff 0警告 | 每次提交 |
| 文档覆盖率 | >80% | 每周 |
| 性能回归 | <5% | 每周 |

---

## 五、Phase 3: 开源准备（第9-12周）

### 5.1 文档建设

```
docs/
├── index.md              # 文档首页
├── quickstart.md         # 5分钟快速开始
├── architecture.md       # 架构详细说明
├── api/                  # API文档（自动生成）
├── tutorials/            # 教程
│   ├── 01_basic_usage.md
│   ├── 02_custom_model.md
│   ├── 03_custom_missing_pattern.md
│   └── 04_reproduce_paper.md
├── contributing.md       # 贡献指南
└── changelog.md          # 变更日志
```

**关键文档：设计决策记录（ADR）**

```markdown
# docs/adr/001-9-layer-architecture.md

# 9层实验架构设计决策

## 状态
已接受

## 背景
电池SOH预测实验涉及多个维度的组合：
- 数据层面（批次、划分）
- 模型层面（架构、MIM使用）
- 缺失模拟层面（模式、比率、插补）

## 决策
采用9层架构，并引入"分界线原则"。

## 原因
1. **计算效率**：分界线以下可复用模型
2. **科学严谨**：清晰分离训练与测试变量
3. **可扩展性**：各层独立变化，正交设计

## 后果
- 正面：实验规模可控，432K测试仅需14.4K训练
- 负面：概念学习曲线陡峭

## 参考
- meta.md 第2节
- Rubin, D. B. (1976). Inference and missing data.
```

### 5.2 示例与教程

```python
# examples/01_basic_usage.py
"""Basic usage example.

This example demonstrates how to:
1. Load battery data
2. Train a model
3. Evaluate under missing data scenarios
"""

from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model
from battery_soh.training import Trainer
from battery_soh.evaluation import Evaluator

# 1. Load data
loader = XJTULoader()
X, y, battery_ids = loader.load_batch("2C")

# 2. Split data (battery-wise to avoid leakage)
splitter = BatteryWiseSplit(seed=42)
train_data, val_data, test_data = splitter.split(X, y, battery_ids)

# 3. Create model (without MIM)
model = create_model("mlp", use_mim=False)

# 4. Train
trainer = Trainer(epochs=200, patience=30)
result = trainer.fit(model, train_data, val_data)
print(f"Best val loss: {result.best_val_loss:.4f}")

# 5. Evaluate under MCAR with 30% missing
evaluator = Evaluator(
    missing_mode="MCAR",
    missing_rate=0.3,
    imputation_method="mean"
)
metrics = evaluator.evaluate(model, test_data)
print(f"Test MAE: {metrics.mae:.4f}")
```

### 5.3 社区建设准备

**贡献者指南**

```markdown
# CONTRIBUTING.md

## 开发流程

1. **Fork and clone**
   ```bash
   git clone https://github.com/yourusername/battery-soh-framework.git
   cd battery-soh-framework
   ```

2. **Set up development environment**
   ```bash
   conda env create -f environment.yml
   conda activate battery-soh
   pip install -e ".[dev]"
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make changes and test**
   ```bash
   # Run tests
   pytest tests/
   
   # Run type check
   mypy battery_soh/ --strict
   
   # Run linting
   ruff check battery_soh/
   ```

5. **Submit PR**
   - PR title follows conventional commits
   - All CI checks pass
   - Code review approved

## 代码规范

- **类型注解**：所有函数必须有完整类型注解
- **文档字符串**：所有公共API必须有docstring
- **测试**：新增代码必须有测试覆盖
- **单一职责**：函数<50行，文件<300行
```

### 5.4 发布准备

**版本策略**

```
主版本.次版本.修订号 (SemVer)
- 主版本：破坏性变更
- 次版本：向后兼容的功能添加
- 修订号：bug修复
```

**Alpha 版本发布检查清单**

- [ ] 核心功能完整（数据加载、训练、评估）
- [ ] 文档完整（API、教程、示例）
- [ ] 测试覆盖率>90%
- [ ] CI/CD配置完成
- [ ] 许可证确定（建议MIT或Apache-2.0）
- [ ] README完整
- [ ] CHANGELOG建立

---

## 六、长期维护策略

### 6.1 维护节奏

| 活动 | 频率 | 负责人 |
|------|------|--------|
| 依赖更新 | 每周 | 自动化（Dependabot） |
| 安全审计 | 每月 | 维护者 |
| 性能基准 | 每月 | CI |
| 社区Q&A | 每周 | 维护者 |
| 发布 | 按需 | 维护者 |

### 6.2 技术债务监控

```python
# 定期运行（每季度）
def technical_debt_audit():
    """Audit codebase for technical debt."""
    
    # 1. Code complexity
    complexity_report = run_radon_cc()
    if complexity_report.avg > 5:
        alert("Average cyclomatic complexity too high")
    
    # 2. Test coverage
    coverage = run_coverage()
    if coverage < 90:
        alert("Test coverage dropped below 90%")
    
    # 3. Type coverage
    type_coverage = run_mypy_coverage()
    if type_coverage < 95:
        alert("Type coverage dropped below 95%")
    
    # 4. Documentation coverage
    doc_coverage = run_docstr_coverage()
    if doc_coverage < 80:
        alert("Documentation coverage dropped below 80%")
```

### 6.3 社区治理

**初期（<10 contributors）**
- 单一维护者决策
- 简单贡献流程

**成长期（10-50 contributors）**
- 建立维护者团队
- 明确的决策流程
- 定期社区会议

**成熟期（>50 contributors）**
- 正式治理模型（如BDFL或委员会）
- 年度路线图
- 基金会支持（如需要）

---

## 七、风险管理

### 7.1 重构风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 重构时间超预期 | 高 | 中 | 每周检查点，必要时缩小范围 |
| 功能回归 | 中 | 高 | 全面测试覆盖，渐进式替换 |
| 社区不认同 | 低 | 高 | 早期社区参与，公开决策过程 |
| 维护者倦怠 | 中 | 高 | 文档化知识，培养核心贡献者 |

### 7.2 长期可持续性风险

| 风险 | 应对策略 |
|------|---------|
| 维护者离开 | 多维护者模式，文档化所有决策 |
| 资金不足 | 申请学术/开源基金，企业赞助 |
| 技术过时 | 保持架构简洁，易于迁移 |
| 社区分裂 | 明确治理模型，透明决策 |

---

## 八、成功度量

### 8.1 技术指标（3个月后）

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| 代码行数 | <5,000 | cloc |
| 测试覆盖率 | >90% | pytest-cov |
| 类型覆盖率 | >95% | mypy |
| 文档覆盖率 | >80% | docstr-coverage |
| 平均函数复杂度 | <5 | radon cc |

### 8.2 社区指标（1年后）

| 指标 | 目标 |
|------|------|
| GitHub Stars | >500 |
| 活跃贡献者 | >10 |
| 学术论文引用 | >10 |
| 企业用户 | >3 |

### 8.3 科学影响指标（3年后）

| 指标 | 目标 |
|------|------|
| 基于框架发表的论文 | >20 |
| 支持的电池数据集 | >5 |
| 集成的深度学习模型 | >10 |
| 应用领域扩展 | 电动汽车、储能、航天 |

---

## 九、立即开始：本周行动计划

### Day 1-2: 决策与准备
- [ ] 确认长期主义目标与团队/导师
- [ ] 申请必要的计算资源
- [ ] 创建重构专用分支 `refactor/long-term`

### Day 3-4: 战略删除
- [ ] 执行Phase 1的删除计划
- [ ] 提交第一个commit："refactor!: strategic deletion"

### Day 5: 重建核心
- [ ] 创建新的包结构 `battery_soh/`
- [ ] 实现 `core/types.py` 和 `core/constants.py`
- [ ] 提交第二个commit："feat(core): establish type system"

### Day 6-7: 验证与规划
- [ ] 运行基础导入测试
- [ ] 细化Phase 2的计划
- [ ] 撰写第一篇开发日志（博客/讨论区）

---

**承诺**：
> 我们承诺以长期主义精神重构此项目，不为短期便利牺牲代码质量，每一行代码都为未来10年的学术发展负责。

**下一步**：开始Phase 1的第一周工作？
