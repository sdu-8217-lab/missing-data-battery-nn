# 代码诊断报告与重构计划

## 一、诊断发现的问题

### 1.1 重复代码问题 (高优先级)

| 问题 | 位置 | 影响 | 说明 |
|------|------|------|------|
| **3个模型工厂** | `models/factory.py`, `factory_simple.py`, `model_factory.py` | 维护困难 | 功能重复，应该合并为1个 |
| **2套缺失数据处理** | `missing/`, `missing_data/` | 逻辑分散 | 新旧两套实现并存 |
| **多个数据加载器** | `xjtu_loader.py`, `loader.py`, `dataset_loader.py`, `datasets.py` | 职责不清 | 应该统一为1个 |
| **冗余文件** | `fixed_feature_missing.py` (715行) | 代码膨胀 | 29KB的旧代码，未使用 |
| **简化版重复** | `*simple*.py` (3个文件) | 版本混乱 | 临时简化版本应该被正取代 |
| **并行实现** | `experiments_new/` (30个文件) | 分支分散 | 之前重构尝试未清理 |

### 1.2 架构设计问题 (中优先级)

| 问题 | 影响 | 说明 |
|------|------|------|
| **sys.path.insert滥用** | 可移植性差 | 多处手动修改Python路径 |
| **循环导入风险** | 维护困难 | 34处 `from src.` 导入 |
| **配置分散** | 管理困难 | 配置在YAML、dataclass、函数参数多处定义 |
| **接口与实现混杂** | 耦合度高 | `interfaces.py` 包含具体实现代码 |

### 1.3 文件规模问题

| 文件 | 行数 | 问题 |
|------|------|------|
| `experiments/run_experiment.py` | 898行 | 过大，职责过多 |
| `src/data/loader_dataloaders.py` | 347行 | 应该拆分 |
| `src/preexperiment/runner.py` | 473行 | 应该拆分 |
| `src/trainers/neural_network_trainer.py` | 507行 | 和Lightning重复 |

### 1.4 测试覆盖问题

- 测试文件10个，但大量核心逻辑未覆盖
- 存在 `test_simplified.py`, `test_p1_mock.py` 等临时测试
- 缺少集成测试

### 1.5 命名规范问题

- 文件命名风格不统一：`xjtu_loader.py` vs `loader_dataloaders.py`
- 类命名有冗余：`SOHLightningModule` 应该为 `SOHModule`
- 函数命名冗长：`train_val_test_split_by_battery` 应该为 `split_batteries`

---

## 二、重构坚守原则

### 2.1 核心原则 (必须遵守)

```
1. 必要性原则
   └── 每个文件、函数、类必须有明确且唯一的职责
   └── 删除任何未使用的代码
   └── 删除任何可以通过简单组合实现的包装代码

2. 有效性原则
   └── 所有代码必须通过类型检查 (mypy --strict)
   └── 所有公共函数必须有文档字符串和类型注解
   └── 所有模块必须可通过简单导入使用

3. 简洁性原则
   └── 行数限制: 文件<300行, 函数<50行, 类<200行
   └── 参数限制: 函数参数<5个，超过用dataclass
   └── 嵌套限制: 缩进层级<4层

4. 最优实现原则
   └── 使用标准库优于第三方库
   └── 使用生成器优于列表 (大数据集)
   └── 使用内置函数优于手写循环

5. 版本管理原则
   └── 所有变更通过git管理
   └── 不在仓库中保留任何旧版本代码
   └── 不在命名中使用版本标识 (simple, v2, new等)
   └── 删除experiments_new/, .rework/等临时目录

6. 高内聚低耦合原则
   └── 模块内部高内聚: 相关功能放在一起
   └── 模块之间低耦合: 通过接口通信
   └── 依赖关系单向: 不循环依赖
   └── 配置与逻辑分离: 不硬编码参数
```

### 2.2 补充原则 (应该遵守)

```
7. 单一事实来源原则 (Single Source of Truth)
   └── 配置只在一处定义 (OmegaConf YAML)
   └── 常量只在constants.py定义
   └── 枚举只在enums.py定义
   └── 不在代码中硬编码 magic numbers

8. 显式优于隐式原则 (Explicit > Implicit)
   └── 不使用**kwargs传递参数
   └── 不使用动态属性访问 (getattr/setattr)
   └── 导入必须显式 (from module import Name)
   └── 类型注解必须完整 (mypy --strict)

9. 失败快速原则 (Fail Fast)
   └── 尽早验证输入 (前置条件检查)
   └── 使用异常而非返回错误码
   └── 错误信息必须包含上下文 (变量值、期望范围)
   └── 不变量检查 (assert for invariants)

10. 可测试性原则 (Testability)
    └── 所有业务逻辑不依赖全局状态
    └── 外部依赖通过参数注入 (依赖注入)
    └── 纯函数优先 (无副作用)
    └── 边界和核心分离 (IO vs Logic)

11. 渐进式复杂度原则 (Progressive Complexity)
    └── 基础功能简单 (80%场景易用)
    └── 高级功能通过组合实现 (20%场景可达)
    └── 不为未来可能的需求过度设计 (YAGNI)

12. 文档即代码原则 (Docs as Code)
    └── README与代码同步
    └── 架构文档在docs/目录
    └── 代码注释解释"为什么"而非"是什么"
    └── 函数文档说明契约 (前置/后置条件)

13. 一致性行为原则 (Consistent Behavior)
    └── 相同输入产生相同输出 (纯函数)
    └── 随机性必须显式控制 (seed参数)
    └── 默认值行为可预测

14. 最小暴露原则 (Minimal Exposure)
    └── 类/函数默认私有 (_前缀)，需要时才公开
    └── 模块__all__显式定义公开接口
    └── 不暴露内部实现细节

15. 资源管理原则 (Resource Management)
    └── 使用上下文管理器 (with语句)
    └── 及时释放大对象 (GPU内存、文件句柄)
    └── 流式处理大数据集

16. 错误边界原则 (Error Boundaries)
    └── 外部输入在边界验证
    └── 内部逻辑假设输入已验证
    └── 异常转换 (底层异常→领域异常)

17. 性能意识原则 (Performance Awareness)
    └── 大数据集使用生成器/迭代器
    └── 避免不必要的拷贝
    └── 向量化操作优于循环
    └── 但: 清晰优先于微优化

18. 领域语言原则 (Ubiquitous Language)
    └── 使用业务领域的命名 (SOH, MIM, MCAR)
    └── 避免缩写 (除非行业标准)
    └── 命名反映意图而非实现

19. 自文档化原则 (Self-Documenting)
    └── 函数名包含动作和对象 (load_dataset)
    └── 布尔参数用谓词命名 (use_mim, is_training)
    └── 类型即文档 (类型注解)

20. 可逆操作原则 (Reversible Operations)
    └── 保存/加载对称
    └── 编码/解码对称
    └── 变换/逆变换成对存在 (如果可能)
```

---

## 三、重构计划

### 阶段一: 清理与准备 (1-2天)

#### 3.1.1 删除冗余文件

```bash
# 删除临时/旧版本目录
rm -rf experiments_new/
rm -rf .rework/
rm -rf src/missing_data/fixed_feature_missing.py

# 删除重复工厂
rm src/models/factory_simple.py
rm src/models/model_factory.py

# 删除简化版文件
rm src/missing/missing_data_simple.py
rm src/experiments/simple_runner.py

# 删除未使用的分析脚本（如已过期）
# 保留: analyze_100seeds_results.py, visualize_100seeds_results.py
# 删除其他重复的分析脚本
```

#### 3.1.2 统一缺失数据处理

```
before:
  src/missing/
    └── imputers/
  src/missing_data/
    ├── mcar.py
    ├── mar.py
    ├── mnar.py
    └── imputation.py

after:
  src/missing/
    ├── generators/     # 缺失模式生成器
    │   ├── base.py
    │   ├── mcar.py
    │   ├── mar.py
    │   └── mnar.py
    └── imputers/       # 插补器
        ├── base.py
        ├── mean.py
        ├── knn.py
        ├── iterative.py
        └── zero.py
```

#### 3.1.3 统一数据加载

```
before:
  src/data/
    ├── xjtu_loader.py
    ├── loader.py
    ├── dataset_loader.py
    ├── datasets.py
    └── loader_dataloaders.py

after:
  src/data/
    ├── loader.py           # 统一入口
    ├── xjtu.py             # XJTU特定逻辑
    ├── transforms.py       # 特征变换
    └── splits.py           # 数据划分
```

### 阶段二: 核心架构重构 (3-4天)

#### 3.2.1 重构interfaces.py

```python
# before: 混合接口定义和具体实现
@dataclass
class BatteryDataset:
    ...
    def __post_init__(self):  # 具体实现
        ...

# after: 纯接口定义
@dataclass(frozen=True)
class BatteryDataset:
    features: np.ndarray
    labels: np.ndarray
    battery_ids: np.ndarray
    # 验证移到工厂函数

class IDataLoader(Protocol):
    @abstractmethod
    def load(self, batch_id: str) -> BatteryDataset: ...
```

#### 3.2.2 重构模型工厂

```python
# src/models/factory.py
# 合并3个工厂为1个，约150行

MODEL_CONFIGS: dict[str, dict] = {...}  # 配置提取到YAML

def create_model(
    model_type: ModelType,
    input_dim: int,
    **kwargs
) -> nn.Module: ...

def get_model_info(model: nn.Module) -> ModelInfo: ...
```

#### 3.2.3 重构训练器

```python
# src/trainers/
# 合并Lightning和基础训练器

# trainer.py
class Trainer(ABC):
    @abstractmethod
    def fit(self, model, train_data, val_data) -> TrainingResult: ...

# lightning_trainer.py
class LightningTrainer(Trainer):
    ...

# callback.py
class EarlyStopping(Callback): ...
```

#### 3.2.4 重构实验入口

```
before:
  experiments/run_experiment.py (898行)
  experiments/run_batch_experiments_v2.py (390行)

after:
  experiments/
    ├── train.py          # 训练入口 (分界线以上)
    ├── evaluate.py       # 评估入口 (分界线以下)
    └── batch.py          # 批量执行
  src/experiments/
    ├── runner.py         # 实验运行器
    └── scheduler.py      # 调度器
```

### 阶段三: 配置系统统一 (1-2天)

#### 3.3.1 统一配置

```
before:
  src/config/pydantic_config.py  (394行，dataclass)
  configs/*.yaml                 (Hydra配置)
  多处硬编码配置

after:
  src/config/
    ├── schema.py       # 配置结构 (dataclass)
    ├── loader.py       # 配置加载
    └── defaults.py     # 默认值
  configs/
    └── *.yaml          # 唯一配置源
```

### 阶段四: 测试重构 (1-2天)

#### 3.4.1 测试架构

```
before:
  tests/
    ├── test_critical_path.py
    ├── test_simplified.py
    ├── test_p1_mock.py
    └── ...

after:
  tests/
    ├── unit/           # 单元测试
    │   ├── test_models.py
    │   ├── test_data.py
    │   └── test_missing.py
    ├── integration/    # 集成测试
    │   └── test_pipeline.py
    └── conftest.py     # 共享fixture
```

### 阶段五: 文档更新 (1天)

#### 3.5.1 文档结构

```
docs/
├── ARCHITECTURE.md     # 架构设计
├── API.md              # API文档
├── CONTRIBUTING.md     # 贡献指南
└── EXAMPLES.md         # 使用示例

README.md               # 精简，指向docs/
```

---

## 四、目标结构

### 4.1 重构后目录结构

```
.
├── src/
│   ├── core/
│   │   ├── types.py          # 类型定义、常量
│   │   ├── interfaces.py     # 纯接口定义
│   │   └── registry.py       # 注册中心
│   ├── data/
│   │   ├── loader.py         # 统一数据加载
│   │   ├── xjtu.py           # XJTU数据集
│   │   ├── transforms.py     # 特征变换
│   │   └── splits.py         # 数据划分
│   ├── models/
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   ├── cnn.py
│   │   └── factory.py
│   ├── missing/
│   │   ├── generators/       # 缺失模式生成
│   │   │   ├── mcar.py
│   │   │   ├── mar.py
│   │   │   └── mnar.py
│   │   └── imputers/         # 插补方法
│   │       ├── mean.py
│   │       ├── knn.py
│   │       └── iterative.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── experiments/
│   │   ├── runner.py
│   │   └── scheduler.py
│   ├── config/
│   │   ├── schema.py
│   │   └── loader.py
│   └── utils/
│       ├── seed.py
│       └── logging.py
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── batch.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── configs/
│   └── *.yaml
├── docs/
└── README.md
```

### 4.2 代码量目标

| 指标 | 当前 | 目标 | 降幅 |
|------|------|------|------|
| src/ Python文件数 | 74 | 35 | -53% |
| src/ 总行数 | 15,486 | 5,000 | -68% |
| experiments/ 文件数 | 16 | 3 | -81% |
| 平均文件行数 | 209 | 143 | -32% |

---

## 五、实施建议

### 5.1 分支策略

```
refactor-code (当前)
  ├── refactor/cleanup       # 阶段一: 清理
  ├── refactor/core          # 阶段二: 核心架构
  ├── refactor/config        # 阶段三: 配置
  ├── refactor/tests         # 阶段四: 测试
  └── refactor/docs          # 阶段五: 文档
```

### 5.2 验证检查点

每个阶段完成后必须：
1. 所有现有测试通过
2. 新增单元测试覆盖率>80%
3. mypy --strict 无错误
4. 文档已更新
5. 代码审查完成

### 5.3 风险缓解

| 风险 | 缓解措施 |
|------|----------|
| 功能丢失 | 每个阶段保留可运行的检查点 |
| 性能退化 | 基准测试对比 |
| 回归错误 | 自动化测试覆盖 |
| 进度延迟 | 每个阶段有明确的完成标准 |

---

*诊断完成时间: 2026-04-10*
