# 重构原则速查表

## 20条必须遵守的原则

### 1. 必要性原则
> 每个文件、函数、类必须有明确且唯一的职责

**检查点**:
- [ ] 删除未使用的代码
- [ ] 删除可通过简单组合实现的包装代码
- [ ] 删除重复功能

```python
# ❌ 违反 - 包装无意义
def load_xjtu_data(batch):
    return XJTULoader().load(batch)

# ✅ 正确 - 直接使用
loader = XJTULoader()
X, y = loader.load(batch)
```

---

### 2. 有效性原则
> 所有代码必须通过类型检查，所有公共API必须有文档

**检查点**:
- [ ] `mypy --strict` 无错误
- [ ] 所有公共函数有 docstring
- [ ] 类型注解完整

```python
# ❌ 违反
def process(data):
    return data.mean()

# ✅ 正确
import numpy as np
from numpy.typing import NDArray

def process(data: NDArray[np.float32]) -> float:
    """计算数据均值.
    
    Args:
        data: 输入数据，形状为 [N, D]
        
    Returns:
        均值标量
        
    Raises:
        ValueError: 数据为空
    """
    if len(data) == 0:
        raise ValueError("Empty data")
    return float(data.mean())
```

---

### 3. 简洁性原则
> 限制规模，保持可读性

**硬性限制**:
| 项目 | 限制 | 说明 |
|------|------|------|
| 文件 | <300行 | 超过则拆分 |
| 函数 | <50行 | 超过则提取 |
| 类 | <200行 | 超过则拆分 |
| 函数参数 | <5个 | 超过用 dataclass |
| 缩进层级 | <4层 | 避免深层嵌套 |

```python
# ❌ 违反 - 参数过多
def train(model, X, y, epochs, lr, batch_size, patience, device, seed):
    ...

# ✅ 正确 - 使用 dataclass
@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 15
    device: str = "auto"
    seed: int = 42

def train(model: Model, data: Data, config: TrainConfig) -> Result:
    ...
```

---

### 4. 最优实现原则
> 使用最佳实践，避免低效代码

**优先级**:
1. 标准库 > 第三方库
2. 生成器 > 列表 (大数据集)
3. 向量化 > 循环
4. 内置函数 > 手写实现

```python
# ❌ 违反 - 低效循环
result = []
for i in range(len(x)):
    result.append(x[i] * 2)

# ✅ 正确 - 向量化
result = x * 2

# ❌ 违反 - 内存浪费
lines = [line.strip() for line in open('file.txt')]

# ✅ 正确 - 生成器
def lines_from_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()
```

---

### 5. 版本管理原则
> Git管理一切，不留冗余

**禁止**:
- ❌ 旧版本文件 (`*_old.py`, `*_v2.py`, `*_simple.py`)
- ❌ 备份目录 (`.rework/`, `backup/`)
- ❌ 临时目录 (`experiments_new/`, `temp/`)
- ❌ 版本号在文件名中

**必须**:
- [ ] 所有变更通过 git commit
- [ ] 有意义的 commit message
- [ ] 定期 push 到远程

---

### 6. 高内聚低耦合原则
> 模块内部高内聚，模块之间低耦合

**检查点**:
- [ ] 单一职责 (SRP)
- [ ] 依赖单向 (无循环依赖)
- [ ] 通过接口通信
- [ ] 配置与逻辑分离

```python
# ❌ 违反 - 高耦合
def train_model(args):
    # 直接依赖全局配置
    config = load_global_config()
    # 混合数据加载、模型创建、训练逻辑
    ...

# ✅ 正确 - 低耦合
def train_model(
    model: Model,
    data: Data,
    config: TrainConfig,
    logger: Logger | None = None
) -> TrainingResult:
    # 依赖通过参数注入
    ...
```

---

### 7. 单一事实来源原则
> 配置、常量只在一处定义

```python
# ❌ 违反 - 多处定义
# config.yaml
epochs: 200

# train.py
def train(epochs=100):  # 不一致!
    ...

# ✅ 正确 - 统一配置
# config/schema.py
@dataclass
class TrainingConfig:
    epochs: int = 200  # 唯一定义

# 所有地方使用此配置
```

---

### 8. 显式优于隐式原则
> 避免魔法，代码意图清晰

**禁止**:
- ❌ `**kwargs` 传递参数
- ❌ 动态属性访问 (`getattr`, `setattr`)
- ❌ 隐式类型转换

```python
# ❌ 违反
def process(**kwargs):
    data = kwargs.get('data')
    rate = kwargs.get('rate', 0.5)
    ...

# ✅ 正确
def process(data: Data, rate: float = 0.5) -> Result:
    ...
```

---

### 9. 失败快速原则
> 尽早发现问题，快速失败

```python
# ❌ 违反 - 延迟失败
def divide(a, b):
    return a / b  # 运行时才发现b=0

# ✅ 正确 - 立即验证
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError(f"Cannot divide by zero: a={a}, b={b}")
    return a / b
```

---

### 10. 可测试性原则
> 设计时就考虑测试

**要求**:
- [ ] 不依赖全局状态
- [ ] 外部依赖可注入
- [ ] 纯函数优先
- [ ] 边界和核心分离

```python
# ❌ 违反 - 难测试
class Trainer:
    def __init__(self):
        self.db = Database()  # 硬编码依赖
        self.logger = Logger()
    
    def train(self, data):
        self.db.save(data)  # 副作用

# ✅ 正确 - 易测试
class Trainer:
    def __init__(
        self,
        db: DatabaseInterface,
        logger: LoggerInterface
    ):
        self.db = db
        self.logger = logger
    
    def train(self, data: Data) -> Model:
        # 纯逻辑，无副作用
        return train_model(data)
```

---

### 11. 渐进式复杂度原则
> 简单开始，按需扩展

```python
# ❌ 违反 - 过度设计
class ExperimentRunner(ABC):
    @abstractmethod
    def setup(self): ...
    @abstractmethod
    def run(self): ...
    @abstractmethod
    def teardown(self): ...
    # ... 大量抽象方法

# ✅ 正确 - 简单开始
def run_experiment(config: Config) -> Result:
    """基础版本，满足80%需求."""
    ...

# 高级需求通过组合实现
class ParallelExperimentRunner:
    """并行版本，按需使用."""
    def __init__(self, runner: Callable[[Config], Result]):
        self.runner = runner
```

---

### 12. 文档即代码原则
> 文档是代码的一部分，同步维护

**要求**:
- [ ] README 与代码同步
- [ ] 架构文档在 `docs/`
- [ ] 注释解释"为什么"而非"是什么"

```python
# ❌ 违反 - 注释无用
# Increment counter
counter += 1

# ✅ 正确 - 解释原因
# 计数器需要原子递增，因为多线程环境下
# 可能出现竞态条件
counter += 1
```

---

### 13. 一致性行为原则
> 相同输入产生相同输出

```python
# ❌ 违反 - 行为不一致
import random

def augment(data):
    return data * random.random()  # 随机行为

# ✅ 正确 - 显式控制随机性
import numpy as np

def augment(data: NDArray, rng: np.random.Generator) -> NDArray:
    """数据增强.
    
    Args:
        data: 输入数据
        rng: 随机数生成器，控制随机性
    """
    return data * rng.random()
```

---

### 14. 最小暴露原则
> 只暴露必要的接口

```python
# ❌ 违反 - 全部公开
class DataLoader:
    def __init__(self):
        self.internal_state = {}  # 应该私有
    
    def _helper(self):  # 应该私有
        ...

# ✅ 正确 - 最小暴露
class DataLoader:
    def __init__(self):
        self._internal_state: dict = {}
    
    def _helper(self) -> None:  # 私有方法
        ...
    
    def load(self, batch_id: str) -> Data:  # 公开接口
        ...

# __all__ 定义公开接口
__all__ = ["DataLoader"]
```

---

### 15. 资源管理原则
> 及时释放资源

```python
# ❌ 违反 - 资源泄漏
f = open('file.txt')
data = f.read()
# 忘记关闭

# ✅ 正确 - 上下文管理器
with open('file.txt') as f:
    data = f.read()

# ✅ 正确 - 大数据集流式处理
def load_large_dataset(path: Path) -> Generator[Sample, None, None]:
    with open(path) as f:
        for line in f:
            yield Sample.from_line(line)
```

---

### 16. 错误边界原则
> 边界验证，内部假设

```python
# ❌ 违反 - 处处验证
def process(data: Data) -> Result:
    if data is None:  # 内部逻辑，应该信任调用者
        raise ValueError()
    ...

def analyze(result: Result) -> Metrics:
    if result is None:  # 重复验证
        raise ValueError()
    ...

# ✅ 正确 - 边界验证
@validate_arguments  # 边界验证
def pipeline(config: Config) -> Metrics:
    data = load(config.data_path)  # 内部信任
    result = process(data)
    return analyze(result)
```

---

### 17. 性能意识原则
> 性能重要，但清晰优先

```python
# ❌ 违反 - 过早优化，牺牲可读性
# 复杂的手动向量化，难以维护

# ✅ 正确 - 清晰优先，必要时优化
def compute_metrics(y_true, y_pred):
    # 清晰实现
    errors = y_true - y_pred
    return {
        'mae': np.abs(errors).mean(),
        'rmse': np.sqrt((errors ** 2).mean()),
    }

# 如果性能瓶颈，添加优化版本
@lru_cache(maxsize=128)
def compute_metrics_cached(y_true_tuple, y_pred_tuple):
    # 缓存版本
    ...
```

---

### 18. 领域语言原则
> 使用业务术语

```python
# ❌ 违反 - 技术术语
class BatteryDataProcessor:
    def extract_x(self, df):
        ...
    def get_y(self, df):
        ...

# ✅ 正确 - 领域术语
class BatteryDataset:
    """电池数据集."""
    
    def extract_features(self, df: DataFrame) -> Features:
        """提取特征."""
        ...
    
    def extract_soh(self, df: DataFrame) -> SoH:
        """提取SOH标签."""
        ...
```

---

### 19. 自文档化原则
> 命名即文档

```python
# ❌ 违反 - 命名不清
def calc(d, f):
    return np.mean(d[f])

# ✅ 正确 - 自文档化
def compute_feature_mean(
    data: BatteryDataset,
    feature_name: str
) -> float:
    """计算特征均值."""
    return float(np.mean(data[feature_name]))

# 布尔参数用谓词
if use_mim:  # 好
if mim_enabled:  # 更好
```

---

### 20. 可逆操作原则
> 对称的API设计

```python
# ✅ 正确 - 保存/加载对称
checkpoint.save(model, path)
model = checkpoint.load(path)

# ✅ 正确 - 变换/逆变换
transform = StandardScaler()
X_scaled = transform.fit_transform(X)
X_original = transform.inverse_transform(X_scaled)

# ❌ 违反 - 不对称
model.save(path)  # 实例方法
model = Model.load(path)  # 类方法？
```

---

## 每日检查清单

提交代码前检查:

- [ ] 通过了 `mypy --strict`
- [ ] 通过了所有测试 `pytest tests/`
- [ ] 覆盖率 > 80%
- [ ] 没有违反20条原则
- [ ] 文档已同步更新
- [ ] 提交了清晰的 commit message

## Commit Message 规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型**:
- `feat`: 新功能
- `fix`: Bug修复
- `refactor`: 重构
- `test`: 测试
- `docs`: 文档
- `chore`: 构建/工具

**示例**:
```
refactor(data): 统一数据加载接口

- 合并3个数据加载器为1个
- 使用Protocol定义抽象接口
- 添加完整类型注解

减少300行代码，提升可维护性
```
