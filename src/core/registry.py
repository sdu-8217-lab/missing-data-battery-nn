"""
注册中心 - 插件系统实现

提供通用的组件注册和发现机制，支持运行时动态加载。

使用示例:
    >>> from src.core.registry import IMPUTERS
    >>> 
    >>> # 注册组件
    >>> @IMPUTERS.register("mean")
    >>> class MeanImputer:
    ...     pass
    >>> 
    >>> # 创建组件实例
    >>> imputer = IMPUTERS.create("mean")
"""

from typing import Type, TypeVar, Generic, Callable, Optional
from functools import wraps
import importlib

T = TypeVar('T')


class Registry(Generic[T]):
    """
    通用注册中心
    
    支持：
    1. 通过装饰器注册组件
    2. 通过名称创建组件实例
    3. 列出所有可用组件
    4. 延迟加载（可选）
    """
    
    def __init__(self, name: str, base_class: Type = None):
        """
        初始化注册中心
        
        Args:
            name: 注册中心名称（用于错误信息）
            base_class: 可选的基类约束，注册组件必须继承此类
        """
        self.name = name
        self._registry: dict[str, Type[T]] = {}
        self._base_class = base_class
        self._lazy_modules: dict[str, str] = {}  # name -> module_path
    
    def register(
        self, 
        name: str, 
        override: bool = False
    ) -> Callable[[Type[T]], Type[T]]:
        """
        装饰器：注册组件
        
        Args:
            name: 组件名称标识
            override: 是否允许覆盖已存在的组件
            
        Returns:
            装饰器函数
            
        Example:
            >>> @registry.register("my_component")
            ... class MyComponent:
            ...     pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry and not override:
                raise KeyError(
                    f"{self.name} '{name}' already registered. "
                    f"Use override=True to replace."
                )
            
            # 如果指定了基类，验证继承关系
            if self._base_class is not None:
                if not issubclass(cls, self._base_class):
                    raise TypeError(
                        f"{cls.__name__} must inherit from "
                        f"{self._base_class.__name__}"
                    )
            
            self._registry[name] = cls
            return cls
        
        return decorator
    
    def register_lazy(self, name: str, module_path: str, class_name: str):
        """
        延迟注册：只记录模块路径，实际需要时才导入
        
        Args:
            name: 组件名称
            module_path: 模块路径（如 'src.missing.imputers.mean'）
            class_name: 类名
        """
        self._lazy_modules[name] = (module_path, class_name)
    
    def get(self, name: str) -> Type[T]:
        """
        获取组件类（不创建实例）
        
        Args:
            name: 组件名称
            
        Returns:
            组件类
            
        Raises:
            KeyError: 如果组件不存在
        """
        # 检查是否需要延迟加载
        if name not in self._registry and name in self._lazy_modules:
            self._load_lazy(name)
        
        if name not in self._registry:
            available = self.list_available()
            raise KeyError(
                f"Unknown {self.name}: '{name}'. "
                f"Available: {available}"
            )
        
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs) -> T:
        """
        创建组件实例
        
        Args:
            name: 组件名称
            *args, **kwargs: 传递给组件构造函数的参数
            
        Returns:
            组件实例
            
        Example:
            >>> imputer = registry.create("mean")
            >>> model = registry.create("mlp", input_dim=32)
        """
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def list_available(self) -> list[str]:
        """列出所有可用的组件名称"""
        # 加载所有延迟模块
        for name in list(self._lazy_modules.keys()):
            if name not in self._registry:
                try:
                    self._load_lazy(name)
                except ImportError:
                    pass  # 忽略无法加载的模块
        
        return sorted(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """检查组件是否已注册"""
        return name in self._registry or name in self._lazy_modules
    
    def unregister(self, name: str):
        """注销组件（主要用于测试）"""
        if name in self._registry:
            del self._registry[name]
        if name in self._lazy_modules:
            del self._lazy_modules[name]
    
    def _load_lazy(self, name: str):
        """加载延迟注册的模块"""
        if name not in self._lazy_modules:
            return
        
        module_path, class_name = self._lazy_modules[name]
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._registry[name] = cls
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to lazy-load {self.name} '{name}' "
                f"from {module_path}.{class_name}: {e}"
            )
    
    def __contains__(self, name: str) -> bool:
        """支持 'in' 操作符"""
        return self.is_registered(name)
    
    def __repr__(self) -> str:
        """字符串表示"""
        available = self.list_available()
        return f"{self.name}Registry(available={available})"


# ============================================================================
# 全局注册中心实例
# ============================================================================

# 插补策略注册中心
IMPUTERS = Registry["IImputer"]("imputer")

# 缺失数据生成器注册中心
MISSING_GENERATORS = Registry["IMissingGenerator"]("missing_generator")

# 模型注册中心
MODELS = Registry["IModel"]("model")

# 训练引擎注册中心
TRAINERS = Registry["ITrainer"]("trainer")

# 数据加载器注册中心
DATA_LOADERS = Registry["IDataLoader"]("data_loader")


# ============================================================================
# 便捷函数
# ============================================================================

def create_imputer(name: str, **kwargs) -> "IImputer":
    """便捷函数：创建插补器"""
    return IMPUTERS.create(name, **kwargs)


def create_missing_generator(name: str, **kwargs) -> "IMissingGenerator":
    """便捷函数：创建缺失数据生成器"""
    return MISSING_GENERATORS.create(name, **kwargs)


def create_model(name: str, **kwargs) -> "IModel":
    """便捷函数：创建模型"""
    return MODELS.create(name, **kwargs)


def create_trainer(name: str, **kwargs) -> "ITrainer":
    """便捷函数：创建训练器"""
    return TRAINERS.create(name, **kwargs)


def create_data_loader(name: str, **kwargs) -> "IDataLoader":
    """便捷函数：创建数据加载器"""
    return DATA_LOADERS.create(name, **kwargs)


def list_all_components() -> dict:
    """列出所有可用的组件"""
    return {
        "imputers": IMPUTERS.list_available(),
        "missing_generators": MISSING_GENERATORS.list_available(),
        "models": MODELS.list_available(),
        "trainers": TRAINERS.list_available(),
        "data_loaders": DATA_LOADERS.list_available(),
    }
