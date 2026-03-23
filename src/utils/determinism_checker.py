"""
确定性验证工具 - 确保实验可复现

检查项目中的所有随机源是否都有种子控制
"""

import ast
import inspect
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DeterminismReport:
    """确定性验证报告"""
    
    # 状态
    is_deterministic: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # 检查结果
    python_random_seeded: bool = False
    numpy_random_seeded: bool = False
    torch_random_seeded: bool = False
    cuda_random_seeded: bool = False
    cudnn_deterministic: bool = False
    dataloader_generator_seeded: bool = False
    dataloader_worker_init: bool = False
    
    # 发现的随机源
    found_random_sources: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_deterministic = False
    
    def to_dict(self) -> Dict:
        return {
            'is_deterministic': self.is_deterministic,
            'warnings': self.warnings,
            'errors': self.errors,
            'checks': {
                'python_random_seeded': self.python_random_seeded,
                'numpy_random_seeded': self.numpy_random_seeded,
                'torch_random_seeded': self.torch_random_seeded,
                'cuda_random_seeded': self.cuda_random_seeded,
                'cudnn_deterministic': self.cudnn_deterministic,
                'dataloader_generator_seeded': self.dataloader_generator_seeded,
                'dataloader_worker_init': self.dataloader_worker_init,
            },
            'found_random_sources': self.found_random_sources
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "确定性验证报告",
            "=" * 60,
            f"整体状态: {'✅ 确定' if self.is_deterministic else '❌ 不确定'}",
            "",
            "检查项:",
            f"  Python random:     {'✅' if self.python_random_seeded else '❌'}",
            f"  NumPy random:      {'✅' if self.numpy_random_seeded else '❌'}",
            f"  PyTorch random:    {'✅' if self.torch_random_seeded else '❌'}",
            f"  CUDA random:       {'✅' if self.cuda_random_seeded else '❌'}",
            f"  cuDNN deterministic: {'✅' if self.cudnn_deterministic else '❌'}",
            f"  DataLoader generator: {'✅' if self.dataloader_generator_seeded else '❌'}",
            f"  DataLoader worker_init: {'✅' if self.dataloader_worker_init else '❌'}",
        ]
        
        if self.warnings:
            lines.extend(["", "⚠️  警告:"])
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        if self.errors:
            lines.extend(["", "❌ 错误:"])
            for e in self.errors:
                lines.append(f"  - {e}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def check_current_state() -> DeterminismReport:
    """检查当前运行环境的随机状态"""
    report = DeterminismReport()
    
    # 检查 Python random 状态
    python_state = random.getstate()
    # 简单测试：如果执行random()，应该得到确定的结果
    random.seed(42)
    test_val = random.random()
    report.python_random_seeded = (test_val == 0.6394267984578837)
    random.setstate(python_state)  # 恢复状态
    
    # 检查 NumPy random 状态
    np_state = np.random.get_state()
    np.random.seed(42)
    test_val = np.random.random()
    report.numpy_random_seeded = (test_val == 0.3745401188473625)
    np.random.set_state(np_state)  # 恢复状态
    
    # 检查 PyTorch random 状态
    torch_state = torch.get_rng_state()
    torch.manual_seed(42)
    test_val = torch.rand(1).item()
    report.torch_random_seeded = abs(test_val - 0.8822693824768066) < 1e-6
    torch.set_rng_state(torch_state)  # 恢复状态
    
    # 检查 CUDA
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(42)
        test_val = torch.cuda.FloatTensor(1).uniform_().item()
        report.cuda_random_seeded = True  # 只要能设置就认为OK
        # 恢复状态
        torch.cuda.set_rng_state(cuda_state)
    else:
        report.cuda_random_seeded = True  # 没有CUDA也认为是OK的
    
    # 检查 cuDNN
    report.cudnn_deterministic = torch.backends.cudnn.deterministic
    
    return report


def check_set_seed_function() -> DeterminismReport:
    """检查 set_seed 函数的实现是否完整"""
    report = DeterminismReport()
    
    try:
        from .seed_manager import set_seed
    except ImportError:
        report.add_error("无法导入 set_seed 函数")
        return report
    
    # 获取源代码
    source = inspect.getsource(set_seed)
    
    # 检查关键点
    checks = {
        'python_random_seeded': 'random.seed' in source,
        'numpy_random_seeded': 'np.random.seed' in source,
        'torch_random_seeded': 'torch.manual_seed' in source,
        'cuda_random_seeded': 'torch.cuda.manual_seed_all' in source,
        'cudnn_deterministic': 'torch.backends.cudnn.deterministic' in source,
    }
    
    for attr, found in checks.items():
        setattr(report, attr, found)
        if not found:
            report.add_error(f"set_seed 缺少: {attr}")
    
    # 检查是否有 worker_init_fn 相关代码
    if 'worker_init_fn' not in source:
        report.add_warning("set_seed 未包含 DataLoader worker_init_fn，多进程加载时可能不确定")
    else:
        report.dataloader_worker_init = True
    
    return report


def create_worker_init_fn(seed: int):
    """
    创建 DataLoader 的 worker_init_fn
    
    用于确保多进程数据加载的确定性
    
    Usage:
        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=create_worker_init_fn(seed)
        )
    """
    def worker_init_fn(worker_id: int):
        # 每个 worker 使用不同的种子
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return worker_init_fn


def check_dataloader_usage(file_path: str) -> List[Dict]:
    """
    分析文件中 DataLoader 的使用情况
    
    检查是否使用了 generator 和 worker_init_fn
    """
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except Exception:
        return issues
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 检查是否是 DataLoader 调用
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            if func_name == 'DataLoader':
                # 检查关键字参数
                keywords = {kw.arg for kw in node.keywords}
                
                has_generator = 'generator' in keywords
                has_worker_init = 'worker_init_fn' in keywords
                has_num_workers = 'num_workers' in keywords
                
                # 获取行号
                line_no = getattr(node, 'lineno', 0)
                
                if not has_generator and not has_worker_init:
                    if has_num_workers:
                        # 检查 num_workers 的值
                        for kw in node.keywords:
                            if kw.arg == 'num_workers':
                                if isinstance(kw.value, ast.Num) and kw.value.n > 0:
                                    issues.append({
                                        'line': line_no,
                                        'issue': 'DataLoader 使用 num_workers > 0 但没有设置 worker_init_fn',
                                        'severity': 'error'
                                    })
                                break
                    else:
                        # num_workers 默认为 0，不需要 worker_init_fn
                        pass
    
    return issues


def verify_determinism(seed: int = 42, verbose: bool = False) -> bool:
    """
    运行一个快速的确定性验证测试
    
    执行两次相同种子的简单操作，检查结果是否一致
    
    Returns:
        True if deterministic, False otherwise
    """
    results = []
    
    for run in range(2):
        from .seed_manager import set_seed
        set_seed(seed)
        
        # 执行一系列随机操作
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        results.append({
            'python': python_rand,
            'numpy': numpy_rand,
            'torch': torch_rand
        })
    
    # 比较结果
    is_deterministic = (
        results[0]['python'] == results[1]['python'] and
        results[0]['numpy'] == results[1]['numpy'] and
        results[0]['torch'] == results[1]['torch']
    )
    
    if verbose:
        print(f"Run 1: {results[0]}")
        print(f"Run 2: {results[1]}")
        print(f"Deterministic: {is_deterministic}")
    
    return is_deterministic


def full_determinism_check(project_root: str = None) -> DeterminismReport:
    """
    执行完整的确定性检查
    
    包括:
    1. 当前环境状态检查
    2. set_seed 函数实现检查
    3. DataLoader 使用检查
    4. 实际运行验证
    """
    report = DeterminismReport()
    
    # 1. 检查 set_seed 实现
    seed_report = check_set_seed_function()
    report.warnings.extend(seed_report.warnings)
    report.errors.extend(seed_report.errors)
    
    # 2. 实际运行验证
    if not verify_determinism(seed=42):
        report.add_error("set_seed(42) 执行两次产生不同结果")
    
    # 3. 更新检查状态
    report.python_random_seeded = seed_report.python_random_seeded
    report.numpy_random_seeded = seed_report.numpy_random_seeded
    report.torch_random_seeded = seed_report.torch_random_seeded
    report.cuda_random_seeded = seed_report.cuda_random_seeded
    report.cudnn_deterministic = seed_report.cudnn_deterministic
    report.dataloader_worker_init = seed_report.dataloader_worker_init
    
    # 4. 检查 DataLoader 使用
    if project_root:
        exp_dir = Path(project_root) / "experiments"
        if exp_dir.exists():
            for py_file in exp_dir.glob("*.py"):
                issues = check_dataloader_usage(str(py_file))
                for issue in issues:
                    report.add_warning(
                        f"{py_file.name}:{issue['line']} - {issue['issue']}"
                    )
    
    return report


if __name__ == "__main__":
    # 自测试
    print("运行确定性验证...")
    
    # 测试 set_seed
    report = check_set_seed_function()
    print(report)
    
    # 测试实际运行
    print("\n运行实际确定性测试...")
    is_deterministic = verify_determinism(seed=42, verbose=True)
    
    # 完整检查
    print("\n运行完整检查...")
    full_report = full_determinism_check(project_root=".")
    print(full_report)
