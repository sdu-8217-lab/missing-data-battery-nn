"""
统计检验工具 - 用于实验结果的显著性检验和效应量分析

提供:
- 配对t检验 (paired t-test): 比较配对样本的差异
- Wilcoxon符号秩检验: 非参数替代方法
- Cohen's d: 效应量计算
- 置信区间: 均值差异的置信区间
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import warnings

# 可选依赖 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    # 效应量
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None
    # 置信区间
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: Optional[float] = None
    # 样本信息
    n_samples: Optional[int] = None
    mean_diff: Optional[float] = None
    std_diff: Optional[float] = None
    
    def __str__(self) -> str:
        lines = [
            f"{self.test_name}:",
            f"  统计量: {self.statistic:.4f}",
            f"  p值: {self.p_value:.2e}",
            f"  显著性 ({self.alpha}): {'✅ 是' if self.significant else '❌ 否'}",
        ]
        if self.effect_size is not None:
            lines.append(f"  效应量 (Cohen's d): {self.effect_size:.4f} ({self.effect_interpretation})")
        if self.ci_lower is not None:
            lines.append(f"  置信区间 ({self.ci_level*100:.0f}%): [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        if self.mean_diff is not None:
            lines.append(f"  均值差异: {self.mean_diff:.4f} ± {self.std_diff:.4f}")
        return "\n".join(lines)


def cohens_d(x1: np.ndarray, x2: np.ndarray, paired: bool = False) -> float:
    """
    计算 Cohen's d 效应量
    
    Cohen's d 表示两组均值差异的标准化大小:
    - Small effect:   d ≈ 0.2
    - Medium effect:  d ≈ 0.5
    - Large effect:   d ≈ 0.8
    
    Args:
        x1: 第一组样本
        x2: 第二组样本
        paired: 是否为配对样本
        
    Returns:
        Cohen's d 值
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    if paired:
        # 配对样本: 使用差值的标准差
        diff = x1 - x2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        d = mean_diff / std_diff if std_diff > 0 else 0
    else:
        # 独立样本: 使用合并标准差
        n1, n2 = len(x1), len(x2)
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return d


def interpret_cohens_d(d: float) -> str:
    """
    解释 Cohen's d 的大小
    
    Args:
        d: Cohen's d 值
        
    Returns:
        解释字符串
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "可忽略"
    elif abs_d < 0.5:
        return "小"
    elif abs_d < 0.8:
        return "中"
    else:
        return "大"


def compute_confidence_interval(
    x1: np.ndarray,
    x2: np.ndarray,
    confidence: float = 0.95,
    paired: bool = False
) -> Tuple[float, float]:
    """
    计算均值差异的置信区间
    
    Args:
        x1: 第一组样本
        x2: 第二组样本
        confidence: 置信水平 (默认0.95)
        paired: 是否为配对样本
        
    Returns:
        (下限, 上限) 元组
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    if paired:
        diff = x1 - x2
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff) if HAS_SCIPY else np.std(diff, ddof=1) / np.sqrt(len(diff))
        df = len(diff) - 1
    else:
        n1, n2 = len(x1), len(x2)
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        
        mean_diff = mean1 - mean2
        se_diff = np.sqrt(var1/n1 + var2/n2)
        # Welch-Satterthwaite 自由度
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    if HAS_SCIPY:
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
    else:
        # 使用正态分布近似
        from math import erf, sqrt
        t_crit = 1.96  # 95% CI 近似
    
    margin = t_crit * se_diff
    return (mean_diff - margin, mean_diff + margin)


def paired_t_test(
    x_before: np.ndarray,
    x_after: np.ndarray,
    alpha: float = 0.05,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
) -> StatisticalTestResult:
    """
    配对t检验
    
    用于比较同一组样本在两种条件下的差异（如：使用MIM前后）
    
    Args:
        x_before: 前测样本
        x_after: 后测样本
        alpha: 显著性水平
        alternative: 备择假设类型
        
    Returns:
        StatisticalTestResult
        
    Example:
        >>> mae_no_mim = [0.5, 0.6, 0.4, ...]  # 100 seeds
        >>> mae_mim = [0.4, 0.5, 0.35, ...]
        >>> result = paired_t_test(mae_no_mim, mae_mim)
        >>> print(result.significant)  # True if MIM significantly improves
    """
    if not HAS_SCIPY:
        raise ImportError("需要 scipy 库进行统计检验。安装: pip install scipy")
    
    x_before = np.asarray(x_before)
    x_after = np.asarray(x_after)
    
    if len(x_before) != len(x_after):
        raise ValueError("配对样本必须有相同长度")
    
    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(x_after, x_before, alternative=alternative)
    
    # 计算效应量
    d = cohens_d(x_after, x_before, paired=True)
    
    # 计算置信区间
    ci_lower, ci_upper = compute_confidence_interval(x_after, x_before, paired=True)
    
    # 确定显著性
    significant = p_value < alpha
    
    return StatisticalTestResult(
        test_name="配对t检验",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=d,
        effect_interpretation=interpret_cohens_d(d),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=0.95,
        n_samples=len(x_before),
        mean_diff=np.mean(x_after - x_before),
        std_diff=np.std(x_after - x_before, ddof=1)
    )


def wilcoxon_test(
    x_before: np.ndarray,
    x_after: np.ndarray,
    alpha: float = 0.05,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
) -> StatisticalTestResult:
    """
    Wilcoxon 符号秩检验 (配对样本的非参数检验)
    
    当数据不满足正态分布假设时使用
    
    Args:
        x_before: 前测样本
        x_after: 后测样本
        alpha: 显著性水平
        alternative: 备择假设类型
        
    Returns:
        StatisticalTestResult
    """
    if not HAS_SCIPY:
        raise ImportError("需要 scipy 库进行统计检验。安装: pip install scipy")
    
    x_before = np.asarray(x_before)
    x_after = np.asarray(x_after)
    
    # 执行 Wilcoxon 检验
    statistic, p_value = stats.wilcoxon(x_after, x_before, alternative=alternative)
    
    # Wilcoxon 检验没有直接的效应量，使用秩相关系数作为近似
    # 计算中位数差异作为效应描述
    median_diff = np.median(x_after - x_before)
    
    significant = p_value < alpha
    
    return StatisticalTestResult(
        test_name="Wilcoxon符号秩检验",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=None,  # Wilcoxon 没有标准效应量
        effect_interpretation=None,
        ci_lower=None,
        ci_upper=None,
        n_samples=len(x_before),
        mean_diff=median_diff,  # 使用中位数差异
        std_diff=None
    )


def independent_t_test(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
) -> StatisticalTestResult:
    """
    独立样本t检验 (Welch's t-test)
    
    用于比较两组独立样本的均值差异
    
    Args:
        x1: 第一组样本
        x2: 第二组样本
        alpha: 显著性水平
        equal_var: 是否假设等方差
        alternative: 备择假设类型
        
    Returns:
        StatisticalTestResult
    """
    if not HAS_SCIPY:
        raise ImportError("需要 scipy 库进行统计检验。安装: pip install scipy")
    
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # 执行独立样本t检验
    t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    
    # 计算效应量
    d = cohens_d(x1, x2, paired=False)
    
    # 计算置信区间
    ci_lower, ci_upper = compute_confidence_interval(x1, x2, paired=False)
    
    significant = p_value < alpha
    
    return StatisticalTestResult(
        test_name="独立样本t检验 (Welch's)" if not equal_var else "独立样本t检验",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=d,
        effect_interpretation=interpret_cohens_d(d),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=0.95,
        n_samples=len(x1) + len(x2),
        mean_diff=np.mean(x1) - np.mean(x2),
        std_diff=None
    )


# ============================================================================
# 批量统计检验 (用于实验结果分析)
# ============================================================================

@dataclass
class ComparisonGroup:
    """比较组定义"""
    name: str
    group_values: np.ndarray
    description: str = ""


def compare_mim_effect(
    df,
    groupby_cols: List[str] = None,
    metric: str = 'test_mae',
    test_type: Literal['paired_t', 'wilcoxon'] = 'paired_t',
    alpha: float = 0.05
) -> List[Dict]:
    """
    批量比较 MIM 效果
    
    对数据框中的每个分组进行 MIM vs non-MIM 的统计检验
    
    Args:
        df: 包含实验结果的数据框
        groupby_cols: 分组列 (如 ['model', 'test_mr'])
        metric: 要比较的指标列
        test_type: 检验类型
        alpha: 显著性水平
        
    Returns:
        检验结果列表
        
    Example:
        >>> results = compare_mim_effect(
        ...     df,
        ...     groupby_cols=['model', 'test_mr'],
        ...     metric='test_mae'
        ... )
    """
    if not HAS_SCIPY:
        raise ImportError("需要 scipy 库。安装: pip install scipy")
    
    if groupby_cols is None:
        groupby_cols = []
    
    results = []
    
    # 按指定列分组
    grouped = df.groupby(groupby_cols) if groupby_cols else [(None, df)]
    
    for group_key, group_df in grouped:
        # 分离 MIM 和 non-MIM 数据
        no_mim_data = group_df[group_df['use_mim'] == False][metric].values
        mim_data = group_df[group_df['use_mim'] == True][metric].values
        
        if len(no_mim_data) < 2 or len(mim_data) < 2:
            continue
        
        # 对齐样本 (按 seed 匹配)
        if 'seed' in group_df.columns:
            pivot_df = group_df.pivot_table(
                index='seed',
                columns='use_mim',
                values=metric
            ).dropna()
            
            if len(pivot_df) < 2:
                continue
                
            no_mim_values = pivot_df[False].values
            mim_values = pivot_df[True].values
        else:
            # 如果没有 seed 列，假设数据已经配对
            min_len = min(len(no_mim_data), len(mim_data))
            no_mim_values = no_mim_data[:min_len]
            mim_values = mim_data[:min_len]
        
        # 执行检验
        if test_type == 'paired_t':
            result = paired_t_test(no_mim_values, mim_values, alpha=alpha)
        elif test_type == 'wilcoxon':
            result = wilcoxon_test(no_mim_values, mim_values, alpha=alpha)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # 构建结果字典
        result_dict = {
            'group': group_key,
            'n_samples': result.n_samples,
            'no_mim_mean': np.mean(no_mim_values),
            'mim_mean': np.mean(mim_values),
            'improvement_percent': (np.mean(no_mim_values) - np.mean(mim_values)) / np.mean(no_mim_values) * 100,
            'test': result.test_name,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'significant': result.significant,
            'effect_size': result.effect_size,
            'effect_interpretation': result.effect_interpretation,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
        }
        
        results.append(result_dict)
    
    return results


def summarize_significance(results: List[Dict]) -> Dict:
    """
    汇总统计显著性结果
    
    Args:
        results: compare_mim_effect 返回的结果列表
        
    Returns:
        汇总统计字典
    """
    if not results:
        return {}
    
    total = len(results)
    significant = sum(1 for r in results if r['significant'])
    
    # 按效应量分类
    effect_sizes = {
        'negligible': 0,
        'small': 0,
        'medium': 0,
        'large': 0
    }
    
    for r in results:
        interpretation = r.get('effect_interpretation', '')
        if interpretation == '可忽略':
            effect_sizes['negligible'] += 1
        elif interpretation == '小':
            effect_sizes['small'] += 1
        elif interpretation == '中':
            effect_sizes['medium'] += 1
        elif interpretation == '大':
            effect_sizes['large'] += 1
    
    # 平均改进
    avg_improvement = np.mean([r['improvement_percent'] for r in results])
    
    return {
        'total_tests': total,
        'significant_tests': significant,
        'significant_rate': significant / total if total > 0 else 0,
        'effect_sizes': effect_sizes,
        'avg_improvement_percent': avg_improvement,
    }
