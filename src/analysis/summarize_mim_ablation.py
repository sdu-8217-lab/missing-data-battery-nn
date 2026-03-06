"""
MIM Ablation Study - Results Summary and Analysis

功能:
1. 读取所有实验CSV文件
2. 计算每个 (方法, MR) 组合的统计指标 (mean ± std)
3. 打印对照表到终端
4. 保存Markdown格式汇总报告

使用方法:
    python src/analysis/summarize_mim_ablation.py

输出:
    - 终端表格
    - results/summary/mim_ablation_cnn1d_mar.md
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# 配置: CSV文件映射 (根据实际文件名修改此处)
# ============================================================================

# 实验配置: (方法显示名, MR, CSV文件名)
EXPERIMENT_CONFIGS = [
    # Baseline
    ("Baseline", "0.3", "mar_0.3_cnn1d_baseline.csv"),
    ("Baseline", "0.6", "mar_0.6_cnn1d_baseline.csv"),
    ("Baseline", "0.9", "mar_0.9_cnn1d_baseline.csv"),
    
    # MIM-Single (旧版)
    ("MIM-Single", "0.3", "mar_0.3_cnn1d_mim.csv"),
    ("MIM-Single", "0.6", "mar_0.6_cnn1d_mim.csv"),
    ("MIM-Single", "0.9", "mar_0.9_cnn1d_mim.csv"),
    
    # MIM-Multi (新版 - 多缺失率训练)
    ("MIM-Multi", "0.3", "mar_0.3_cnn1d_mim_corrected.csv"),
    ("MIM-Multi", "0.6", "mar_0.6_cnn1d_mim_corrected.csv"),
    ("MIM-Multi", "0.9", "mar_0.9_cnn1d_mim_corrected.csv"),
]

# 指标列表
METRICS = ["test_mae", "test_rmse", "test_r2"]


def load_and_compute_stats(csv_dir: Path, filename: str) -> Dict[str, Tuple[float, float]]:
    """
    加载CSV并计算统计指标
    
    返回: {metric: (mean, std), ...}
    """
    filepath = csv_dir / filename
    
    if not filepath.exists():
        print(f"[WARNING] File not found: {filepath}")
        return {m: (np.nan, np.nan) for m in METRICS}
    
    try:
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            print(f"[WARNING] Empty file: {filepath}")
            return {m: (np.nan, np.nan) for m in METRICS}
        
        stats = {}
        for metric in METRICS:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                stats[metric] = (mean_val, std_val)
            else:
                print(f"[WARNING] Column {metric} not found in {filepath}")
                stats[metric] = (np.nan, np.nan)
        
        return stats
        
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return {m: (np.nan, np.nan) for m in METRICS}


def format_value(mean: float, std: float) -> str:
    """格式化 mean ± std"""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


def compute_improvement(baseline_mean: float, method_mean: float) -> str:
    """计算相对改善百分比"""
    if np.isnan(baseline_mean) or np.isnan(method_mean) or baseline_mean == 0:
        return "N/A"
    
    # 对于MAE和RMSE，越小越好，改善 = (base - method) / base * 100%
    # 对于R2，越大越好，改善 = (method - base) / |base| * 100%
    improvement = (baseline_mean - method_mean) / baseline_mean * 100
    return f"{improvement:+.1f}%"


def print_summary_table(results: List[Dict]):
    """打印汇总表到终端"""
    print("\n" + "=" * 100)
    print("MIM Ablation Study - Summary Results (CNN1D + MAR)")
    print("=" * 100)
    
    # 表头
    print(f"\n{'Method':<12} {'MR':<6} {'MAE (mean+std)':<20} {'RMSE (mean+std)':<20} {'R2 (mean+std)':<20}")
    print("-" * 100)
    
    # 按MR分组打印
    for mr in ["0.3", "0.6", "0.9"]:
        print(f"\n[Missing Rate = {mr}]")
        for r in results:
            if r["mr"] == mr:
                mae_str = format_value(r["stats"]["test_mae"][0], r["stats"]["test_mae"][1])
                rmse_str = format_value(r["stats"]["test_rmse"][0], r["stats"]["test_rmse"][1])
                r2_str = format_value(r["stats"]["test_r2"][0], r["stats"]["test_r2"][1])
                
                print(f"{r['method']:<12} {r['mr']:<6} {mae_str:<20} {rmse_str:<20} {r2_str:<20}")
    
    print("\n" + "=" * 100)


def generate_improvement_table(results: List[Dict]) -> str:
    """生成改善百分比表 (相对于Baseline)"""
    lines = []
    lines.append("\n## Relative Improvement vs Baseline\n")
    lines.append("| MR | Method | MAE Improvement | RMSE Improvement | R² Improvement |")
    lines.append("|----|--------|-----------------|------------------|----------------|")
    
    for mr in ["0.3", "0.6", "0.9"]:
        # 找到Baseline
        baseline = next((r for r in results if r["mr"] == mr and r["method"] == "Baseline"), None)
        if not baseline:
            continue
        
        baseline_stats = baseline["stats"]
        
        for method in ["MIM-Single", "MIM-Multi"]:
            method_data = next((r for r in results if r["mr"] == mr and r["method"] == method), None)
            if not method_data:
                continue
            
            method_stats = method_data["stats"]
            
            mae_imp = compute_improvement(baseline_stats["test_mae"][0], method_stats["test_mae"][0])
            rmse_imp = compute_improvement(baseline_stats["test_rmse"][0], method_stats["test_rmse"][0])
            # R²改善计算不同（越大越好）
            r2_imp = "N/A"
            if not np.isnan(baseline_stats["test_r2"][0]) and not np.isnan(method_stats["test_r2"][0]):
                r2_diff = method_stats["test_r2"][0] - baseline_stats["test_r2"][0]
                r2_imp = f"{r2_diff:+.3f}"
            
            lines.append(f"| {mr} | {method} | {mae_imp} | {rmse_imp} | {r2_imp} |")
    
    return "\n".join(lines)


def save_markdown_report(results: List[Dict], output_path: Path):
    """保存Markdown格式报告"""
    
    lines = []
    lines.append("# MIM Ablation Study - Results Summary\n")
    lines.append(f"**Model**: CNN1D  ")
    lines.append(f"**Missing Mechanism**: MAR (Missing At Random)  ")
    lines.append(f"**Missing Rates**: 0.3, 0.6, 0.9  ")
    lines.append(f"**Seeds**: 20 (42, 101-119)  ")
    lines.append(f"**Total Experiments**: 180 (3 methods × 3 MRs × 20 seeds)\n")
    
    lines.append("## Methods Compared\n")
    lines.append("1. **Baseline**: No MIM, 16-dim input, single MR training")
    lines.append("2. **MIM-Single**: MIM with 32-dim input, single MR training (old implementation)")
    lines.append("3. **MIM-Multi**: MIM with 32-dim input, multi-MR training (0.0-0.9, new implementation)\n")
    
    lines.append("## Statistical Results (mean ± std)\n")
    lines.append("| Method | MR | MAE | RMSE | R2 |")
    lines.append("|--------|----|-----|------|-----|")
    
    for mr in ["0.3", "0.6", "0.9"]:
        for method in ["Baseline", "MIM-Single", "MIM-Multi"]:
            r = next((x for x in results if x["mr"] == mr and x["method"] == method), None)
            if not r:
                continue
            
            mae_str = format_value(r["stats"]["test_mae"][0], r["stats"]["test_mae"][1])
            rmse_str = format_value(r["stats"]["test_rmse"][0], r["stats"]["test_rmse"][1])
            r2_str = format_value(r["stats"]["test_r2"][0], r["stats"]["test_r2"][1])
            
            lines.append(f"| {method} | {mr} | {mae_str} | {rmse_str} | {r2_str} |")
    
    # 添加改善表
    lines.append(generate_improvement_table(results))
    
    lines.append("\n## Key Findings\n")
    
    # 自动提取关键发现
    for mr in ["0.3", "0.6", "0.9"]:
        baseline = next((r for r in results if r["mr"] == mr and r["method"] == "Baseline"), None)
        multi = next((r for r in results if r["mr"] == mr and r["method"] == "MIM-Multi"), None)
        
        if baseline and multi:
            base_mae = baseline["stats"]["test_mae"][0]
            multi_mae = multi["stats"]["test_mae"][0]
            
            if not np.isnan(base_mae) and not np.isnan(multi_mae):
                improvement = (base_mae - multi_mae) / base_mae * 100
                lines.append(f"- **MR={mr}**: MIM-Multi achieves **{improvement:.1f}%** lower MAE than Baseline")
    
    lines.append("\n---\n")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n[OK] Markdown report saved to: {output_path}")


def main():
    """主函数"""
    # 路径设置
    csv_dir = Path("results/csv")
    output_path = Path("results/summary/mim_ablation_cnn1d_mar.md")
    
    print("=" * 70)
    print("MIM Ablation Study - Results Analysis")
    print("=" * 70)
    print(f"\nCSV directory: {csv_dir.absolute()}")
    print(f"Output report: {output_path.absolute()}")
    
    if not csv_dir.exists():
        print(f"\n[ERROR] CSV directory not found: {csv_dir}")
        print("Please run experiments first!")
        sys.exit(1)
    
    # 加载所有结果
    print("\nLoading results...")
    results = []
    
    for method, mr, filename in EXPERIMENT_CONFIGS:
        stats = load_and_compute_stats(csv_dir, filename)
        results.append({
            "method": method,
            "mr": mr,
            "filename": filename,
            "stats": stats
        })
        
        # 打印加载状态
        sample_count = 0
        filepath = csv_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                sample_count = len(df)
            except:
                pass
        
        status = f"({sample_count} seeds)" if sample_count > 0 else "[NOT FOUND]"
        print(f"  {method} MR={mr}: {status}")
    
    # 打印汇总表
    print_summary_table(results)
    
    # 保存Markdown报告
    save_markdown_report(results, output_path)
    
    print("\n" + "=" * 70)
    print("Analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
