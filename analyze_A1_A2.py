#!/usr/bin/env python3
"""
分析 A1 & A2 实验结果

生成报告和可视化，回答核心问题:
- A1: MIM增益来源 (正则化 vs 信息 vs 维度)
- A2: 块状缺失下MIM效果 (MCAR vs Block)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def load_results(csv_file: str = "results/A1_A2/A1_A2_results.csv") -> pd.DataFrame:
    """Load experiment results."""
    df = pd.read_csv(csv_file)
    return df


def analyze_A1(df: pd.DataFrame):
    """Analyze A1: MIM gain source."""
    
    print("\n" + "="*60)
    print("A1: MIM 增益来源分析")
    print("="*60)
    
    # Filter A1 groups
    a1_groups = ["G1", "G2", "G3", "G4"]
    a1_df = df[df["group"].isin(a1_groups)]
    
    if len(a1_df) == 0:
        print("No A1 data found!")
        return
    
    # Group statistics
    print("\n各组 MAE 统计:")
    print("-" * 40)
    
    stats = []
    for group in a1_groups:
        group_df = a1_df[a1_df["group"] == group]
        if len(group_df) == 0:
            continue
        
        mae_mean = group_df["mae"].mean()
        mae_std = group_df["mae"].std()
        mae_min = group_df["mae"].min()
        mae_max = group_df["mae"].max()
        
        group_name = group_df["group_name"].iloc[0]
        
        print(f"{group} ({group_name}):")
        print(f"  MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  Range: [{mae_min:.4f}, {mae_max:.4f}]")
        print(f"  n={len(group_df)}")
        
        stats.append({
            "group": group,
            "name": group_name,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mae_min": mae_min,
            "mae_max": mae_max,
            "n": len(group_df)
        })
    
    # Compare G1 vs others
    print("\n对比分析 (vs G1 - Standard MIM):")
    print("-" * 40)
    
    g1_mae = a1_df[a1_df["group"] == "G1"]["mae"].mean()
    
    for group in ["G2", "G3", "G4"]:
        group_df = a1_df[a1_df["group"] == group]
        if len(group_df) == 0:
            continue
        
        group_mae = group_df["mae"].mean()
        diff = group_mae - g1_mae
        pct_diff = (diff / g1_mae) * 100 if g1_mae > 0 else 0
        
        group_name = group_df["group_name"].iloc[0]
        
        print(f"{group} vs G1:")
        print(f"  MAE diff: {diff:+.4f} ({pct_diff:+.1f}%)")
        
        if abs(pct_diff) < 10:
            print(f"  → {group_name} 效果接近 G1")
        elif diff > 0:
            print(f"  → {group_name} 效果差于 G1")
        else:
            print(f"  → {group_name} 效果好于 G1")
    
    # A1 conclusion
    print("\n" + "="*60)
    print("A1 结论:")
    print("="*60)
    
    g2_mae = a1_df[a1_df["group"] == "G2"]["mae"].mean()
    g3_mae = a1_df[a1_df["group"] == "G3"]["mae"].mean()
    g4_mae = a1_df[a1_df["group"] == "G4"]["mae"].mean()
    
    g2_diff = abs(g2_mae - g1_mae) / g1_mae * 100 if g1_mae > 0 else 0
    g3_diff = abs(g3_mae - g1_mae) / g1_mae * 100 if g1_mae > 0 else 0
    g4_diff = abs(g4_mae - g1_mae) / g1_mae * 100 if g1_mae > 0 else 0
    
    if g2_diff < 10:
        print("⚠️  正则化假说: G2(随机指示器) ≈ G1(MIM)")
        print("   → MIM增益可能主要来自维度扩展的隐式正则化")
    else:
        print("✓ 信息假说: G2(随机指示器) ≠ G1(MIM)")
        print("   → 真实缺失指示器携带的信息有作用")
    
    if g4_diff < 10:
        print("⚠️  维度假说: G4(复制特征) ≈ G1(MIM)")
        print("   → 仅维度翻倍就能达到MIM效果")
    else:
        print("✓ 维度效应有限: G4(复制特征) ≠ G1(MIM)")
        print("   → 单纯增加维度不能完全解释MIM增益")
    
    if g3_diff < 10:
        print("⚠️  打乱后效果相近: 局部信息结构不重要")
    else:
        print("✓ 打乱后效果变差: 缺失指示器的空间结构有作用")


def analyze_A2(df: pd.DataFrame):
    """Analyze A2: Block missing robustness."""
    
    print("\n" + "="*60)
    print("A2: 块状缺失鲁棒性分析")
    print("="*60)
    
    # Filter A2 groups
    a2_groups = ["B1", "B2"]
    a2_df = df[df["group"].isin(a2_groups)]
    
    if len(a2_df) == 0:
        print("No A2 data found!")
        return
    
    print("\n各组 MAE 统计:")
    print("-" * 40)
    
    for group in a2_groups:
        group_df = a2_df[a2_df["group"] == group]
        if len(group_df) == 0:
            continue
        
        mae_mean = group_df["mae"].mean()
        mae_std = group_df["mae"].std()
        group_name = group_df["group_name"].iloc[0]
        
        print(f"{group} ({group_name}):")
        print(f"  MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  n={len(group_df)}")
    
    # Compare B2 vs B1
    b1_df = a2_df[a2_df["group"] == "B1"]
    b2_df = a2_df[a2_df["group"] == "B2"]
    
    if len(b1_df) > 0 and len(b2_df) > 0:
        b1_mae = b1_df["mae"].mean()
        b2_mae = b2_df["mae"].mean()
        
        diff = b2_mae - b1_mae
        pct_diff = (diff / b1_mae) * 100 if b1_mae > 0 else 0
        
        print(f"\nB2 (Block) vs B1 (MCAR):")
        print(f"  MAE diff: {diff:+.4f} ({pct_diff:+.1f}%)")
        
        print("\n" + "="*60)
        print("A2 结论:")
        print("="*60)
        
        if abs(pct_diff) < 10:
            print("✓ MIM在块状缺失下仍有效")
            print("  → MIM对真实传感器故障场景鲁棒")
            print("  → 外部有效性高")
        elif diff > 0:
            print("⚠️  块状缺失下效果下降")
            print("  → MIM在结构化缺失下效果减弱")
            print(f"  → 论文贡献边界收缩到{abs(pct_diff):.0f}%以内")
        else:
            print("★ 块状缺失下效果更好!")
            print("  → 新发现: 结构化缺失是MIM的主战场")


def generate_report(output_file: str = "results/A1_A2/A1_A2_findings_report.md"):
    """Generate markdown report."""
    
    try:
        df = load_results()
    except FileNotFoundError:
        print("No results file found. Run experiments first!")
        return
    
    # Create report
    lines = []
    lines.append("# A1 & A2 实验结果报告\n")
    lines.append("## 实验设计\n")
    lines.append("### A1: MIM 增益来源研究\n")
    lines.append("- **G1 (MIM)**: z = [x̂ | m] - 标准MIM\n")
    lines.append("- **G2 (Random)**: z = [x̂ | r], r~Bernoulli(0.4) - 正则化假说\n")
    lines.append("- **G3 (Shuffled)**: z = [x̂ | m_shuffled] - 信息假说\n")
    lines.append("- **G4 (Copy)**: z = [x̂ | x̂] - 维度假说\n")
    lines.append("\n")
    lines.append("### A2: 块状缺失鲁棒性\n")
    lines.append("- **B1 (MCAR)**: 随机均匀缺失\n")
    lines.append("- **B2 (Block)**: 连续5-cycle缺失 (传感器故障模拟)\n")
    lines.append("\n")
    
    # A1 Results
    lines.append("## A1 结果\n")
    a1_df = df[df["group"].isin(["G1", "G2", "G3", "G4"])]
    
    if len(a1_df) > 0:
        lines.append("| 组别 | 名称 | MAE (mean±std) | n |\n")
        lines.append("|------|------|----------------|---|\n")
        
        for group in ["G1", "G2", "G3", "G4"]:
            group_df = a1_df[a1_df["group"] == group]
            if len(group_df) > 0:
                name = group_df["group_name"].iloc[0]
                mae_mean = group_df["mae"].mean()
                mae_std = group_df["mae"].std()
                lines.append(f"| {group} | {name} | {mae_mean:.4f}±{mae_std:.4f} | {len(group_df)} |\n")
        
        lines.append("\n")
    
    # A2 Results
    lines.append("## A2 结果\n")
    a2_df = df[df["group"].isin(["B1", "B2"])]
    
    if len(a2_df) > 0:
        lines.append("| 组别 | 名称 | MAE (mean±std) | n |\n")
        lines.append("|------|------|----------------|---|\n")
        
        for group in ["B1", "B2"]:
            group_df = a2_df[a2_df["group"] == group]
            if len(group_df) > 0:
                name = group_df["group_name"].iloc[0]
                mae_mean = group_df["mae"].mean()
                mae_std = group_df["mae"].std()
                lines.append(f"| {group} | {name} | {mae_mean:.4f}±{mae_std:.4f} | {len(group_df)} |\n")
        
        lines.append("\n")
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"\n报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze A1 & A2 results")
    parser.add_argument("--csv", type=str, default="results/A1_A2/A1_A2_results.csv",
                       help="Results CSV file")
    parser.add_argument("--report", action="store_true",
                       help="Generate markdown report")
    
    args = parser.parse_args()
    
    try:
        df = load_results(args.csv)
        print(f"Loaded {len(df)} results from {args.csv}")
        
        analyze_A1(df)
        analyze_A2(df)
        
        if args.report:
            generate_report()
    
    except FileNotFoundError:
        print(f"Results file not found: {args.csv}")
        print("Run experiments first:")
        print("  python run_A1_A2_experiments.py --run-all")


if __name__ == "__main__":
    main()
