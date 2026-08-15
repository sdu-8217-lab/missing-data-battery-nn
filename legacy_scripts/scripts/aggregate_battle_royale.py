#!/usr/bin/env python3
"""Battle Royale 结果聚合与统计分析

用途：
1. 汇集所有 n=30 battle royale + graphmim + linear baseline + two-stage 结果
2. 输出主表（方法 × 缺失模式 × MR）
3. 输出配对 Wilcoxon + BH 校正的胜负矩阵
4. 输出诊断图（MAE-MR 曲线，六个缺失模式并列）

用法：
    python scripts/aggregate_battle_royale.py --output ./results/aggregated_battle_royale
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


PATTERNS = ["bernoulli", "block", "channel", "group", "mixed", "road_course"]
MRS = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]


# ------------------------- 数据加载 -------------------------

def load_battle_royale(root: Path) -> pd.DataFrame:
    """加载 battle_royale_* 目录下的所有 n=30 结果。"""
    rows = []
    for pat in PATTERNS:
        d = root / f"battle_royale_{pat}"
        if not d.exists():
            continue
        for run in sorted(d.glob("2026*/")):
            csvs = list((run / "results").glob("experiment_results_*.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            if df["seed"].nunique() < 20:
                continue  # 跳过 n=5/10 早期 smoke
            df["missing_pattern"] = pat
            df["source"] = "battle_royale"
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_graphmim(root: Path) -> pd.DataFrame:
    """加载 graphmim_n30_* 结果。"""
    rows = []
    for pat in PATTERNS:
        d = root / f"graphmim_n30_{pat}"
        if not d.exists():
            continue
        for run in sorted(d.glob("2026*/")):
            csvs = list((run / "results").glob("experiment_results_*.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            df["missing_pattern"] = pat
            df["source"] = "graphmim"
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_linear(root: Path) -> pd.DataFrame:
    """加载 linear_baseline_n30 结果。"""
    rows = []
    for pat in PATTERNS:
        csvs = list((root / "linear_baseline_n30" / f"XJTU_3C_{pat}").glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        df["missing_pattern"] = pat
        df["source"] = "linear"
        # 确保有共同列
        need = {"model", "seed", "missing_rate", "mae", "rmse", "r2"}
        if not need.issubset(df.columns):
            print(f"WARN linear {pat} 列不匹配: {df.columns.tolist()}")
            continue
        rows.append(df[list(need) + ["missing_pattern", "source"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_two_stage(root: Path) -> pd.DataFrame:
    """加载 two_stage_n30 (优先) 或 two_stage_n5 结果并归一化列。"""
    rows = []
    # 优先使用 n=30 目录；如果同一 pattern×imputer 同时存在 n=30 与 n=5，
    # 只保留 n=30。
    seen_keys = set()
    for candidate_dir in ["two_stage_n30", "two_stage_n5"]:
        d = root / candidate_dir
        if not d.exists():
            continue
        for sub in sorted(d.glob("*_n*")):
            csv = sub / "results.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df = df.rename(columns={"MAE": "mae", "RMSE": "rmse", "R2": "r2"})
            df["model"] = df["model"].astype(str) + "-" + df["imputer"].astype(str)
            df["source"] = "two_stage"
            key = (df["missing_pattern"].iloc[0], df["imputer"].iloc[0])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_all(root: Path) -> pd.DataFrame:
    parts = [
        load_battle_royale(root),
        load_graphmim(root),
        load_linear(root),
        load_two_stage(root),
    ]
    parts = [p for p in parts if not p.empty]
    df = pd.concat(parts, ignore_index=True)
    keep = ["model", "seed", "missing_rate", "missing_pattern", "mae", "rmse", "r2", "source"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["mae"])
    # 舍入 missing_rate 防止浮点漂移
    df["missing_rate"] = df["missing_rate"].round(2)
    # 归一化 seed：不同架构在 experiment_runner 中被加了 offset (MLP=0, LSTM=100000, GRU=200000, CNN=300000)
    # 用 seed % 100000 得到重复索引 i (0..29)，用于跨架构配对检验
    df["seed_idx"] = (df["seed"].astype(int) % 100000)
    return df


# ------------------------- 统计 -------------------------

def bh_correction(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """按 (pattern, model, MR) 输出 MAE mean/std/n。"""
    agg = df.groupby(["missing_pattern", "model", "missing_rate"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        n=("seed", "nunique"),
    ).reset_index()
    return agg


def per_pattern_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """按 pattern 汇总每个 model 在 9 个 MR 上的平均 MAE，然后排名。"""
    avg = summary.groupby(["missing_pattern", "model"]).agg(
        mean_mae_over_mr=("mae_mean", "mean"),
        n_min=("n", "min"),
    ).reset_index()
    avg["rank"] = avg.groupby("missing_pattern")["mean_mae_over_mr"].rank(method="min")
    return avg.sort_values(["missing_pattern", "rank"])


def paired_wilcoxon_matrix(df: pd.DataFrame, pattern: str, reference: str,
                            metric: str = "mae") -> pd.DataFrame:
    """在给定 pattern 下，比较 reference vs 其他模型（同 seed_idx × MR 配对）。

    使用 seed_idx (= seed % 100000) 而非 seed，以便跨架构（LSTM/MLP 有不同 offset）配对。
    """
    sub = df[df["missing_pattern"] == pattern]
    if reference not in sub["model"].unique():
        return pd.DataFrame()
    key_cols = ["seed_idx", "missing_rate"]
    ref = sub[sub["model"] == reference][key_cols + [metric]].rename(
        columns={metric: "ref"}
    )
    rows = []
    for m in sorted(sub["model"].unique()):
        if m == reference:
            continue
        alt = sub[sub["model"] == m][key_cols + [metric]].rename(
            columns={metric: "alt"}
        )
        merged = pd.merge(ref, alt, on=key_cols)
        if len(merged) < 10:
            rows.append({
                "reference": reference, "compared": m,
                "n_pairs": len(merged),
                "ref_mae": merged["ref"].mean() if len(merged) else np.nan,
                "alt_mae": merged["alt"].mean() if len(merged) else np.nan,
                "delta_pct": np.nan,
                "wilcoxon_p": np.nan,
                "note": "insufficient_pairs",
            })
            continue
        diff = merged["ref"] - merged["alt"]
        # 若几乎全零就直接标记
        if diff.abs().sum() == 0:
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(merged["ref"], merged["alt"],
                                       alternative="two-sided", zero_method="wilcox")
            except Exception:
                p = np.nan
        rows.append({
            "reference": reference,
            "compared": m,
            "n_pairs": len(merged),
            "ref_mae": merged["ref"].mean(),
            "alt_mae": merged["alt"].mean(),
            "delta_pct": (merged["alt"].mean() - merged["ref"].mean())
                          / merged["ref"].mean() * 100,
            "wilcoxon_p": p,
            "note": "",
        })
    out = pd.DataFrame(rows)
    if len(out):
        valid = out["wilcoxon_p"].notna()
        q = np.full(len(out), np.nan)
        if valid.any():
            q[valid.values] = bh_correction(out.loc[valid, "wilcoxon_p"].values)
        out["wilcoxon_q_bh"] = q
        out["missing_pattern"] = pattern
    return out


# ------------------------- 绘图 -------------------------

def plot_curves(summary: pd.DataFrame, output_dir: Path,
                highlight: list = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if highlight is None:
        highlight = ["MLP-GraphMIM-Uniform", "LSTM-GraphMIM-Uniform",
                     "MLP-MIM-Uniform", "LSTM-MIM-Uniform",
                     "Linear-MIM"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    all_models = sorted(summary["model"].unique())
    color_map = {m: plt.cm.tab20(i / max(len(all_models) - 1, 1))
                 for i, m in enumerate(all_models)}
    for idx, pat in enumerate(PATTERNS):
        ax = axes[idx]
        sub = summary[summary["missing_pattern"] == pat]
        if sub.empty:
            ax.set_title(f"{pat} (no data)")
            ax.axis("off")
            continue
        for m in sorted(sub["model"].unique()):
            mdf = sub[sub["model"] == m].sort_values("missing_rate")
            if len(mdf) == 0:
                continue
            is_highlight = m in highlight
            ax.plot(
                mdf["missing_rate"], mdf["mae_mean"],
                marker="o" if is_highlight else ".",
                linewidth=2.0 if is_highlight else 0.8,
                alpha=1.0 if is_highlight else 0.4,
                color=color_map[m], label=m if is_highlight else None,
            )
            if is_highlight:
                ci = 1.96 * mdf["mae_std"] / np.sqrt(mdf["n"].clip(lower=1))
                ax.fill_between(
                    mdf["missing_rate"],
                    mdf["mae_mean"] - ci, mdf["mae_mean"] + ci,
                    color=color_map[m], alpha=0.15,
                )
        ax.set_title(pat)
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    out = output_dir / "mae_curves_by_pattern.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig -> {out}")


def plot_ranking_heatmap(summary: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = summary.groupby(["missing_pattern", "model"])["mae_mean"].mean().unstack("missing_pattern")
    # 按平均 MAE 排序
    pivot["_avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_avg")
    pivot = pivot.drop(columns=["_avg"])
    pivot = pivot[[p for p in PATTERNS if p in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v * 1000:.1f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title("MAE avg over MR  (numbers = MAE × 1000)")
    plt.colorbar(im, ax=ax, label="MAE")
    plt.tight_layout()
    out = output_dir / "mae_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig -> {out}")


# ------------------------- 主流程 -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./results")
    parser.add_argument("--output", type=str, default="./results/aggregated_battle_royale")
    parser.add_argument("--references", type=str, nargs="+",
                        default=["MLP-GraphMIM-Uniform", "LSTM-GraphMIM-Uniform"],
                        help="用于配对 Wilcoxon 检验的参考方法")
    args = parser.parse_args()

    root = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading from {root} ...")
    df = load_all(root)
    print(f"[INFO] total rows = {len(df):,}")
    print(f"[INFO] models = {df['model'].nunique()}, patterns = {df['missing_pattern'].nunique()}")
    print(f"[INFO] per-source rows:")
    print(df.groupby("source").size())

    df.to_csv(out_dir / "all_results_long.csv", index=False)

    # 1. 主表
    summary = summary_table(df)
    summary.to_csv(out_dir / "summary_mae_mean_std.csv", index=False)
    print(f"[OUT] {out_dir / 'summary_mae_mean_std.csv'}")

    # 1b. 每个 (pattern, model) 在 9 个 MR 上平均后的排名
    ranking = per_pattern_ranking(summary)
    ranking.to_csv(out_dir / "ranking_per_pattern.csv", index=False)
    print(f"[OUT] {out_dir / 'ranking_per_pattern.csv'}")

    # 2. 胜负矩阵（多个参考方法）
    all_wilcoxon = []
    for ref in args.references:
        for pat in PATTERNS:
            tab = paired_wilcoxon_matrix(df, pat, ref)
            if not tab.empty:
                all_wilcoxon.append(tab)
    if all_wilcoxon:
        wil = pd.concat(all_wilcoxon, ignore_index=True)
        wil.to_csv(out_dir / "wilcoxon_paired_bh.csv", index=False)
        print(f"[OUT] {out_dir / 'wilcoxon_paired_bh.csv'}")

    # 3. 图
    plot_curves(summary, fig_dir)
    plot_ranking_heatmap(summary, fig_dir)

    # 4. 屏幕摘要
    print("\n=== Top-3 methods per missing pattern (avg MAE over MR) ===")
    for pat in PATTERNS:
        sub = ranking[ranking["missing_pattern"] == pat].head(5)
        if sub.empty:
            continue
        print(f"\n[{pat}]")
        for _, row in sub.iterrows():
            print(f"  rank {int(row['rank'])}  MAE={row['mean_mae_over_mr']:.5f}  "
                  f"n>={int(row['n_min'])}  {row['model']}")

    print("\n=== GraphMIM vs others (avg MAE ratio, MR=0.5) ===")
    mr05 = summary[summary["missing_rate"] == 0.5]
    for pat in PATTERNS:
        pat_sub = mr05[mr05["missing_pattern"] == pat]
        if pat_sub.empty:
            continue
        best = pat_sub.sort_values("mae_mean").iloc[0]
        print(f"[{pat}] best@MR0.5: {best['model']}  MAE={best['mae_mean']:.5f}  (n={int(best['n'])})")


if __name__ == "__main__":
    main()
