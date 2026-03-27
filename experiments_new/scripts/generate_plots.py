#!/usr/bin/env python3
"""
图表生成脚本 - 生成两张3×3大图

图1: MAE随缺失率变化曲线
    - 3×3网格: 行=缺失模式(MCAR/MAR/MNAR), 列=模型架构
    - 每张子图: 8条线 (4插补方法 × 2 MIM状态)
    - 颜色: 相同插补方法用同色
    - 线型: MIM实线, non-MIM虚线

图2: 改进率曲线
    - 3×3网格: 同图1布局
    - 每张子图: 4条线 (4插补方法的改进率)
    - 改进率 = (non-MIM MAE - MIM MAE) / non-MIM MAE × 100%

用法:
    python generate_plots.py --results results/phase1/3C/20260327/test_results.csv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger


def set_plot_style():
    """设置绘图样式"""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def get_model_architecture(model_name: str) -> str:
        """从模型名称提取架构类型"""
        model_name_lower = model_name.lower()
        if 'lstm' in model_name_lower:
            return 'LSTM'
        elif 'cnn' in model_name_lower or 'conv' in model_name_lower:
            return 'CNN'
        elif 'mlp' in model_name_lower or 'dense' in model_name_lower or 'fc' in model_name_lower:
            return 'MLP'
        else:
            # 默认根据常见命名规则推断
            if any(x in model_name_lower for x in ['lstm', 'rnn', 'gru']):
                return 'LSTM'
            elif any(x in model_name_lower for x in ['cnn', 'conv', '1d']):
                return 'CNN'
            else:
                return 'MLP'


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """准备数据，添加架构信息"""
    df = df.copy()
    
    # 添加模型架构列
    if 'architecture' not in df.columns:
        if 'model_name' in df.columns:
            df['architecture'] = df['model_name'].apply(get_model_architecture)
        else:
            # 从模型路径推断
            df['architecture'] = 'MLP'  # 默认值
    
    # 确保use_mim是布尔类型
    df['use_mim'] = df['use_mim'].astype(bool)
    
    # 格式化missing_rate为2位小数
    df['missing_rate'] = df['missing_rate'].round(2)
    
    return df


def plot_mae_by_missing_rate(df: pd.DataFrame, output_dir: Path):
    """
    绘制MAE随缺失率变化曲线 - 3×3大图
    
    行: MCAR, MAR, MNAR (缺失模式)
    列: MLP, LSTM, CNN (模型架构)
    
    每张子图: 8条线 (4插补 × 2 MIM状态)
    """
    # 准备数据
    df = prepare_data(df)
    
    # 定义维度
    modes = ['MCAR', 'MAR', 'MNAR']
    architectures = sorted(df['architecture'].unique())  # 使用实际存在的架构
    imputation_methods = ['zero', 'mean', 'knn', 'iterative']
    
    # 颜色映射 - 每种插补方法一种颜色
    colors = {
        'zero': '#1f77b4',      # 蓝色
        'mean': '#ff7f0e',      # 橙色
        'knn': '#2ca02c',       # 绿色
        'iterative': '#d62728', # 红色
    }
    
    # 创建大图
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('MAE vs Missing Rate by Mode, Architecture, Imputation Method and MIM Usage', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for row_idx, mode in enumerate(modes):
        for col_idx, arch in enumerate(architectures):
            ax = axes[row_idx, col_idx]
            
            # 筛选数据
            mode_arch_data = df[
                (df['missing_mode'] == mode) & 
                (df['architecture'] == arch)
            ]
            
            if len(mode_arch_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{mode} - {arch}')
                continue
            
            # 绘制8条线
            for imp_method in imputation_methods:
                imp_data = mode_arch_data[mode_arch_data['imputation_method'] == imp_method]
                
                if len(imp_data) == 0:
                    continue
                
                color = colors.get(imp_method, '#333333')
                
                # MIM = True (实线)
                mim_data = imp_data[imp_data['use_mim'] == True]
                if len(mim_data) > 0:
                    agg = mim_data.groupby('missing_rate')['mae'].agg(['mean', 'std']).reset_index()
                    ax.plot(agg['missing_rate'], agg['mean'], 
                           label=f'{imp_method} (MIM)' if row_idx == 0 and col_idx == 0 else "",
                           color=color, linestyle='-', linewidth=2, marker='o', markersize=3)
                    ax.fill_between(agg['missing_rate'], 
                                   agg['mean'] - agg['std'], 
                                   agg['mean'] + agg['std'],
                                   color=color, alpha=0.1)
                
                # MIM = False (虚线)
                non_mim_data = imp_data[imp_data['use_mim'] == False]
                if len(non_mim_data) > 0:
                    agg = non_mim_data.groupby('missing_rate')['mae'].agg(['mean', 'std']).reset_index()
                    ax.plot(agg['missing_rate'], agg['mean'], 
                           label=f'{imp_method} (No MIM)' if row_idx == 0 and col_idx == 0 else "",
                           color=color, linestyle='--', linewidth=2, marker='s', markersize=3)
                    ax.fill_between(agg['missing_rate'], 
                                   agg['mean'] - agg['std'], 
                                   agg['mean'] + agg['std'],
                                   color=color, alpha=0.05)
            
            # 设置标题和标签
            # 显示标题，CNN1D显示为CNN
            display_arch = 'CNN' if arch == 'CNN1D' else arch
            ax.set_title(f'{mode} - {display_arch}', fontweight='bold')
            ax.set_xlabel('Missing Rate')
            if col_idx == 0:
                ax.set_ylabel('MAE')
            ax.grid(True, alpha=0.3)
            
            # 只在第一张子图显示图例
            if row_idx == 0 and col_idx == 0:
                # 创建双列图例
                handles, labels = ax.get_legend_handles_labels()
                # 重新排序图例: 先MIM后non-MIM
                new_handles = []
                new_labels = []
                for imp in imputation_methods:
                    for suffix in [' (MIM)', ' (No MIM)']:
                        label = imp + suffix
                        if label in labels:
                            idx = labels.index(label)
                            new_handles.append(handles[idx])
                            new_labels.append(label)
                ax.legend(new_handles, new_labels, loc='upper left', ncol=2, fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'mae_by_missing_rate_3x3.png', bbox_inches='tight')
    plt.savefig(output_dir / 'mae_by_missing_rate_3x3.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ MAE曲线图已保存: {output_dir / 'mae_by_missing_rate_3x3.png'}")


def plot_improvement_rate(df: pd.DataFrame, output_dir: Path):
    """
    绘制改进率曲线 - 3×3大图
    
    行: MCAR, MAR, MNAR (缺失模式)
    列: MLP, LSTM, CNN (模型架构)
    
    每张子图: 4条线 (4插补方法的改进率)
    改进率 = (non-MIM MAE - MIM MAE) / non-MIM MAE × 100%
    """
    # 准备数据
    df = prepare_data(df)
    
    # 定义维度
    modes = ['MCAR', 'MAR', 'MNAR']
    architectures = sorted(df['architecture'].unique())  # 使用实际存在的架构
    imputation_methods = ['zero', 'mean', 'knn', 'iterative']
    
    # 颜色映射 - 与图1相同
    colors = {
        'zero': '#1f77b4',      # 蓝色
        'mean': '#ff7f0e',      # 橙色
        'knn': '#2ca02c',       # 绿色
        'iterative': '#d62728', # 红色
    }
    
    # 创建大图
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('MIM Improvement Rate vs Missing Rate\nImprovement = (Non-MIM MAE - MIM MAE) / Non-MIM MAE × 100%', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for row_idx, mode in enumerate(modes):
        for col_idx, arch in enumerate(architectures):
            ax = axes[row_idx, col_idx]
            
            # 筛选数据
            mode_arch_data = df[
                (df['missing_mode'] == mode) & 
                (df['architecture'] == arch)
            ]
            
            if len(mode_arch_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{mode} - {arch}')
                continue
            
            # 绘制4条改进率曲线
            for imp_method in imputation_methods:
                imp_data = mode_arch_data[mode_arch_data['imputation_method'] == imp_method]
                
                if len(imp_data) == 0:
                    continue
                
                # 按missing_rate分组，计算MIM和non-MIM的MAE
                mim_data = imp_data[imp_data['use_mim'] == True].groupby('missing_rate')['mae'].mean()
                non_mim_data = imp_data[imp_data['use_mim'] == False].groupby('missing_rate')['mae'].mean()
                
                # 计算改进率
                common_rates = set(mim_data.index) & set(non_mim_data.index)
                if len(common_rates) == 0:
                    continue
                
                improvement_rates = []
                missing_rates = sorted(common_rates)
                
                for mr in missing_rates:
                    non_mim_mae = non_mim_data[mr]
                    mim_mae = mim_data[mr]
                    if non_mim_mae > 0:
                        improvement = (non_mim_mae - mim_mae) / non_mim_mae * 100
                        improvement_rates.append(improvement)
                    else:
                        improvement_rates.append(0)
                
                if len(improvement_rates) > 0:
                    color = colors.get(imp_method, '#333333')
                    ax.plot(missing_rates, improvement_rates, 
                           label=imp_method if row_idx == 0 and col_idx == 0 else "",
                           color=color, linestyle='-', linewidth=2.5, marker='o', markersize=4)
            
            # 添加0%参考线
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # 设置标题和标签
            # 显示标题，CNN1D显示为CNN
            display_arch = 'CNN' if arch == 'CNN1D' else arch
            ax.set_title(f'{mode} - {display_arch}', fontweight='bold')
            ax.set_xlabel('Missing Rate')
            if col_idx == 0:
                ax.set_ylabel('Improvement Rate (%)')
            ax.grid(True, alpha=0.3)
            
            # 自动调整Y轴范围，确保0%可见
            ax.autoscale_view()
            y_min, y_max = ax.get_ylim()
            if y_max < 5:
                ax.set_ylim(top=5)
            if y_min > -5:
                ax.set_ylim(bottom=-5)
            
            # 只在第一张子图显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'improvement_rate_3x3.png', bbox_inches='tight')
    plt.savefig(output_dir / 'improvement_rate_3x3.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 改进率图已保存: {output_dir / 'improvement_rate_3x3.png'}")


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """生成详细汇总报告"""
    df = prepare_data(df)
    
    # 按关键维度汇总
    summary = df.groupby(['architecture', 'missing_mode', 'imputation_method', 'use_mim']).agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).round(4)
    
    # 保存CSV
    summary.to_csv(output_dir / 'detailed_summary.csv')
    
    # 计算改进率汇总
    improvement_summary = []
    
    for arch in df['architecture'].unique():
        for mode in df['missing_mode'].unique():
            for imp in df['imputation_method'].unique():
                subset = df[
                    (df['architecture'] == arch) & 
                    (df['missing_mode'] == mode) & 
                    (df['imputation_method'] == imp)
                ]
                
                mim_mae = subset[subset['use_mim'] == True]['mae'].mean()
                non_mim_mae = subset[subset['use_mim'] == False]['mae'].mean()
                
                if pd.notna(mim_mae) and pd.notna(non_mim_mae) and non_mim_mae > 0:
                    improvement = (non_mim_mae - mim_mae) / non_mim_mae * 100
                    improvement_summary.append({
                        'architecture': arch,
                        'missing_mode': mode,
                        'imputation_method': imp,
                        'non_mim_mae': non_mim_mae,
                        'mim_mae': mim_mae,
                        'improvement_rate_%': improvement
                    })
    
    imp_df = pd.DataFrame(improvement_summary)
    if len(imp_df) > 0:
        imp_df.to_csv(output_dir / 'improvement_summary.csv', index=False)
    
    # 生成Markdown报告
    with open(output_dir / 'report.md', 'w') as f:
        f.write("# 实验结果报告\n\n")
        f.write("## 数据概览\n\n")
        f.write(f"- 总记录数: {len(df)}\n")
        f.write(f"- 模型架构: {', '.join(df['architecture'].unique())}\n")
        f.write(f"- 缺失模式: {', '.join(df['missing_mode'].unique())}\n")
        f.write(f"- 插补方法: {', '.join(df['imputation_method'].unique())}\n")
        f.write(f"- MIM状态: {df['use_mim'].unique().tolist()}\n\n")
        
        # 改进率汇总
        if len(imp_df) > 0:
            f.write("## MIM改进率汇总\n\n")
            f.write("| 架构 | 缺失模式 | 插补方法 | Non-MIM MAE | MIM MAE | 改进率(%) |\n")
            f.write("|------|----------|----------|-------------|---------|----------|\n")
            
            for _, row in imp_df.sort_values('improvement_rate_%', ascending=False).iterrows():
                f.write(f"| {row['architecture']} | {row['missing_mode']} | {row['imputation_method']} | "
                       f"{row['non_mim_mae']:.4f} | {row['mim_mae']:.4f} | {row['improvement_rate_%']:.2f} |\n")
            f.write("\n")
            
            # 平均改进率
            f.write("### 平均改进率\n\n")
            f.write(f"- 总体平均改进率: {imp_df['improvement_rate_%'].mean():.2f}%\n")
            f.write(f"- 最佳改进率: {imp_df['improvement_rate_%'].max():.2f}%\n")
            f.write(f"- 最差改进率: {imp_df['improvement_rate_%'].min():.2f}%\n\n")
            
            # 按架构
            f.write("#### 按架构\n\n")
            arch_imp = imp_df.groupby('architecture')['improvement_rate_%'].mean().sort_values(ascending=False)
            for arch, imp in arch_imp.items():
                f.write(f"- {arch}: {imp:.2f}%\n")
            f.write("\n")
            
            # 按缺失模式
            f.write("#### 按缺失模式\n\n")
            mode_imp = imp_df.groupby('missing_mode')['improvement_rate_%'].mean().sort_values(ascending=False)
            for mode, imp in mode_imp.items():
                f.write(f"- {mode}: {imp:.2f}%\n")
            f.write("\n")
            
            # 按插补方法
            f.write("#### 按插补方法\n\n")
            imp_method_imp = imp_df.groupby('imputation_method')['improvement_rate_%'].mean().sort_values(ascending=False)
            for imp_m, imp in imp_method_imp.items():
                f.write(f"- {imp_m}: {imp:.2f}%\n")
            f.write("\n")
        else:
            f.write("## MIM改进率汇总\n\n")
            f.write("未找到完整的MIM和Non-MIM对比数据。\n\n")
    
    print(f"  ✓ 汇总报告已保存: {output_dir / 'report.md'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成3×3大图")
    parser.add_argument("--results", required=True, help="测试结果CSV文件")
    parser.add_argument("--output-dir", help="输出目录")
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="plot",
        log_file=str(output_dir / "plot.log"),
        level="INFO"
    )
    
    logger.info("="*60)
    logger.info("图表生成阶段 - 3×3大图")
    logger.info("="*60)
    
    # 加载数据
    logger.info(f"加载结果: {results_path}")
    df = pd.read_csv(results_path)
    logger.info(f"记录数: {len(df)}")
    
    # 检查必要的列
    required_cols = ['missing_mode', 'missing_rate', 'imputation_method', 'use_mim', 'mae']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"缺少必要列: {missing_cols}")
        return 1
    
    # 设置绘图样式
    set_plot_style()
    
    # 生成图表
    logger.info("生成图表...")
    
    print("\n生成MAE曲线图 (3×3)...")
    plot_mae_by_missing_rate(df, output_dir)
    
    print("\n生成改进率图 (3×3)...")
    plot_improvement_rate(df, output_dir)
    
    print("\n生成汇总报告...")
    generate_summary_report(df, output_dir)
    
    logger.info("="*60)
    logger.info(f"图表生成完成")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*60)
    
    print(f"\n✅ 所有图表已保存到: {output_dir}")
    print(f"   - mae_by_missing_rate_3x3.png")
    print(f"   - improvement_rate_3x3.png")
    print(f"   - report.md")
    
    return 0


if __name__ == "__main__":
    main()
