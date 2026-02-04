"""架构搜索结果分析与可视化"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional


class ArchitectureAnalyzer:
    """架构搜索分析器"""
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: 搜索结果CSV文件路径
        """
        self.df = pd.read_csv(results_path)
        self.output_dir = Path(results_path).parent / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
    
    def analyze_all(self):
        """运行所有分析"""
        print("=" * 70)
        print("架构搜索结果分析")
        print("=" * 70)
        
        # 1. 帕累托前沿分析
        self.plot_pareto_frontier()
        
        # 2. 每个模型的最佳架构
        self.print_best_architectures()
        
        # 3. 参数量 vs 性能关系
        self.plot_param_vs_performance()
        
        # 4. 综合评分排名
        self.calculate_comprehensive_score()
        
        # 5. 架构复杂度分布
        self.plot_complexity_distribution()
        
        print(f"\n分析图表已保存到: {self.output_dir}")
    
    def find_pareto_frontier(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str,
        minimize_x: bool = True,
        minimize_y: bool = True
    ) -> pd.DataFrame:
        """
        找到帕累托前沿
        
        帕累托最优定义：
        - 不存在另一个点，在x和y上都优于（或等于）当前点，且至少在一个上严格优于
        """
        points = df[[x_col, y_col]].values
        is_pareto = np.ones(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            for j, other in enumerate(points):
                if i != j:
                    # 检查 other 是否支配 point
                    x_better = (other[0] <= point[0]) if minimize_x else (other[0] >= point[0])
                    y_better = (other[1] <= point[1]) if minimize_y else (other[1] >= point[1])
                    
                    x_strict = (other[0] < point[0]) if minimize_x else (other[0] > point[0])
                    y_strict = (other[1] < point[1]) if minimize_y else (other[1] > point[1])
                    
                    if x_better and y_better and (x_strict or y_strict):
                        is_pareto[i] = False
                        break
        
        return df[is_pareto].sort_values(x_col)
    
    def plot_pareto_frontier(self):
        """绘制帕累托前沿图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 参数量 vs MAE（所有模型）
        ax = axes[0, 0]
        pareto = self.find_pareto_frontier(self.df, 'param_count', 'mae_mean')
        
        for model_type in self.df['model_type'].unique():
            model_df = self.df[self.df['model_type'] == model_type]
            ax.scatter(model_df['param_count'], model_df['mae_mean'], 
                      label=model_type, s=80, alpha=0.6)
        
        ax.plot(pareto['param_count'], pareto['mae_mean'], 'r--', 
               linewidth=2, label='Pareto Frontier')
        
        # 标记帕累托点
        for _, row in pareto.iterrows():
            ax.annotate(row['config_name'], 
                       (row['param_count'], row['mae_mean']),
                       fontsize=7, alpha=0.7)
        
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel('Mean MAE')
        ax.set_title('Pareto Frontier: Parameters vs Performance')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # 2. 参数量 vs 训练时间
        ax = axes[0, 1]
        pareto = self.find_pareto_frontier(self.df, 'param_count', 'training_time')
        
        scatter = ax.scatter(self.df['param_count'], self.df['training_time'],
                           c=self.df['mae_mean'], cmap='viridis_r', 
                           s=80, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='MAE')
        
        ax.plot(pareto['param_count'], pareto['training_time'], 'r--',
               linewidth=2, label='Pareto Frontier')
        
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Parameters vs Training Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # 3. 推理时间 vs MAE
        ax = axes[1, 0]
        if 'inference_time_ms' in self.df.columns:
            pareto = self.find_pareto_frontier(self.df, 'inference_time_ms', 'mae_mean')
            
            for model_type in self.df['model_type'].unique():
                model_df = self.df[self.df['model_type'] == model_type]
                ax.scatter(model_df['inference_time_ms'], model_df['mae_mean'],
                          label=model_type, s=80, alpha=0.6)
            
            ax.plot(pareto['inference_time_ms'], pareto['mae_mean'], 'r--',
                   linewidth=2, label='Pareto Frontier')
            
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel('Mean MAE')
            ax.set_title('Inference Speed vs Performance')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # 4. 各模型帕累托前沿对比
        ax = axes[1, 1]
        for model_type in self.df['model_type'].unique():
            model_df = self.df[self.df['model_type'] == model_type]
            pareto = self.find_pareto_frontier(model_df, 'param_count', 'mae_mean')
            ax.plot(pareto['param_count'], pareto['mae_mean'], 
                   marker='o', label=model_type, linewidth=2)
        
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel('Mean MAE')
        ax.set_title('Pareto Frontier by Model Type')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pareto_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pareto_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print("[OK] Pareto frontier analysis completed")
    
    def print_best_architectures(self):
        """打印每个模型的最佳架构"""
        print("\n" + "=" * 70)
        print("各模型最佳架构（按MAE）")
        print("=" * 70)
        
        for model_type in self.df['model_type'].unique():
            model_df = self.df[self.df['model_type'] == model_type]
            best = model_df.loc[model_df['mae_mean'].idxmin()]
            
            print(f"\n【{model_type.upper()}】")
            print(f"  配置: {best['config_name']}")
            print(f"  MAE: {best['mae_mean']:.4f}")
            print(f"  参数量: {best['param_count']:,}")
            print(f"  训练时间: {best['training_time']:.1f}s")
            if 'inference_time_ms' in best:
                print(f"  推理时间: {best['inference_time_ms']:.2f}ms")
    
    def plot_param_vs_performance(self):
        """绘制参数量与性能关系"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['mae_mean', 'mae_max', 'r2_0.5' if 'r2_0.5' in self.df.columns else 'r2']
        titles = ['Mean MAE', 'Max MAE (Worst Case)', 'R² (Missing Rate 0.5)']
        
        for ax, metric, title in zip(axes, metrics, titles):
            if metric not in self.df.columns:
                continue
                
            for model_type in self.df['model_type'].unique():
                model_df = self.df[self.df['model_type'] == model_type]
                ax.scatter(model_df['param_count'], model_df[metric],
                          label=model_type, s=80, alpha=0.6)
            
            ax.set_xlabel('Parameter Count')
            ax.set_ylabel(title)
            ax.set_title(f'Parameters vs {title}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'param_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'param_vs_performance.pdf', bbox_inches='tight')
        plt.close()
        
        print("[OK] Parameter-performance plot completed")
    
    def calculate_comprehensive_score(self):
        """计算综合评分并排名"""
        # 归一化各指标
        df = self.df.copy()
        
        # 需要最小化的指标
        for col in ['mae_mean', 'mae_max', 'training_time', 'param_count']:
            if col in df.columns:
                df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        
        # 需要最大化的指标（R²）
        if 'r2_0.5' in df.columns:
            df['r2_norm'] = 1 - (df['r2_0.5'] - df['r2_0.5'].min()) / (df['r2_0.5'].max() - df['r2_0.5'].min() + 1e-8)
        
        # 综合评分（越低越好）
        weights = {
            'mae_mean_norm': 0.35,
            'mae_max_norm': 0.25,
            'param_count_norm': 0.20,
            'training_time_norm': 0.15,
            'r2_norm': 0.05,
        }
        
        score_cols = [col for col in weights.keys() if col in df.columns]
        df['comprehensive_score'] = sum(df[col] * weights.get(col, 0) for col in score_cols)
        
        # 排序
        df_sorted = df.sort_values('comprehensive_score')
        
        # 保存排名
        df_sorted[['config_name', 'model_type', 'mae_mean', 'param_count', 
                   'training_time', 'comprehensive_score']].to_csv(
            self.output_dir / 'ranking.csv', index=False
        )
        
        # 打印 top 10
        print("\n" + "=" * 70)
        print("综合评分 Top 10")
        print("=" * 70)
        print(f"{'Rank':<6} {'Config':<20} {'Model':<10} {'MAE':<8} {'Params':<12} {'Score':<8}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:<6} {row['config_name']:<20} {row['model_type']:<10} "
                  f"{row['mae_mean']:<8.4f} {row['param_count']:<12,} {row['comprehensive_score']:<8.3f}")
        
        # 可视化
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_15 = df_sorted.head(15)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_15)))
        
        bars = ax.barh(range(len(top_15)), top_15['comprehensive_score'], color=colors)
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels(top_15['config_name'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Comprehensive Score (Lower is Better)')
        ax.set_title('Top 15 Architectures by Comprehensive Score')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_15['comprehensive_score'])):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Comprehensive score analysis completed")
        return df_sorted
    
    def plot_complexity_distribution(self):
        """绘制架构复杂度分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 各模型的参数量分布（箱线图）
        ax = axes[0]
        model_types = []
        param_counts = []
        
        for model_type in self.df['model_type'].unique():
            model_df = self.df[self.df['model_type'] == model_type]
            model_types.append(model_type)
            param_counts.append(model_df['param_count'].values)
        
        bp = ax.boxplot(param_counts, labels=model_types, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Parameter Count')
        ax.set_title('Parameter Count Distribution by Model Type')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # 2. MAE 分布
        ax = axes[1]
        mae_data = []
        for model_type in self.df['model_type'].unique():
            model_df = self.df[self.df['model_type'] == model_type]
            mae_data.append(model_df['mae_mean'].values)
        
        bp = ax.boxplot(mae_data, labels=model_types, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        
        ax.set_ylabel('Mean MAE')
        ax.set_title('MAE Distribution by Model Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Complexity distribution analysis completed")
    
    def recommend_best_balance(self, param_budget: int = 50000) -> pd.Series:
        """
        推荐最佳平衡的架构
        
        Args:
            param_budget: 参数量预算（默认50K）
        """
        # 在预算内选择MAE最小的
        within_budget = self.df[self.df['param_count'] <= param_budget]
        
        if len(within_budget) == 0:
            print(f"警告: 没有架构满足参数量预算 {param_budget:,}")
            return None
        
        best = within_budget.loc[within_budget['mae_mean'].idxmin()]
        
        print("\n" + "=" * 70)
        print(f"最佳平衡架构（参数量 ≤ {param_budget:,}）")
        print("=" * 70)
        print(f"配置: {best['config_name']}")
        print(f"模型: {best['model_type']}")
        print(f"MAE: {best['mae_mean']:.4f}")
        print(f"参数量: {best['param_count']:,}")
        print(f"性价比: {best['mae_mean'] / best['param_count'] * 100000:.4f} MAE/100K params")
        
        return best


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='架构搜索结果分析')
    parser.add_argument('results_path', help='搜索结果CSV文件路径')
    parser.add_argument('--budget', type=int, default=50000, 
                       help='参数量预算（默认50K）')
    
    args = parser.parse_args()
    
    analyzer = ArchitectureAnalyzer(args.results_path)
    analyzer.analyze_all()
    analyzer.recommend_best_balance(args.budget)


if __name__ == '__main__':
    main()
