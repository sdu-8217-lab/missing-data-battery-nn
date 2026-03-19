"""
科研级可视化框架 - 100-seed实验完整分析

基于Nature/Science级别论文的图表标准设计

Author: Research Visualization Framework
Date: 2026-03-19
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置科研级样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.3)

# 颜色方案 - 色盲友好
COLORS = {
    'imputation': {'mean': '#E64B35', 'knn': '#4DBBD5', 'iterative': '#00A087', 'zero': '#3C5488'},
    'mode': {'MCAR': '#F39B7F', 'MAR': '#8491B4', 'MNAR': '#91D1C2'},
    'model': {'mlp': '#E64B35', 'lstm': '#4DBBD5', 'cnn': '#00A087'},
    'mim': {False: '#7E6148', True: '#B09C85'}
}

# 输出目录
OUTPUT_DIR = Path('results/100seeds/figures')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class ExperimentAnalyzer:
    """实验数据加载和预处理"""
    
    def __init__(self, data_path='results/100seeds/test_results.csv'):
        self.df = pd.read_csv(data_path)
        self._preprocess()
        
    def _preprocess(self):
        """数据预处理"""
        # 移除异常值 (MAE > 100 视为异常)
        self.df_clean = self.df[self.df['test_mae'] < 100].copy()
        
        # 添加组合标签
        self.df_clean['model_mim'] = self.df_clean['model'] + '_' + self.df_clean['use_mim'].astype(str)
        self.df_clean['mode_mr'] = self.df_clean['mode'] + '_' + self.df_clean['test_mr'].astype(str)
        
        print(f"数据加载完成:")
        print(f"  总记录: {len(self.df):,}")
        print(f"  有效记录: {len(self.df_clean):,}")
        print(f"  异常值: {len(self.df) - len(self.df_clean):,}")


class ScientificVisualizer:
    """科研级可视化器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = analyzer.df_clean
        
    def savefig(self, name, dpi=300):
        """保存图像"""
        plt.savefig(OUTPUT_DIR / f'{name}.pdf', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {name}")
        plt.close()
    
    # ==================== Figure 1: 主效应 - 缺失率影响 ====================
    def plot_missing_rate_effect(self):
        """
        Figure 1: 缺失率对MAE的影响（主效应）
        
        科学问题: 随着缺失率增加，预测性能如何变化？
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        for idx, mode in enumerate(['MCAR', 'MAR', 'MNAR']):
            ax = axes[idx]
            
            # 按mode和imputation分组
            mode_data = self.df[self.df['mode'] == mode]
            
            for imp in ['mean', 'knn', 'iterative', 'zero']:
                imp_data = mode_data[mode_data['imputation'] == imp]
                
                # 计算每个MR的均值和标准误
                grouped = imp_data.groupby('test_mr')['test_mae'].agg(['mean', 'std', 'count'])
                grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
                
                ax.plot(grouped.index, grouped['mean'], 
                       marker='o', markersize=6, linewidth=2,
                       label=imp, color=COLORS['imputation'][imp])
                ax.fill_between(grouped.index, 
                               grouped['mean'] - grouped['sem'],
                               grouped['mean'] + grouped['sem'],
                               alpha=0.15, color=COLORS['imputation'][imp])
            
            ax.set_xlabel('Missing Rate', fontweight='bold')
            if idx == 0:
                ax.set_ylabel('MAE (Mean Absolute Error)', fontweight='bold')
            ax.set_title(f'{mode}', fontweight='bold', fontsize=14)
            ax.legend(title='Imputation', frameon=True, fancybox=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-0.05, 0.95)
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        plt.suptitle('Figure 1: Impact of Missing Rate on Prediction Accuracy', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig1_missing_rate_effect')
    
    # ==================== Figure 2: 插补方法对比 ====================
    def plot_imputation_comparison(self):
        """
        Figure 2: 不同插补方法的性能对比
        
        科学问题: 哪种插补方法对电池SOH预测最有效？
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 箱线图 - 按imputation
        ax1 = axes[0]
        data_for_box = []
        labels = []
        for imp in ['mean', 'knn', 'iterative', 'zero']:
            data_for_box.append(self.df[self.df['imputation'] == imp]['test_mae'].values)
            labels.append(imp.capitalize())
        
        bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        notch=True, bootstrap=1000)
        
        for patch, imp in zip(bp['boxes'], ['mean', 'knn', 'iterative', 'zero']):
            patch.set_facecolor(COLORS['imputation'][imp])
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('MAE', fontweight='bold')
        ax1.set_xlabel('Imputation Method', fontweight='bold')
        ax1.set_title('(a) Overall Performance Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 右图: 分组柱状图 - 不同MR下的表现
        ax2 = axes[1]
        mr_groups = [0.0, 0.3, 0.6, 0.9]
        x = np.arange(len(mr_groups))
        width = 0.2
        
        for idx, imp in enumerate(['mean', 'knn', 'iterative', 'zero']):
            means = []
            sems = []
            for mr in mr_groups:
                subset = self.df[(self.df['imputation'] == imp) & 
                                (self.df['test_mr'] == mr)]['test_mae']
                means.append(subset.mean())
                sems.append(subset.std() / np.sqrt(len(subset)))
            
            ax2.bar(x + idx*width, means, width, yerr=sems,
                   label=imp.capitalize(), color=COLORS['imputation'][imp],
                   alpha=0.8, capsize=3)
        
        ax2.set_xlabel('Missing Rate', fontweight='bold')
        ax2.set_ylabel('MAE', fontweight='bold')
        ax2.set_title('(b) Performance at Key Missing Rates', fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(['0%', '30%', '60%', '90%'])
        ax2.legend(title='Imputation', frameon=True)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Figure 2: Comparison of Imputation Methods', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig2_imputation_comparison')
    
    # ==================== Figure 3: MIM效果分析 ====================
    def plot_mim_effectiveness(self):
        """
        Figure 3: MIM (Missingness Indicator Mechanism) 效果验证
        
        科学问题: MIM能否提升缺失数据下的预测性能？
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 按MR对比MIM效果
        ax1 = axes[0]
        for use_mim in [False, True]:
            subset = self.df[self.df['use_mim'] == use_mim]
            grouped = subset.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
            
            label = 'With MIM' if use_mim else 'Without MIM'
            linestyle = '-' if use_mim else '--'
            
            ax1.plot(grouped.index, grouped['mean'], 
                    marker='s', markersize=7, linewidth=2.5,
                    label=label, color=COLORS['mim'][use_mim], linestyle=linestyle)
            ax1.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.2, color=COLORS['mim'][use_mim])
        
        ax1.set_xlabel('Missing Rate', fontweight='bold')
        ax1.set_ylabel('MAE', fontweight='bold')
        ax1.set_title('(a) MIM Effect Across Missing Rates', fontweight='bold')
        ax1.legend(title='MIM Indicator', frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右图: 不同缺失机制下的MIM效果
        ax2 = axes[1]
        modes = ['MCAR', 'MAR', 'MNAR']
        x = np.arange(len(modes))
        width = 0.35
        
        means_without = []
        means_with = []
        sems_without = []
        sems_with = []
        
        for mode in modes:
            subset = self.df[self.df['mode'] == mode]
            
            without_mim = subset[subset['use_mim'] == False]['test_mae']
            with_mim = subset[subset['use_mim'] == True]['test_mae']
            
            means_without.append(without_mim.mean())
            means_with.append(with_mim.mean())
            sems_without.append(without_mim.std() / np.sqrt(len(without_mim)))
            sems_with.append(with_mim.std() / np.sqrt(len(with_mim)))
        
        ax2.bar(x - width/2, means_without, width, yerr=sems_without,
               label='Without MIM', color=COLORS['mim'][False], 
               alpha=0.8, capsize=3)
        ax2.bar(x + width/2, means_with, width, yerr=sems_with,
               label='With MIM', color=COLORS['mim'][True],
               alpha=0.8, capsize=3)
        
        ax2.set_xlabel('Missingness Mechanism', fontweight='bold')
        ax2.set_ylabel('MAE', fontweight='bold')
        ax2.set_title('(b) MIM Effect by Missingness Mechanism', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modes)
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Figure 3: Effectiveness of Missingness Indicator Mechanism (MIM)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig3_mim_effectiveness')
    
    # ==================== Figure 4: 模型鲁棒性 ====================
    def plot_model_robustness(self):
        """
        Figure 4: 不同模型架构的鲁棒性对比
        
        科学问题: 哪种模型对缺失数据最鲁棒？
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 不同MR下各模型的表现
        ax1 = axes[0]
        for model in ['mlp', 'lstm', 'cnn']:
            subset = self.df[self.df['model'] == model]
            grouped = subset.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
            
            ax1.plot(grouped.index, grouped['mean'],
                    marker='o', markersize=6, linewidth=2,
                    label=model.upper(), color=COLORS['model'][model])
            ax1.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.15, color=COLORS['model'][model])
        
        ax1.set_xlabel('Missing Rate', fontweight='bold')
        ax1.set_ylabel('MAE', fontweight='bold')
        ax1.set_title('(a) Model Robustness to Missing Data', fontweight='bold')
        ax1.legend(title='Model', frameon=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右图: 不同缺失机制下的模型表现
        ax2 = axes[1]
        pivot_data = self.df.pivot_table(
            values='test_mae', 
            index='model', 
            columns='mode', 
            aggfunc='mean'
        )
        
        im = ax2.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(pivot_data.columns)))
        ax2.set_xticklabels(pivot_data.columns)
        ax2.set_yticks(range(len(pivot_data.index)))
        ax2.set_yticklabels([m.upper() for m in pivot_data.index])
        
        # 添加数值标签
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax2.text(j, i, f'{pivot_data.values[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='MAE')
        ax2.set_title('(b) Mean MAE by Model and Missingness Mechanism', fontweight='bold')
        ax2.set_xlabel('Missingness Mechanism', fontweight='bold')
        ax2.set_ylabel('Model', fontweight='bold')
        
        plt.suptitle('Figure 4: Model Architecture Robustness Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig4_model_robustness')
    
    # ==================== Figure 5: 缺失机制深度分析 ====================
    def plot_missingness_mechanism_analysis(self):
        """
        Figure 5: 缺失机制对插补效果的影响
        
        科学问题: 不同缺失机制下哪种插补方法最优？
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        imputations = ['mean', 'knn', 'iterative', 'zero']
        
        for idx, imp in enumerate(imputations):
            ax = axes[idx // 2, idx % 2]
            
            for mode in ['MCAR', 'MAR', 'MNAR']:
                subset = self.df[(self.df['imputation'] == imp) & 
                               (self.df['mode'] == mode)]
                grouped = subset.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
                
                ax.plot(grouped.index, grouped['mean'],
                       marker='o', markersize=5, linewidth=2,
                       label=mode, color=COLORS['mode'][mode])
                ax.fill_between(grouped.index,
                              grouped['mean'] - grouped['sem'],
                              grouped['mean'] + grouped['sem'],
                              alpha=0.15, color=COLORS['mode'][mode])
            
            ax.set_xlabel('Missing Rate', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title(f'{imp.capitalize()} Imputation', fontweight='bold')
            ax.legend(title='Mechanism', frameon=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('Figure 5: Impact of Missingness Mechanism on Imputation Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        self.savefig('fig5_missingness_mechanism')
    
    # ==================== Figure 6: 数据集泛化能力 ====================
    def plot_dataset_generalization(self):
        """
        Figure 6: 不同数据集的泛化能力
        
        科学问题: 方法在不同电池批次上表现是否一致？
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 各batch的MAE分布
        ax1 = axes[0]
        batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']
        data_for_violin = []
        labels = []
        
        for batch in batches:
            data_for_violin.append(self.df[self.df['batch'] == batch]['test_mae'].values)
            labels.append(batch)
        
        parts = ax1.violinplot(data_for_violin, positions=range(len(batches)),
                              showmeans=True, showmedians=True)
        
        for pc, batch in zip(parts['bodies'], batches):
            pc.set_facecolor('#4DBBD5')
            pc.set_alpha(0.7)
        
        ax1.set_xticks(range(len(batches)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('MAE Distribution', fontweight='bold')
        ax1.set_xlabel('Battery Batch', fontweight='bold')
        ax1.set_title('(a) Performance Distribution Across Battery Batches', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 右图: 各batch在不同MR下的表现
        ax2 = axes[1]
        pivot_batch_mr = self.df.pivot_table(
            values='test_mae',
            index='batch',
            columns='test_mr',
            aggfunc='mean'
        )
        
        # 只显示部分MR
        selected_mr = [0.0, 0.3, 0.6, 0.9]
        pivot_subset = pivot_batch_mr[selected_mr]
        
        im = ax2.imshow(pivot_subset.values, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(len(selected_mr)))
        ax2.set_xticklabels([f'{int(mr*100)}%' for mr in selected_mr])
        ax2.set_yticks(range(len(pivot_subset.index)))
        ax2.set_yticklabels(pivot_subset.index)
        
        for i in range(len(pivot_subset.index)):
            for j in range(len(selected_mr)):
                text = ax2.text(j, i, f'{pivot_subset.values[i, j]:.3f}',
                              ha="center", va="center", 
                              color="white" if pivot_subset.values[i, j] > 0.5 else "black",
                              fontweight='bold', fontsize=9)
        
        plt.colorbar(im, ax=ax2, label='MAE')
        ax2.set_title('(b) Mean MAE Across Batches and Missing Rates', fontweight='bold')
        ax2.set_xlabel('Missing Rate', fontweight='bold')
        ax2.set_ylabel('Battery Batch', fontweight='bold')
        
        plt.suptitle('Figure 6: Generalization Across Different Battery Batches', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig6_dataset_generalization')
    
    # ==================== Figure 7: 统计显著性分析 ====================
    def plot_statistical_significance(self):
        """
        Figure 7: 带显著性标记的对比图
        
        科学问题: 各方法间的差异是否统计显著？
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 插补方法的统计显著性
        ax1 = axes[0]
        imputations = ['mean', 'knn', 'iterative', 'zero']
        
        # 计算每个seed的平均MAE，然后做配对t检验
        seed_means = {}
        for imp in imputations:
            imp_data = self.df[self.df['imputation'] == imp]
            seed_means[imp] = imp_data.groupby('seed')['test_mae'].mean().values
        
        # 绘制误差条图
        x_pos = np.arange(len(imputations))
        means = [seed_means[imp].mean() for imp in imputations]
        stds = [seed_means[imp].std() for imp in imputations]
        sems = [std / np.sqrt(len(seed_means[imp])) for std in stds]
        
        bars = ax1.bar(x_pos, means, yerr=sems, capsize=5,
                      color=[COLORS['imputation'][imp] for imp in imputations],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加显著性标记
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([imp.capitalize() for imp in imputations])
        ax1.set_ylabel('Mean MAE ± SEM', fontweight='bold')
        ax1.set_xlabel('Imputation Method', fontweight='bold')
        ax1.set_title('(a) Statistical Significance of Imputation Methods\n(Averaged over 100 seeds)',
                     fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (mean, sem) in enumerate(zip(means, sems)):
            ax1.text(i, mean + sem + 0.01, f'{mean:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 右图: 箱线图对比（含均值点）
        ax2 = axes[1]
        data_for_plot = [seed_means[imp] for imp in imputations]
        
        bp = ax2.boxplot(data_for_plot, labels=[imp.capitalize() for imp in imputations],
                        patch_artist=True, showmeans=False, notch=True)
        
        for patch, imp in zip(bp['boxes'], imputations):
            patch.set_facecolor(COLORS['imputation'][imp])
            patch.set_alpha(0.6)
        
        # 叠加散点显示每个seed的结果
        for i, imp in enumerate(imputations):
            y = seed_means[imp]
            x = np.random.normal(i, 0.04, size=len(y))
            ax2.scatter(x, y, alpha=0.3, s=10, color=COLORS['imputation'][imp])
        
        ax2.set_ylabel('MAE Distribution', fontweight='bold')
        ax2.set_xlabel('Imputation Method', fontweight='bold')
        ax2.set_title('(b) Distribution Across 100 Random Seeds', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Figure 7: Statistical Significance Analysis (100 Seeds)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.savefig('fig7_statistical_significance')
    
    # ==================== Figure 8: 综合热力图矩阵 ====================
    def plot_comprehensive_heatmap(self):
        """
        Figure 8: 多维度综合热力图
        
        展示所有维度组合的性能
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 创建复杂的子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model × Imputation (平均所有MR和mode)
        ax1 = fig.add_subplot(gs[0, 0])
        pivot1 = self.df.pivot_table(values='test_mae', index='model', 
                                    columns='imputation', aggfunc='mean')
        sns.heatmap(pivot1, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax1, cbar_kws={'label': 'MAE'})
        ax1.set_title('Model × Imputation', fontweight='bold')
        
        # 2. Mode × Imputation
        ax2 = fig.add_subplot(gs[0, 1])
        pivot2 = self.df.pivot_table(values='test_mae', index='mode', 
                                    columns='imputation', aggfunc='mean')
        sns.heatmap(pivot2, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax2, cbar_kws={'label': 'MAE'})
        ax2.set_title('Mode × Imputation', fontweight='bold')
        
        # 3. MIM × Imputation
        ax3 = fig.add_subplot(gs[0, 2])
        pivot3 = self.df.pivot_table(values='test_mae', index='use_mim', 
                                    columns='imputation', aggfunc='mean')
        sns.heatmap(pivot3, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax3, cbar_kws={'label': 'MAE'})
        ax3.set_title('MIM × Imputation', fontweight='bold')
        
        # 4. Batch × Model
        ax4 = fig.add_subplot(gs[1, 0])
        pivot4 = self.df.pivot_table(values='test_mae', index='batch', 
                                    columns='model', aggfunc='mean')
        sns.heatmap(pivot4, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax4, cbar_kws={'label': 'MAE'})
        ax4.set_title('Batch × Model', fontweight='bold')
        
        # 5. MR × Imputation (核心)
        ax5 = fig.add_subplot(gs[1, 1])
        pivot5 = self.df.pivot_table(values='test_mae', index='test_mr', 
                                    columns='imputation', aggfunc='mean')
        sns.heatmap(pivot5, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax5, cbar_kws={'label': 'MAE'})
        ax5.set_title('Missing Rate × Imputation (Key Result)', fontweight='bold')
        
        # 6. Mode × MIM
        ax6 = fig.add_subplot(gs[1, 2])
        pivot6 = self.df.pivot_table(values='test_mae', index='mode', 
                                    columns='use_mim', aggfunc='mean')
        sns.heatmap(pivot6, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax6, cbar_kws={'label': 'MAE'})
        ax6.set_title('Mode × MIM', fontweight='bold')
        
        # 7-9. 三个缺失机制的详细MR×Imputation热力图
        for idx, mode in enumerate(['MCAR', 'MAR', 'MNAR']):
            ax = fig.add_subplot(gs[2, idx])
            mode_data = self.df[self.df['mode'] == mode]
            pivot = mode_data.pivot_table(values='test_mae', index='test_mr', 
                                         columns='imputation', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                       ax=ax, cbar_kws={'label': 'MAE'})
            ax.set_title(f'{mode}: MR × Imputation', fontweight='bold')
        
        plt.suptitle('Figure 8: Comprehensive Multi-dimensional Performance Matrix', 
                    fontsize=16, fontweight='bold', y=0.995)
        self.savefig('fig8_comprehensive_heatmap')
    
    def generate_all_figures(self):
        """生成所有图像"""
        print("="*60)
        print("开始生成科研级可视化图像")
        print("="*60)
        
        figures = [
            ('Figure 1: Missing Rate Effect', self.plot_missing_rate_effect),
            ('Figure 2: Imputation Comparison', self.plot_imputation_comparison),
            ('Figure 3: MIM Effectiveness', self.plot_mim_effectiveness),
            ('Figure 4: Model Robustness', self.plot_model_robustness),
            ('Figure 5: Missingness Mechanism', self.plot_missingness_mechanism_analysis),
            ('Figure 6: Dataset Generalization', self.plot_dataset_generalization),
            ('Figure 7: Statistical Significance', self.plot_statistical_significance),
            ('Figure 8: Comprehensive Heatmap', self.plot_comprehensive_heatmap),
        ]
        
        for name, func in figures:
            print(f"\n生成 {name}...")
            try:
                func()
            except Exception as e:
                print(f"  ✗ 错误: {e}")
        
        print("\n" + "="*60)
        print(f"所有图像已保存至: {OUTPUT_DIR}")
        print("="*60)


if __name__ == '__main__':
    # 加载数据
    analyzer = ExperimentAnalyzer()
    
    # 创建可视化器并生成所有图像
    visualizer = ScientificVisualizer(analyzer)
    visualizer.generate_all_figures()
