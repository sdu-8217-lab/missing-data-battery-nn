"""Batch Experiment Visualization - Statistical Results"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


class BatchExperimentPlots:
    """Batch experiment statistical plotting class with academic standards"""
    
    # Color scheme: same model uses same color
    MODEL_COLORS = {
        'MLP': '#1f77b4',      # Blue
        'LSTM': '#ff7f0e',     # Orange
        'GRU': '#2ca02c',      # Green
        'CNN1D': '#d62728',    # Red
        'XGBoost': '#9467bd',  # Purple
    }
    
    # Line styles: Baseline=dashed, MIM=solid
    BASELINE_STYLE = '--'
    MIM_STYLE = '-'
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Academic style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        
    def plot_all(self, df: pd.DataFrame):
        """Generate all statistical plots"""
        # Metric curves with confidence intervals
        self.plot_metric_curves(df, 'mae', 'MAE')
        self.plot_metric_curves(df, 'rmse', 'RMSE')
        self.plot_metric_curves(df, 'r2', 'R²')
        
        # Improvement rates
        self.plot_improvement_rate(df, 'mae')
        self.plot_improvement_rate(df, 'rmse')
        
        # Distribution plots
        self.plot_metric_distribution(df, 'mae')
        self.plot_metric_distribution(df, 'rmse')
        
        # Heatmaps
        self.plot_metric_heatmap(df, 'mae')
        
        # Seed stability (use all available seeds)
        self.plot_seed_stability(df)
    
    def _get_model_color(self, model_name: str) -> str:
        """Get color for a model (same for Baseline and MIM)"""
        for model_type in self.MODEL_COLORS:
            if model_type in model_name:
                return self.MODEL_COLORS[model_type]
        return '#333333'  # Default gray
    
    def _get_line_style(self, model_name: str) -> str:
        """Get line style (Baseline=dashed, MIM=solid)"""
        return self.MIM_STYLE if 'MIM' in model_name else self.BASELINE_STYLE
    
    def plot_metric_curves(self, df: pd.DataFrame, metric: str, metric_label: str):
        """Plot metric curves with 95% confidence intervals"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in sorted(df['model_name'].unique()):
            model_data = df[df['model_name'] == model_name]
            color = self._get_model_color(model_name)
            linestyle = self._get_line_style(model_name)
            
            # Calculate statistics per missing rate
            stats_list = []
            for mr in sorted(model_data['missing_rate'].unique()):
                mr_values = model_data[model_data['missing_rate'] == mr][metric].values
                mean = np.mean(mr_values)
                sem = stats.sem(mr_values)
                ci = sem * stats.t.ppf(0.975, len(mr_values)-1) if len(mr_values) > 1 else 0
                stats_list.append({'missing_rate': mr, 'mean': mean, 'ci': ci})
            
            stats_df = pd.DataFrame(stats_list)
            
            # Plot mean line
            ax.plot(stats_df['missing_rate'], stats_df['mean'],
                   label=model_name, color=color, linestyle=linestyle,
                   linewidth=2, marker='o', markersize=4)
            
            # Plot confidence interval
            ax.fill_between(stats_df['missing_rate'],
                           stats_df['mean'] - stats_df['ci'],
                           stats_df['mean'] + stats_df['ci'],
                           color=color, alpha=0.15)
        
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} vs Missing Rate with 95% Confidence Interval',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_curves_ci.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'{metric}_curves_ci.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_improvement_rate(self, df: pd.DataFrame, metric: str):
        """Plot improvement rate of MIM over Baseline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_types = ['MLP', 'LSTM', 'GRU', 'CNN1D', 'XGBoost']
        
        for model_type in model_types:
            baseline_data = df[(df['model_type'] == model_type.lower()) & 
                              (df['use_mim'] == False)]
            mim_data = df[(df['model_type'] == model_type.lower()) & 
                         (df['use_mim'] == True)]
            
            if len(baseline_data) == 0 or len(mim_data) == 0:
                continue
            
            # Calculate improvement rate per missing rate
            improvement_stats = []
            for mr in sorted(df['missing_rate'].unique()):
                base_vals = baseline_data[baseline_data['missing_rate'] == mr][metric].values
                mim_vals = mim_data[mim_data['missing_rate'] == mr][metric].values
                
                if len(base_vals) > 0 and len(mim_vals) > 0:
                    # For MAE/RMSE: lower is better, improvement = (base - mim) / base
                    # For R2: higher is better, improvement = (mim - base) / |base|
                    if metric in ['mae', 'rmse']:
                        improvements = (base_vals[:len(mim_vals)] - mim_vals[:len(base_vals)]) / base_vals[:len(mim_vals)] * 100
                    else:  # r2
                        improvements = (mim_vals[:len(base_vals)] - base_vals[:len(mim_vals)]) / np.abs(base_vals[:len(mim_vals)]) * 100
                    
                    improvement_stats.append({
                        'missing_rate': mr,
                        'mean': np.mean(improvements),
                        'std': np.std(improvements)
                    })
            
            if improvement_stats:
                imp_df = pd.DataFrame(improvement_stats)
                color = self.MODEL_COLORS.get(model_type, '#333333')
                ax.plot(imp_df['missing_rate'], imp_df['mean'],
                       label=model_type, color=color, linewidth=2,
                       marker='s', markersize=5)
                ax.fill_between(imp_df['missing_rate'],
                               imp_df['mean'] - imp_df['std'],
                               imp_df['mean'] + imp_df['std'],
                               color=color, alpha=0.2)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel(f'Improvement Rate (%)', fontsize=11)
        ax.set_title(f'MIM Improvement Rate over Baseline ({metric.upper()})',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_improvement_rate.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'{metric}_improvement_rate.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_metric_distribution(self, df: pd.DataFrame, metric: str):
        """Plot violin plots showing distribution across seeds"""
        missing_rates = sorted(df['missing_rate'].unique())
        n_cols = 3
        n_rows = (len(missing_rates) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        for idx, mr in enumerate(missing_rates):
            mr_data = df[df['missing_rate'] == mr]
            
            plot_data = []
            labels = []
            colors = []
            
            for model_name in sorted(mr_data['model_name'].unique()):
                model_vals = mr_data[mr_data['model_name'] == model_name][metric].values
                plot_data.append(model_vals)
                labels.append(model_name)
                colors.append(self._get_model_color(model_name))
            
            # Violin plot
            parts = axes[idx].violinplot(plot_data, positions=range(len(labels)),
                                         showmeans=True, showmedians=True)
            
            # Color violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            
            axes[idx].set_xticks(range(len(labels)))
            axes[idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[idx].set_ylabel(metric.upper(), fontsize=10)
            axes[idx].set_title(f'Missing Rate = {mr}', fontsize=11)
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(missing_rates), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{metric.upper()} Distribution Across Random Seeds',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'{metric}_distribution.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_metric_heatmap(self, df: pd.DataFrame, metric: str):
        """Plot heatmap of mean metric values"""
        summary = df.groupby(['model_name', 'missing_rate'])[metric].mean().reset_index()
        pivot = summary.pivot(index='model_name', columns='missing_rate', values=metric)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': metric.upper()})
        
        ax.set_title(f'Mean {metric.upper()} Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel('Model Configuration', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'{metric}_heatmap.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_seed_stability(self, df: pd.DataFrame):
        """Plot seed stability analysis using ALL available seeds"""
        # Dynamically detect model types from data
        available_models = df['model_type'].unique()
        model_types = []
        for m in ['MLP', 'LSTM', 'GRU', 'CNN1D', 'XGBoost']:
            if m.lower() in available_models:
                model_types.append(m)
        n_models = len(model_types)
        
        # Calculate grid size
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        # Get all available seeds (not limited to 10)
        all_seeds = sorted(df['seed'].unique())
        n_seeds = len(all_seeds)
        
        for idx, model_type in enumerate(model_types):
            model_data = df[df['model_type'] == model_type.lower()]
            color = self.MODEL_COLORS.get(model_type, '#333333')
            
            # Plot individual seed curves (with low alpha)
            for seed in all_seeds:
                seed_data = model_data[model_data['seed'] == seed]
                
                # Baseline
                baseline = seed_data[seed_data['use_mim'] == False]
                if len(baseline) > 0:
                    baseline = baseline.sort_values('missing_rate')
                    axes[idx].plot(baseline['missing_rate'], baseline['mae'],
                                  color=color, linestyle=self.BASELINE_STYLE,
                                  alpha=0.15, linewidth=0.5)
                
                # MIM
                mim = seed_data[seed_data['use_mim'] == True]
                if len(mim) > 0:
                    mim = mim.sort_values('missing_rate')
                    axes[idx].plot(mim['missing_rate'], mim['mae'],
                                  color=color, linestyle=self.MIM_STYLE,
                                  alpha=0.15, linewidth=0.5)
            
            # Plot mean curves
            baseline_mean = model_data[model_data['use_mim'] == False].groupby('missing_rate')['mae'].mean()
            mim_mean = model_data[model_data['use_mim'] == True].groupby('missing_rate')['mae'].mean()
            
            if len(baseline_mean) > 0:
                axes[idx].plot(baseline_mean.index, baseline_mean.values,
                              color=color, linestyle=self.BASELINE_STYLE,
                              linewidth=2.5, label=f'{model_type} Baseline')
            
            if len(mim_mean) > 0:
                axes[idx].plot(mim_mean.index, mim_mean.values,
                              color=color, linestyle=self.MIM_STYLE,
                              linewidth=2.5, label=f'{model_type} MIM')
            
            axes[idx].set_title(f'{model_type} (n={n_seeds} seeds)', fontsize=11)
            axes[idx].set_xlabel('Missing Rate', fontsize=10)
            axes[idx].set_ylabel('MAE', fontsize=10)
            axes[idx].legend(loc='best', framealpha=0.9)
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Seed Stability Analysis (All {n_seeds} Seeds)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'seed_stability.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'seed_stability.pdf', bbox_inches='tight')
        plt.close()
