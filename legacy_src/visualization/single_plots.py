"""Single Experiment Visualization - Individual Seed Results"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


class SingleExperimentPlots:
    """Single experiment plotting class with academic standards"""
    
    # Color scheme: same as batch plots for consistency
    MODEL_COLORS = {
        'MLP': '#1f77b4',
        'LSTM': '#ff7f0e',
        'GRU': '#2ca02c',
        'CNN1D': '#d62728',
        'XGBoost': '#9467bd',
    }
    
    BASELINE_STYLE = '--'
    MIM_STYLE = '-'
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Academic style
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    
    def _get_model_color(self, model_name: str) -> str:
        """Get color for model"""
        for model_type in self.MODEL_COLORS:
            if model_type in model_name:
                return self.MODEL_COLORS[model_type]
        return '#333333'
    
    def _get_line_style(self, model_name: str) -> str:
        """Get line style"""
        return self.MIM_STYLE if 'MIM' in model_name else self.BASELINE_STYLE
    
    def plot_all(self, df: pd.DataFrame):
        """Generate all plots for single experiment"""
        # Metric curves
        self.plot_metric_curves(df, 'mae', 'MAE')
        self.plot_metric_curves(df, 'rmse', 'RMSE')
        self.plot_metric_curves(df, 'r2', 'R²')
        
        # Model comparison
        self.plot_model_comparison(df)
        
        # Improvement analysis
        self.plot_improvement_analysis(df)
    
    def plot_metric_curves(self, df: pd.DataFrame, metric: str, metric_label: str):
        """Plot metric vs missing rate curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in sorted(df['model_name'].unique()):
            model_data = df[df['model_name'] == model_name]
            grouped = model_data.groupby('missing_rate')[metric].mean()
            
            color = self._get_model_color(model_name)
            linestyle = self._get_line_style(model_name)
            
            ax.plot(grouped.index, grouped.values,
                   label=model_name, color=color, linestyle=linestyle,
                   linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} vs Missing Rate (Seed {df["seed"].iloc[0]})',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, df: pd.DataFrame):
        """Plot model comparison at key missing rates"""
        key_rates = [0.1, 0.5, 0.9]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, mr in enumerate(key_rates):
            mr_data = df[df['missing_rate'] == mr]
            grouped = mr_data.groupby('model_name')['mae'].mean().sort_values()
            
            if len(grouped) == 0:
                axes[idx].set_title(f'Missing Rate = {mr}', fontsize=11)
                axes[idx].set_xlabel('MAE')
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            colors = [self._get_model_color(name) for name in grouped.index]
            grouped.plot(kind='barh', ax=axes[idx], color=colors)
            axes[idx].set_title(f'Missing Rate = {mr}', fontsize=11)
            axes[idx].set_xlabel('MAE')
        
        plt.suptitle('Model Comparison at Key Missing Rates', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_improvement_analysis(self, df: pd.DataFrame):
        """Plot MIM improvement over baseline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_types = ['MLP', 'LSTM', 'GRU', 'CNN1D', 'XGBoost']
        
        for model_type in model_types:
            baseline = df[(df['model_type'] == model_type.lower()) & (df['use_mim'] == False)]
            mim = df[(df['model_type'] == model_type.lower()) & (df['use_mim'] == True)]
            
            if len(baseline) == 0 or len(mim) == 0:
                continue
            
            improvements = []
            missing_rates = sorted(df['missing_rate'].unique())
            
            for mr in missing_rates:
                base_mae = baseline[baseline['missing_rate'] == mr]['mae'].mean()
                mim_mae = mim[mim['missing_rate'] == mr]['mae'].mean()
                
                if pd.notna(base_mae) and pd.notna(mim_mae) and base_mae > 0:
                    imp = (base_mae - mim_mae) / base_mae * 100
                    improvements.append(imp)
                else:
                    improvements.append(0)
            
            color = self.MODEL_COLORS.get(model_type, '#333333')
            ax.plot(missing_rates, improvements,
                   label=model_type, color=color,
                   linewidth=2, marker='s', markersize=5)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel('Improvement (%)', fontsize=11)
        ax.set_title(f'MIM Improvement over Baseline (Seed {df["seed"].iloc[0]})',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
