import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
methods = ['mim', 'mean', 'knn']
labels = ['MIM (Proposed)', 'Mean Imputation', 'KNN Imputation']
colors = ['#2E86AB', '#A23B72', '#F18F01']
results = {}

for method in methods:
    df = pd.read_csv(f'results/battery_soh_experiment_mlp_{method}.csv')
    results[method] = df.groupby('missing_rate')['test_mae'].agg(['mean', 'std'])

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
    mr = results[method].index
    mean = results[method]['mean']
    std = results[method]['std']
    
    ax.plot(mr, mean, 'o-', label=label, color=color, linewidth=2, markersize=8)
    ax.fill_between(mr, mean - std, mean + std, alpha=0.2, color=color)

ax.set_xlabel('Missing Rate', fontsize=12)
ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
ax.set_title('MIM vs Traditional Imputation Methods\n(XJTU 2C, MLP, 3 seeds)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.25)

plt.tight_layout()
plt.savefig('results/comparison_plot.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to: results/comparison_plot.png")
