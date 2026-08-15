"""PINN 物理约束权重敏感性实验

扫描 monotonicity / smoothness 的不同权重组合，快速判断物理约束是否有价值。
"""
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


PYTHON = '/home/chen/research/battery-research/.venv/bin/python'
PATTERN = 'channel'
N_REPEATS = 10
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 15

# 扫描 λ 组合：lambda_monotonicity, lambda_smoothness
LAMBDA_GRID = [
    (0.0, 0.0),
    (0.001, 0.0),
    (0.01, 0.0),
    (0.1, 0.0),
    (1.0, 0.0),
    (0.0, 0.001),
    (0.0, 0.01),
    (0.0, 0.1),
    (0.01, 0.001),
    (0.1, 0.01),
]


def run_experiment(lambda_mono, lambda_smooth):
    name = f"MLP-GraphMIM-Phys-m{lambda_mono}-s{lambda_smooth}"
    results_dir = f"./results/pinn_sensitivity_{PATTERN}/" + name.replace('.', 'p')
    
    cmd = [
        PYTHON, '-m', 'src.main',
        '--dataset', 'XJTU',
        '--batch', '3C',
        '--data_dir', './data/XJTU data',
        '--missing_pattern', PATTERN,
        '--models', 'MLP-GraphMIM-Uniform-Phys',
        '--n_repeats', str(N_REPEATS),
        '--epochs', str(EPOCHS),
        '--batch_size', str(BATCH_SIZE),
        '--lr', str(LR),
        '--patience', str(PATIENCE),
        '--physics_mono', str(lambda_mono),
        '--physics_smooth', str(lambda_smooth),
        '--results_dir', results_dir,
    ]
    
    print(f"\n=== Running {name} ===")
    subprocess.run(cmd, cwd=project_root, check=True)


def main():
    for lambda_mono, lambda_smooth in LAMBDA_GRID:
        run_experiment(lambda_mono, lambda_smooth)


if __name__ == '__main__':
    main()
