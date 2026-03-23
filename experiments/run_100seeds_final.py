#!/usr/bin/env python3
import os
"""
100随机种子实验 - 最终版
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']
MODELS = ['mlp', 'lstm', 'cnn']
MIM_SETTINGS = ['false', 'true']
SEEDS = list(range(100))

EPOCHS = 200
PATIENCE = 30
MODEL_DIR = "models/100seeds_v2"
RESULTS_FILE = "results/100seeds_v2/test_results.jsonl"

def run_cmd(cmd_list):
    """运行命令"""
    import subprocess
    try:
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            timeout=600,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.returncode == 0
    except:
        return False

def main():
    print("="*70)
    print(f"100 Seeds - 200 Epochs - Multi-MR Average")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    total = len(SEEDS) * len(BATCHES) * len(MODELS) * len(MIM_SETTINGS)
    
    # 训练
    print(f"\nPhase 1: Training {total} models")
    print("-"*70)
    
    start = time.time()
    completed = failed = 0
    
    for seed in SEEDS:
        for batch in BATCHES:
            for model in MODELS:
                for mim in MIM_SETTINGS:
                    model_file = f"{MODEL_DIR}/seed{seed}_batch{batch}_model{model}_{'mim' if mim == 'true' else 'no_mim'}.pt"
                    
                    if os.path.exists(model_file):
                        completed += 1
                        continue
                    
                    success = run_cmd([
                        'python', 'experiments/run_experiment.py', '--phase', 'train',
                        '--seed', str(seed), '--batch', batch, '--model', model, '--use-mim', mim,
                        '--epochs', str(EPOCHS), '--patience', str(PATIENCE), '--val-mr', '-1',
                        '--save-model', '--model-dir', MODEL_DIR
                    ])
                    
                    completed += 1
                    if not success:
                        failed += 1
                    
                    if completed % 50 == 0:
                        elapsed = time.time() - start
                        eta = (total - completed) * elapsed / completed / 3600
                        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | Failed: {failed} | ETA: {eta:.1f}h")
    
    print(f"\nTraining: {completed} done, {failed} failed, {(time.time()-start)/3600:.2f}h")
    
    # 测试
    print(f"\nPhase 2: Testing")
    print("-"*70)
    
    with open(RESULTS_FILE, 'w') as f:
        pass
    
    start = time.time()
    completed = failed = 0
    total_tests = 0
    
    for seed in SEEDS:
        for batch in BATCHES:
            for model in MODELS:
                for mim in MIM_SETTINGS:
                    success = run_cmd([
                        'python', 'experiments/run_experiment.py', '--phase', 'batch-test',
                        '--seed', str(seed), '--batch', batch, '--model', model, '--use-mim', mim,
                        '--model-dir', MODEL_DIR
                    ])
                    
                    completed += 1
                    if not success:
                        failed += 1
                    
                    if completed % 100 == 0:
                        elapsed = time.time() - start
                        eta = (total - completed) * elapsed / completed / 3600
                        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | Failed: {failed} | ETA: {eta:.1f}h")
    
    print(f"\nTesting: {completed} done, {failed} failed, {(time.time()-start)/3600:.2f}h")
    print(f"\nResults: {RESULTS_FILE}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
