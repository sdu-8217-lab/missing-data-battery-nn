#!/usr/bin/env python3
"""
100种子批量测试脚本 - 测试所有3600个模型
"""
import os
import sys
import subprocess
import time
from datetime import datetime
import json
import pandas as pd

# 配置
SEEDS = list(range(100))
BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']
MODELS = ['mlp', 'lstm', 'cnn']
MIM_FLAGS = ['false', 'true']
MODEL_DIR = 'models/100seeds_v2_final'
RESULTS_DIR = 'results/100seeds_v2_final'

def test_single_model(seed, batch, model, mim):
    """测试单个模型，返回结果列表"""
    model_file = f"{MODEL_DIR}/seed{seed}_batch{batch}_model{model}_{'mim' if mim == 'true' else 'no_mim'}.pt"
    result_file = f"{RESULTS_DIR}/seed{seed}_batch{batch}_model{model}_{'mim' if mim == 'true' else 'no_mim'}_results.csv"
    
    if not os.path.exists(model_file):
        return None, 'model_not_found'
    
    # 如果结果已存在，跳过
    if os.path.exists(result_file):
        return None, 'skipped'
    
    start = time.time()
    cmd = [
        sys.executable, 'experiments/run_experiment.py',
        '--phase', 'batch-test',
        '--seed', str(seed),
        '--batch', batch,
        '--model', model,
        '--use-mim', mim,
        '--model-dir', MODEL_DIR
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            # 解析 JSON 结果
            stdout = result.stdout
            start_marker = 'JSON_RESULTS_START'
            end_marker = 'JSON_RESULTS_END'
            
            start_idx = stdout.find(start_marker)
            end_idx = stdout.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                json_str = stdout[start_idx + len(start_marker):end_idx]
                data = json.loads(json_str)
                
                # 保存为 CSV
                df = pd.DataFrame(data)
                df.to_csv(result_file, index=False)
                
                return elapsed, 'success'
            else:
                return elapsed, 'parse_error'
        else:
            return elapsed, f'failed: {result.stderr[:100]}'
    except subprocess.TimeoutExpired:
        return time.time() - start, 'timeout'
    except Exception as e:
        return time.time() - start, f'error: {str(e)[:100]}'

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 构建测试列表
    tasks = []
    for seed in SEEDS:
        for batch in BATCHES:
            for model in MODELS:
                for mim in MIM_FLAGS:
                    tasks.append((seed, batch, model, mim))
    
    print("="*60)
    print("100种子批量测试")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总模型数: {len(tasks)}")
    
    # 检查已完成的测试
    existing = sum(1 for t in tasks if os.path.exists(
        f"{RESULTS_DIR}/seed{t[0]}_batch{t[1]}_model{t[2]}_{'mim' if t[3] == 'true' else 'no_mim'}_results.csv"
    ))
    
    print(f"已完成: {existing}")
    print(f"待测试: {len(tasks) - existing}")
    print()
    
    # 开始测试
    start_time = time.time()
    success = fail = skip = 0
    times = []
    
    for i, (seed, batch, model, mim) in enumerate(tasks):
        # 显示进度
        if i % 50 == 0 or i == len(tasks) - 1:
            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            eta_seconds = (len(tasks) - i) / speed if speed > 0 else 0
            eta_mins = eta_seconds / 60
            print(f"\n[{i}/{len(tasks)}] 成功:{success} 跳过:{skip} 失败:{fail} | 速度:{speed*60:.1f}模型/分钟 | 预计剩余:{eta_mins:.1f}分钟")
        
        elapsed, status = test_single_model(seed, batch, model, mim)
        
        if status == 'success':
            success += 1
            times.append(elapsed)
            print(".", end='', flush=True)
        elif status == 'skipped':
            skip += 1
            print("s", end='', flush=True)
        else:
            fail += 1
            print(f"\n  [FAIL] seed{seed}_{batch}_{model}_{'mim' if mim=='true' else 'no_mim'}: {status}")
        
        # 每100个保存一次进度
        if (i + 1) % 100 == 0:
            progress = {
                'total': len(tasks),
                'completed': i + 1,
                'success': success,
                'skipped': skip,
                'failed': fail,
                'avg_time': sum(times)/len(times) if times else 0,
                'elapsed': time.time() - start_time
            }
            with open(f'{RESULTS_DIR}/test_progress.json', 'w') as f:
                json.dump(progress, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print("测试完成!")
    print(f"总用时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"成功: {success}, 跳过: {skip}, 失败: {fail}")
    if times:
        print(f"平均每个模型: {sum(times)/len(times):.1f}秒")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
