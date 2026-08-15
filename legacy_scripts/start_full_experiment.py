#!/usr/bin/env python3
"""
启动完整实验（100次重复，所有10种模型配置）
在后台运行，即使终端关闭也会继续
"""
import subprocess
import sys
import os
from datetime import datetime

def main():
    # 确保输出目录存在
    os.makedirs('experiments', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'experiments/full_experiment_100repeats_{timestamp}.log'
    
    print("=" * 60)
    print("SOH预测缺失数据处理完整实验")
    print("=" * 60)
    print(f"实验批次: 3C")
    print(f"重复次数: 100")
    print(f"模型配置: 10种 (5模型 × 2配置)")
    print(f"总评估数: 100 × 10 × 9 = 9000")
    print(f"日志文件: {log_file}")
    print(f"预估时间: 6-12小时")
    print("=" * 60)
    print()
    
    # 使用subprocess启动实验进程
    cmd = [
        sys.executable,
        'run_full_experiment.py',
        '--batch', '3C',
        '--n_repeats', '100',
        '--epochs', '50'
    ]
    
    print(f"正在启动实验...")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    # 启动进程，重定向输出到日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Windows: 创建新控制台
        )
    
    print(f"实验已在后台启动，PID: {process.pid}")
    print(f"查看进度: tail -f {log_file}")
    print(f"关闭此终端不会影响实验运行")
    print()
    
    # 保存PID到文件
    with open(f'experiments/experiment_pid_{timestamp}.txt', 'w') as f:
        f.write(str(process.pid))
    
    return process.pid

if __name__ == '__main__':
    pid = main()
    print(f"PID: {pid}")
