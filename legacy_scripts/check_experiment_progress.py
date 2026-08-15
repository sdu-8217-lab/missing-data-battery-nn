#!/usr/bin/env python3
"""
检查实验进度和结果
"""
import os
import pandas as pd
import glob
from datetime import datetime

def find_latest_experiment():
    """找到最新的实验目录"""
    exp_dirs = glob.glob('experiments/*_3C')
    if not exp_dirs:
        return None
    return max(exp_dirs, key=os.path.getmtime)

def check_progress(exp_dir):
    """检查实验进度"""
    print("=" * 70)
    print("实验进度检查")
    print("=" * 70)
    print(f"实验目录: {exp_dir}")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查结果CSV
    results_dir = os.path.join(exp_dir, 'results')
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print("[!] 尚未生成结果文件，实验可能刚开始或尚未完成任何模型")
        return
    
    # 读取并分析结果
    all_results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
        except Exception as e:
            print(f"读取 {csv_file} 失败: {e}")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"[OK] 已完成评估数: {len(combined)}")
        print()
        
        # 按模型统计
        model_counts = combined.groupby(['model_type', 'use_mim']).size().reset_index(name='count')
        model_counts['model'] = model_counts['model_type'] + '-' + model_counts['use_mim'].map({True: 'MIM', False: 'Baseline'})
        
        print("各模型完成进度:")
        print("-" * 50)
        for _, row in model_counts.iterrows():
            print(f"  {row['model']:20s}: {row['count']:3d} / 900 (预期)")
        print()
        
        # 按种子统计
        seed_counts = combined.groupby('seed').size()
        print(f"已完成种子数: {len(seed_counts)} / 100")
        print(f"种子范围: {seed_counts.index.min()} - {seed_counts.index.max()}")
        print()
        
        # 显示部分结果示例
        print("最新结果示例 (MLP, seed=42):")
        print("-" * 50)
        example = combined[(combined['model_type'] == 'MLP') & (combined['use_mim'] == False) & (combined['seed'] == 42)]
        if not example.empty:
            example_display = example[['missing_rate', 'mae', 'rmse', 'r2']].sort_values('missing_rate')
            print(example_display.to_string(index=False))
        print()
        
        # 预估剩余时间
        models_done = len(combined) / 9000 * 10  # 已完成的模型数（估算）
        if models_done > 0:
            exp_time = os.path.getmtime(exp_dir)
            elapsed = datetime.now().timestamp() - exp_time
            eta_seconds = elapsed / models_done * (10 - models_done)
            eta_hours = eta_seconds / 3600
            print(f"预估剩余时间: {eta_hours:.1f} 小时")
    
    # 检查模型文件
    models_dir = os.path.join(exp_dir, 'models')
    model_files = glob.glob(os.path.join(models_dir, '*.pth'))
    print(f"\n已保存模型数: {len(model_files)}")
    for mf in model_files[:5]:
        print(f"  - {os.path.basename(mf)}")
    if len(model_files) > 5:
        print(f"  ... 还有 {len(model_files) - 5} 个模型")

def main():
    exp_dir = find_latest_experiment()
    if exp_dir:
        check_progress(exp_dir)
    else:
        print("未找到实验目录")

if __name__ == '__main__':
    main()
