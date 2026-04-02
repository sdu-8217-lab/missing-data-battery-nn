#!/usr/bin/env python3
"""
自动监控训练进度并在完成后启动下一环节
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "nine_grid_complete"
LOG_FILE = Path("/home/chen/github/missing-data-battery-nn/logs/nine_grid_mcar_training.log")
STATUS_FILE = BASE_DIR / "results" / "training_status.json"

def count_completed_models():
    """统计已完成的模型数"""
    if not RESULTS_DIR.exists():
        return 0
    return len([d for d in RESULTS_DIR.iterdir() if d.is_dir()])

def check_training_running():
    """检查是否有训练进程在运行"""
    result = subprocess.run(
        ["pgrep", "-f", "run_nine_grid_training.py"],
        capture_output=True, text=True
    )
    return result.returncode == 0

def get_last_log_lines(n=10):
    """获取最后 n 行日志"""
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        return lines[-n:]
    except:
        return []

def start_training(mode):
    """启动指定模式的训练"""
    log_path = f"/home/chen/github/missing-data-battery-nn/logs/nine_grid_{mode.lower()}_training.log"
    cmd = [
        "nohup", "python", "-u", "run_nine_grid_training.py",
        "--mode", mode,
        ">", log_path, "2>&1", "&"
    ]
    
    script_dir = BASE_DIR / "scripts"
    subprocess.Popen(
        " ".join(cmd),
        shell=True,
        cwd=script_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print(f"🚀 已启动 {mode} 训练")
    return True

def load_status():
    """加载状态文件"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_modes": [], "current_mode": None}

def save_status(status):
    """保存状态文件"""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def main():
    """主监控循环"""
    print("="*60)
    print("🔍 九宫格训练自动监控启动")
    print("="*60)
    
    status = load_status()
    modes = ["MCAR", "MAR", "MNAR"]
    
    # 确定当前进度
    current_mode = None
    for mode in modes:
        if mode not in status["completed_modes"]:
            current_mode = mode
            break
    
    if current_mode is None:
        print("✅ 所有训练已完成！")
        return
    
    print(f"📋 计划顺序: {' → '.join(modes)}")
    print(f"⏳ 当前环节: {current_mode}")
    print(f"⏰ 检查间隔: 60 秒")
    print("-"*60)
    
    last_count = 0
    stagnant_count = 0
    
    while True:
        completed = count_completed_models()
        running = check_training_running()
        
        # 计算当前模式的进度
        mode_index = modes.index(current_mode)
        models_per_mode = 72
        expected_total = (mode_index * models_per_mode) + models_per_mode
        current_mode_completed = max(0, completed - (mode_index * models_per_mode))
        
        # 显示进度
        progress_pct = (current_mode_completed / models_per_mode) * 100
        print(f"[{time.strftime('%H:%M:%S')}] {current_mode}: {current_mode_completed}/{models_per_mode} "
              f"({progress_pct:.1f}%) | 总计: {completed} | 运行中: {'✅' if running else '❌'}")
        
        # 检查是否卡住
        if completed == last_count:
            stagnant_count += 1
        else:
            stagnant_count = 0
            last_count = completed
        
        # 检查当前模式是否完成
        if current_mode_completed >= models_per_mode or (not running and stagnant_count > 3):
            if current_mode not in status["completed_modes"]:
                status["completed_modes"].append(current_mode)
                save_status(status)
                print(f"\n🎉 {current_mode} 完成！已保存 {completed} 个模型\n")
            
            # 启动下一模式
            next_mode = None
            for mode in modes:
                if mode not in status["completed_modes"]:
                    next_mode = mode
                    break
            
            if next_mode:
                print(f"🚀 自动启动下一环节: {next_mode}")
                start_training(next_mode)
                current_mode = next_mode
                last_count = completed
                stagnant_count = 0
            else:
                print("\n" + "="*60)
                print("🎉🎉🎉 所有训练已完成！")
                print(f"总计: {completed} 个模型")
                print("="*60)
                break
        
        # 如果卡住了但没有进程在运行，尝试重启当前模式
        if stagnant_count > 5 and not running:
            print(f"⚠️ 检测到训练停滞，尝试重启 {current_mode}")
            start_training(current_mode)
            stagnant_count = 0
        
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        sys.exit(0)
