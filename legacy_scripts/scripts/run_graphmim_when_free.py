"""监控 battle_royale_n30.log，待其长时间未更新后自动启动 GraphMIM n=5。"""
import sys
import time
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

LOG_FILE = project_root / 'results' / 'battle_royale_n30.log'
PATTERNS = ['channel', 'road_course']
N_REPEATS = 5
EPOCHS = 100
CHECK_INTERVAL = 60
IDLE_THRESHOLD = 300


def log(msg):
    print(f'[{time.ctime()}] {msg}', flush=True)


def get_log_mtime():
    if not LOG_FILE.exists():
        return 0
    return LOG_FILE.stat().st_mtime


def run_graphmim(pattern):
    cmd = [
        sys.executable,
        '-m', 'src.main',
        '--dataset', 'XJTU',
        '--batch', '3C',
        '--data_dir', './data/XJTU data',
        '--missing_pattern', pattern,
        '--models', 'MLP-GraphMIM-Uniform', 'LSTM-GraphMIM-Uniform',
        '--n_repeats', str(N_REPEATS),
        '--epochs', str(EPOCHS),
        '--batch_size', '32',
        '--lr', '0.001',
        '--patience', '15',
        '--results_dir', f'./results/graphmim_n5_{pattern}',
    ]
    return subprocess.Popen(cmd, cwd=project_root)


def main():
    log('开始监控 battle_royale_n30...')
    last_mtime = get_log_mtime()

    while True:
        time.sleep(CHECK_INTERVAL)
        current_mtime = get_log_mtime()
        idle_time = time.time() - max(current_mtime, last_mtime)

        if current_mtime != last_mtime:
            log(f'log 仍在更新，继续等待 (idle={idle_time:.0f}s)')
            last_mtime = current_mtime
            continue

        if idle_time > IDLE_THRESHOLD:
            log(f'log 已空闲 {idle_time:.0f}s，准备启动 GraphMIM')
            break

        log(f'log 未更新，但空闲时间不足 (idle={idle_time:.0f}s)')

    for pattern in PATTERNS:
        log(f'启动 GraphMIM / {pattern}')
        p = run_graphmim(pattern)
        p.wait()
        log(f'GraphMIM / {pattern} 完成')

    log('所有 GraphMIM 实验完成')


if __name__ == '__main__':
    main()
