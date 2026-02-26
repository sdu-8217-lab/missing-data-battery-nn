# GPU加速实验配置 (8217专用)

## 硬件配置
- GPU: NVIDIA RTX 4060 Ti (8GB)
- CUDA cores: 4352
- 预估加速: 5-10x vs CPU

## 快速开始

### 1. 克隆代码
```bash
cd ~/
git clone git@github.com:sdu-8217-lab/missing-data-battery-nn.git
cd missing-data-battery-nn
git checkout dev
```

### 2. 配置环境
```bash
# 创建conda环境
conda create -n battery python=3.12 -y
conda activate battery

# 安装GPU版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install pytorch-lightning hydra-core omegaconf wandb pandas numpy scikit-learn matplotlib

# 验证GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. 配置WandB
```bash
wandb login
# 输入API key
```

### 4. 运行实验
```bash
# 单配置测试 (验证GPU工作)
python src/main.py data=xjtu model=mlp method=mim experiment.seeds=[42] training.epochs=10

# 全量实验
python run_full_experiment.py
```

## 性能优化设置

### 多进程并行 (利用GPU+CPU)
```python
# run_parallel_gpu.py
import subprocess
import multiprocessing as mp

# 同时运行4个配置 (GPU内存限制)
max_workers = 4

experiments = [
    ("xjtu", "mcar", "mim", "mlp"),
    ("xjtu", "mcar", "mean", "mlp"),
    # ... 其他配置
]

def run_exp(dataset, mode, method, model):
    cmd = f"python src/main.py data={dataset} missing={mode} method={method} model={model}"
    subprocess.run(cmd, shell=True)

with mp.Pool(max_workers) as pool:
    pool.starmap(run_exp, experiments)
```

### 混合精度训练 (额外2x加速)
```yaml
# configs/config.yaml 中添加
training:
  precision: 16  # FP16混合精度
```

### 减少种子数 (如果需要)
```yaml
# 30 seeds足够统计显著
experiment:
  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
```

## 预期性能

| 配置 | CPU (i7-14700) | GPU (4060Ti) | 加速比 |
|------|---------------|-------------|--------|
| 单种子 | 3-5分钟 | 30-60秒 | 5-10x |
| 单配置(100 seeds) | 5-8小时 | 1-1.5小时 | 5-6x |
| 总实验(160配置) | 33-53天 | 7-10天 | 5x |

**多进程并行后**: 2-3天完成全部实验

## 协作流程

1. **主电脑 (OpenClaw-A)**: 代码开发、调试、文档
2. **8217 (OpenClaw-B)**: 大规模训练、GPU加速

同步方式:
```bash
# A开发完 → 推送到GitHub
git push origin dev

# B拉取 → 运行实验
git pull origin dev
python run_full_experiment.py

# B结果 → WandB自动同步
# 主电脑查看: https://wandb.ai/.../battery-soh-missing-data
```

## 监控

- **WandB**: https://wandb.ai/chenqingyang-shandong-university/battery-soh-missing-data
- **本地日志**: `tail -f results/full_experiment*.log`
- **GPU监控**: `watch -n 1 nvidia-smi`
