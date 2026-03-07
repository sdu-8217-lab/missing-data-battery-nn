# 环境配置指南

> 统一环境配置文档，适用于所有开发者

---

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Linux/macOS/Windows(WSL2) | Ubuntu 22.04+ |
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 10GB可用空间 | 50GB+ |
| GPU | 可选(CPU可运行) | NVIDIA GPU with 8GB+ VRAM |

---

## 快速开始（3分钟）

### 1. 安装 Miniforge（如未安装）

```bash
# Linux/macOS
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 2. 创建并激活环境

```bash
# 克隆代码
git clone git@github.com:sdu-8217-lab/missing-data-battery-nn.git
cd missing-data-battery-nn

# 创建环境（自动安装所有依赖）
conda env create -f environment.yml
conda activate battery-nn
```

### 3. 验证安装

```bash
python -c "
import torch
import pandas
import pytorch_lightning
print(f'Python: OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

---

## 详细配置

### 环境变量（可选）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export PATH="$HOME/miniforge3/bin:$PATH"
```

### GPU 配置（如有NVIDIA GPU）

```bash
# 验证GPU驱动
nvidia-smi

# 预期输出显示GPU型号和驱动版本
```

### 手动安装（不推荐）

如无法使用 `environment.yml`：

```bash
conda create -n battery-nn python=3.13 -y
conda activate battery-nn

# 安装PyTorch（自动检测GPU）
conda install pytorch pytorch-lightning -c conda-forge

# 安装其他依赖
pip install hydra-core omegaconf wandb pandas numpy scikit-learn matplotlib seaborn tqdm
```

---

## 依赖版本

| 包 | 版本 | 用途 |
|----|------|------|
| Python | 3.13+ | 编程语言 |
| PyTorch | 2.10.0+ | 深度学习框架 |
| PyTorch Lightning | 2.6.0+ | 训练框架 |
| Hydra | 1.3.0+ | 配置管理 |
| Pandas | 3.0.0+ | 数据处理 |
| NumPy | 2.4.0+ | 数值计算 |

完整依赖列表见 `environment.yml`。

---

## 常见问题

### Q1: 安装速度慢？

```bash
# 更换conda国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### Q2: CUDA版本不匹配？

```bash
# 查看CUDA版本
nvcc --version

# 安装对应版本的PyTorch
conda install pytorch cudatoolkit=12.1 -c pytorch -c conda-forge
```

### Q3: 内存不足？

```bash
# 减小批次大小运行实验
python src/main.py ... training.batch_size=32  # 默认64
```

---

## 下一步

环境配置完成后，阅读：
- [快速开始](QUICKSTART.md) - 5分钟运行首个实验
- [项目指南](PROJECT_GUIDE.md) - 完整项目介绍

---

*最后更新: 2026-03-07*
