# WandB 配置指南

## 默认行为

**无需 WandB 账户即可运行代码。**

默认配置中 `wandb.enabled: false`，所有实验结果会保存到本地 CSV 文件（`./results/` 目录）。

---

## 为什么要使用 WandB？

| 功能 | 本地 CSV | WandB |
|------|---------|-------|
| 记录训练过程 | ❌ 仅最终结果 | ✅ 实时 loss/MAE 曲线 |
| 实验对比 | ❌ 手动整理 | ✅ 网页直观对比 |
| 超参数追踪 | ❌ 易丢失 | ✅ 自动保存 |
| 团队协作 | ❌ 文件传递 | ✅ 共享链接 |
| 论文可复现 | ❌ 难验证 | ✅ 提供实验链接 |

**推荐场景**：正式实验、多组对比、需要远程监控、论文发表。

---

## 快速配置（2分钟）

### 1. 安装并登录

```bash
# 确保已安装 wandb
pip install wandb

# 登录（会提示输入 API key）
wandb login
```

获取 API key：
1. 访问 https://wandb.ai/site
2. 注册/登录账户
3. 点击右上角头像 → "Settings" → "API keys"
4. 复制 key 粘贴到命令行

### 2. 启用 WandB

**方式一：命令行临时启用**
```bash
python src/main.py wandb.enabled=true
```

**方式二：修改配置文件**
编辑 `configs/config.yaml`：
```yaml
wandb:
  enabled: true  # 改为 true
  entity: your-username  # 可选：指定用户名或团队名
  project: battery-soh-missing-data
```

**方式三：环境变量**
```bash
export WANDB_MODE=online  # online/offline/disabled
python src/main.py
```

---

## 查看实验结果

运行后会在终端显示链接：
```
wandb: 🚀 View run at https://wandb.ai/username/battery-soh-missing-data/runs/xxxxx
```

点击链接即可在网页查看：
- 实时训练曲线
- 超参数配置
- 指标对比
- 系统资源使用

---

## 离线模式（服务器无外网）

如果运行环境无外网连接：

```bash
# 1. 设置为离线模式
export WANDB_MODE=offline

# 2. 运行实验
python src/main.py wandb.enabled=true

# 3. 实验结束后，在有网络的机器上上传
wandb sync wandb/offline-run-*
```

---

## 故障排除

### 问题：提示 "No API key configured"

**解决**：
```bash
# 方案一：禁用 WandB（默认行为）
python src/main.py wandb.enabled=false

# 方案二：配置 API key
wandb login
```

### 问题：无法连接 WandB 服务器

**解决**：
```bash
# 使用离线模式
export WANDB_MODE=offline
python src/main.py wandb.enabled=true
```

### 问题：不想看到 WandB 警告信息

**解决**：
```bash
# 完全禁用
export WANDB_MODE=disabled
```

---

## 相关文档

- [WANDB_GUIDE.md](WANDB_GUIDE.md) - 完整 WandB 使用指南
- https://docs.wandb.ai - 官方文档
