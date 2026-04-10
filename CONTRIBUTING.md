# 贡献指南

## 开发流程

```bash
# 1. 克隆并安装
pip install -e ".[dev]"

# 2. 运行测试
pytest tests/ -v

# 3. 提交代码（遵循规范）
git commit -m "type: 描述"
```

## 提交规范

| 类型 | 说明 |
|------|------|
| feat | 新功能 |
| fix | Bug修复 |
| refactor | 代码重构 |
| docs | 文档更新 |
| test | 测试相关 |

**示例**: `feat: 添加LSTM变体支持`
