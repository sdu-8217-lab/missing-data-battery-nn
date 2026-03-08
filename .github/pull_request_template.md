## 改动描述
<!-- 简要描述这个PR做了什么 -->

## 改动类型
<!-- 勾选适用的类型 -->
- [ ] `feat`: 新功能
- [ ] `fix`: Bug修复
- [ ] `refactor`: 代码重构
- [ ] `docs`: 文档更新
- [ ] `test`: 测试相关
- [ ] `chore`: 构建/工具/配置
- [ ] `experiments`: 实验结果
- [ ] `paper`: 论文相关

## 影响范围
<!-- 勾选受影响的模块 -->
- [ ] `data` - 数据加载与处理
- [ ] `models` - 模型架构
- [ ] `experiments` - 实验框架
- [ ] `configs` - 配置文件
- [ ] `scripts` - 脚本工具
- [ ] `docs` - 文档
- [ ] `paper` - 论文
- [ ] `root` - 根目录配置

## 检查清单
- [ ] 代码遵循项目风格规范（Black格式化）
- [ ] 提交消息符合[Conventional Commits规范](../CONTRIBUTING.md)
- [ ] 本地测试通过（`pytest`或手动测试）
- [ ] 文档已更新（如适用）
- [ ] 添加了必要的注释
- [ ] 没有引入新的警告或错误
- [ ] CI检查通过

## 测试方式
<!-- 描述如何测试这些改动 -->
```bash
# 示例测试命令
pytest tests/test_experiments.py -v
python scripts/run_experiments.py init --dry-run
```

## 关联Issue
<!-- 如有，请填写 -->
Fixes #
Related to #

## 截图（如适用）
<!-- 对于UI改动或实验结果，请提供截图 -->

## 额外说明
<!-- 任何需要审查者特别注意的事项 -->

---

### 审查者指南

**Self-review checklist** (提交者填写):
- [ ] 我是否已经 self-review 了这些代码？
- [ ] 这些改动是否清晰易懂？
- [ ] 是否有需要拆分为更小PR的地方？

**Reviewer checklist** (审查者填写):
- [ ] 代码逻辑正确
- [ ] 命名清晰有意义
- [ ] 没有冗余代码
- [ ] 边界情况已处理
- [ ] 符合项目架构设计
