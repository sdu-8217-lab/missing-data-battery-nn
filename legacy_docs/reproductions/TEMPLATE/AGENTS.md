# TEMPLATE 本地约定

> 本文件记录该复现子项目特有的约定、决策和注意事项。通用约定见 `reproductions/README.md`。

## 1. 命名规范

- 子项目文件夹：`<短标题>_<第一作者姓氏><发表年份>/`
- 模型文件：`src/models/<model_name>.py`
- 配置文件：`configs/<dataset_or_task>.yaml`
- 脚本：`scripts/<动作>_<对象>.py`
- 测试结果：`tests/test_<模块>_<功能>.py`

## 2. 依赖管理

- 优先复用项目根 `requirements.txt` 中已有的依赖。
- 若必须新增依赖（如 `torchcde`、`torchdiffeq`），应在本子目录创建 `requirements.txt` 并在 `README.md` 中说明安装命令。
- 不要修改项目根 `requirements.txt`，除非该依赖被多个复现项目共同需要且经过评审。

## 3. 数据与结果

- `data/` 与 `results/` 已加入根目录 `.gitignore`，不会进入 git。
- 数据获取方式应在 `README.md` 和 `data/README.md` 中说明。
- 实验结果应保存结构化格式（CSV/JSON），便于后续汇总。

## 4. 代码组织

- 模型实现应自包含，避免依赖项目主 `src/models/` 中未稳定的实验性代码。
- 可复用的工具函数（种子、指标、日志）可以通过相对路径导入项目 `src/utils/`、`src/evaluators/`。
- 每个脚本都应支持 `--seed` 参数，默认使用固定种子以保证可复现性。

## 5. 测试要求

- 至少包含：
  1. 一个前向传播形状测试；
  2. 一个数据加载 / 预处理测试；
  3. 一个端到端 smoke test（1 epoch 可跑通）。

## 6. 文档更新

- 修改模型结构、数据流、运行方式后，同步更新 `README.md`、`notes.md` 和本文件。
- 当复现完成时，在 `reproductions/README.md` 的“当前复现项目”表格中更新状态。
