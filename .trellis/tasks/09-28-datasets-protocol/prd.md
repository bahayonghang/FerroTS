# 实验数据集获取与评测协议

父任务：`09-28-ferrots-implementation`。计划书对应：第 6 节、第 10.2 节、第 14.1 节第 4 层。

## Goal

让后续每个模型子任务只需引用协议 ID 就能在公共数据集与 `cement-twin` 上运行同一套切分、窗口与指标。本任务交付数据清单、数据卡、评测协议文档与私有数据检查脚本，不交付 Rust 数据加载代码。

## 前置

- 无。可与 `09-28-p0-foundation` 并行。
- DataSpec 的 TOML 格式由 `09-28-p1-data-train-loop` 定义；本任务只固定协议参数，由 P1 转写为配置文件。

## 已知事实

- 公共数据已就位：长期预测 6 个目录从 HF 快照 `thuml/Time-Series-Library@2b66e59ee19dac8f6f19fb5d4997f289fdfea357` 拷贝到 `data/`，9 个文件的字节数、行列数、SHA256 见父任务 `research/public-datasets.md`。
- 水泥数据：只有 `data/阳泉回转窑联合训练孪生样本/`；`data_in.csv` 与 `data_out.csv` 字节相同，统计见父任务 `research/dataset-profile.md`。
- TSLib ETT 切分：小时级训练/验证/测试为 12/4/4 个月（8640/2880/2880 行），验证与测试起点向前借 `seq_len` 行作为历史，只使用前 14,400 行；分钟级行数乘 4。其他长期预测数据集按 `Dataset_Custom` 的 70/10/20 切分。scaler 只用训练段拟合。
- 官方 iTransformer 把时间特征作为额外变量 token 输入，输出只保留原变量。

## Requirements

### 公共数据集

- D1 清单：`docs/datasets/manifest.toml` 记录每个公共文件的来源（HF 仓库、快照 revision）、相对路径、字节数、SHA256。该文件可提交。
- D2 协议 `ett-lsf-v1`：固定行边界切分；L=96；H ∈ {96, 192, 336, 720}；多变量（7 变量）模式；时间特征类型与频率参数取自官方脚本的固定参数；主指标为标准化空间 MSE/MAE（元素加权），同时输出原始单位指标。
- D3 协议 `custom-lsf-v1`：用于 Weather、Exchange、ECL、Traffic、ILI（父任务 D5：在 P4 运行并报告，不设门槛）；`Dataset_Custom` 比例切分；L 与 H 集合取自官方 iTransformer 脚本（ILI 使用其专用 L/H）；其余同 D2。
- D4 参考数值：收集 iTransformer、Informer、DLinear 论文或官方仓库在 9 个公共数据文件上的公开数值，注明来源与提交 SHA。公开数值只作参考；ETT 的验收基准为父任务 D4（本机官方 PyTorch 3 种子，max(2σ, 2%)）。
- D5 官方基准运行环境：`tools/reference/` 下用 `uv` 固定 Python 与 torch 版本，记录运行官方 TSLib/iTransformer 代码的命令与 SHA，供 P1–P3 生成基准结果。

### 阳泉回转窑孪生样本

- D6 数据卡 `docs/datasets/cement-yangquan.md`：列清单、时间范围、采样间隔、缺失率、停窑比例、断档统计、数据质量标记（`三次风温均值` 下限约 383 的原因未查明）、`data_in.csv` 与 `data_out.csv` 的字节数与 SHA256。只写统计量，不写原始行。列单位未提供时标注“单位未提供”。
- D7 协议 `cement-twin-v1`：
  - 已确认（父任务 D3）：1 min 网格均值聚合；预测全部 18 个过程变量；L=96 min；H ∈ {15, 30, 60} min。
  - 默认值（可在本任务实施时依据 DatasetReport 调整并记录理由）：1 min 内有效原始样本少于 3 个时该分钟记为缺失；连续缺失 > 5 min 断段；按日期顺序 70/10/20 切分，边界落在段间隔处；`窑喂料反馈`、`分解炉内温度1反馈` 使用受限因果填充 + 掩码，最大陈旧时间 10 min。
  - 已确认（父任务 D6）：只用稳定运行段。`窑头煤反馈 > 5` 且持续 ≥ 30 min 记为运行段；历史与目标窗口完全落在运行段内；开停窑过渡与停窑时段不参与训练与评价。数据卡报告运行段数量、长度分布与各切分有效窗口数。
  - 指标：分变量原始单位 MAE/RMSE + 标准化空间 MSE/MAE；不报告跨变量原始单位混合总分。

### 私有数据保护

- D8 `.gitignore` 增加 `runs/`。
- D9 检查脚本 `tools/check_private_data.sh`（或等价脚本）：`git ls-files` 中出现 `data/` 或 `runs/` 路径，或 `tests/fixtures/` 下出现文件头含水泥数据列名的 CSV 时，以非零码退出。

## Acceptance Criteria

- [ ] `docs/datasets/manifest.toml` 中每个文件的 SHA256 与 `data/` 下实际文件一致（脚本校验）。
- [ ] `docs/datasets/ett.md` 写明 `ett-lsf-v1` 的全部参数（切分行号、L、H 集合、时间特征、指标定义），并附参考数值与来源；`docs/datasets/custom-lsf.md` 写明 `custom-lsf-v1` 对 5 个非 ETT 数据集的全部参数。
- [ ] `docs/datasets/cement-yangquan.md` 中的行数、缺失数、断档数与父任务 `research/dataset-profile.md` 一致，且不含原始数据行。
- [ ] `cement-twin-v1` 的全部参数为具体数值，无 TBD。
- [ ] `tools/reference/` 能在本机运行官方 TSLib DLinear 在 ETTh1、H=96 上的一次训练并输出 MSE/MAE（验证基准环境可用）。
- [ ] `.gitignore` 含 `runs/`；D9 检查脚本在当前仓库退出码为 0，在临时加入一个 `data/` 路径到索引时退出码非 0。

## Out of Scope

- Rust 数据加载实现（P1）。
- 分类、异常检测、M4 数据集。
- f-CaO 相关协议（父任务 D2）。
