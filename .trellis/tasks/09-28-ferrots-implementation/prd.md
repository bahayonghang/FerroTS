# FerroTS Rust 时序预测平台实施（父任务）

## Goal

按 `FerroTS_Rust_Implementation_Plan.md`（方案 0.1，以下简称“计划书”）分阶段实施 FerroTS。第一版发布范围沿用计划书第 15 节：本地 CSV、CPU/单卡 CUDA、FP32、DLinear/iTransformer/Informer、完整检查点、评价与批量预测。

用户追加要求：每个模型阶段的实验除公共数据集外，还必须在阳泉回转窑孪生样本（`cement-twin`）上运行。

本任务是父任务：持有需求来源、计划评审结论、子任务映射、跨子任务验收与最终集成评审。本任务不直接承载实现。

## 计划评审结论

证据：`research/upstream-verification.md`、`research/dataset-profile.md`、`research/public-datasets.md`。

### A. 成立并保留的部分

- A1 技术路线：Candle + 自建训练安全算子层 + 统一实验运行时。计划书第 4 节引用的 Candle 行为在 0.11.0（`31f35b14`）上全部核对成立：`softmax_last_dim`/`layer_norm` 无反向，`LayerNorm::forward` 对连续输入走无反向快路径，`var_keepdim` 除以 n-1，`cumsum` 构造 n×n 三角矩阵，MaxPool2D 在 kernel≠stride 时反向报错，FlashAttn 无反向，AdamW 矩状态随参数 dtype 创建。
- A2 上游语义：iTransformer 的 mean detach、`unbiased=False`、输出 `[:, :, :N]` 切片；TSLib 只用训练段拟合 scaler、`Dataset_Custom` 远程回退，均核对成立。
- A3 阶段顺序：先完成小而完整的训练核心（P0→P1），再扩充模型（P2→P3→P4）。复现版与工程变体分名，能力表以测试结果为准。
- A4 数据协议：掩码损失、按有效元素聚合指标、训练段拟合 scaler、窗口不跨段边界，均适用于水泥数据。

### B. 需要修改的部分（每项映射到子任务需求）

- B1 版本状态过期：Candle `0.11.0` 已于 2026-06-26 发布，可直接固定 tag `0.11.0` / 提交 `31f35b147389700ed2a178ee66a91c3cc25cc80d`，`cudarc 0.19.8`。→ `p0-foundation`
- B2 GPU 平台假设与本机不符：计划书建议首个 GPU 验收环境为 Linux x86_64；本机为 Windows 11 + RTX 5090 Laptop（sm_120）+ CUDA 13.3 + 驱动 617.14，WSL2 Ubuntu-24.04 可用。该组合在 Candle 上未实测。→ `p0-foundation` 构建门槛：Windows 原生优先，失败时切换 WSL2。
- B3 ETT 复现协议缺失：计划书第 2.1 节只描述 `Dataset_Custom` 的 70/10/20 切分；ETT 使用固定 12/4/4 个月行边界（分钟级乘 4），且只使用前 14,400 行（分钟级 57,600 行）。官方 iTransformer 默认把时间特征作为额外变量 token，计划书 P1 未列出时间特征生成。计划书的“固定数据集多种子验证”没有数值门槛（已由 D4 决定）。→ `datasets-protocol`、`p1-data-train-loop`、`p2-itransformer`
- B4 水泥数据与首版数据协议不匹配：10 s 采样 1 GB CSV、54% 时间停窑、20 处 >1 h 断档、两列缺失率 9%–13%、中文列名。计划书首版不含网格对齐与聚合、运行段判定规则和解析缓存。→ `datasets-protocol`、`p1-data-train-loop`
- B5 私有数据保护未定义：`runs/` 未加入 `.gitignore`；计划书未限制由私有数据生成测试夹具或在报告中输出原始行。→ 跨子任务需求 X1

### C. 风险判断

沿用计划书第 16 节：梯度链路被无反向算子切断、算子复杂度退化、恢复状态不完整、数据可获得时间建模错误。

## 已确认决定

- D1 公共数据来源：用户已通过 `hf download thuml/Time-Series-Library --repo-type=dataset` 下载；快照 `2b66e59ee19dac8f6f19fb5d4997f289fdfea357`。长期预测 6 个目录已拷贝到 `data/`（ETT-small、electricity、exchange_rate、illness、traffic、weather），9 个文件 SHA256 与缓存一致（`research/public-datasets.md`）。
- D2 f-CaO 软测量移出范围：用户删除了 `data/阳泉回转窑联合训练扩充样本/`，水泥实验只使用孪生样本。原软测量子任务 `09-28-cement-fcao-softsensor` 已解除链接并标记 `cancelled`，目录由用户手动删除。
- D3 `cement-twin` 协议主参数：1 min 均值聚合；预测全部 18 个过程变量；L=96（min）；H ∈ {15, 30, 60}（min）。
- D4 复现数值门槛：同一台机器、同一切分下运行官方 PyTorch 代码 3 个种子；Rust 实现 3 个种子的均值落在官方均值 ± max(2 倍官方标准差, 2% 相对误差) 内。适用于 P1 DLinear、P2 iTransformer、P3 Informer 的 ETT 实验。
- D5 验收数据集：P1–P3 的 D4 门槛只用 ETT 四个数据集判定；Weather、Exchange、ECL、Traffic、ILI 在 P4 统一运行并报告，不设门槛。
- D6 停窑处理：`cement-twin` 只使用稳定运行段。`窑头煤反馈 > 5` 且持续 ≥ 30 min 记为运行段；历史与目标窗口必须完全落在运行段内；开停窑过渡与停窑时段不参与训练与评价。
- D7 后续模型：用户于 2026-10-13 要求在 `dev` 规划 Informer、WaveForM 等模型。Informer 复用 P3；WaveForM 独立为 `10-13-waveform`，其 Weather 对齐是模型首轮子集，其他五数据集完整比较仍归 P4。PatchTST、TSMixer/TimeMixer、TCN 为候选，不作为本轮全部实施承诺。路线、依赖与实施审批边界见本任务 `design.md`、`implement.md`。

## 子任务映射

| 子任务 | 计划书阶段 | 交付 | 前置子任务 | 实验数据 |
|---|---|---|---|---|
| `09-28-p0-foundation` | P0 | 工作区、固定依赖、`doctor`、训练安全算子、StatefulAdam/AdamW、本机 CUDA 构建门槛 | 无 | 合成数据 |
| `09-28-datasets-protocol` | 新增 | 数据清单与校验和、数据卡、切分与指标协议、私有数据检查 | 无（可与 P0 并行） | 公共数据集、`cement-twin` |
| `09-28-p1-data-train-loop` | P1 | 数据管线、网格聚合与运行段、时间特征、窗口、scaler、掩码损失、基线、基本检查点、CLI | P0、datasets-protocol | 公共数据集、`cement-twin` |
| `09-28-p2-itransformer` | P2 | iTransformer 复现、权重映射、完整恢复状态 | P1 | 公共数据集、`cement-twin` |
| `09-28-p3-informer` | P3 | Informer 复现与专用算子 | P2 已交付运行时；剩余验收单列 | 公共数据集、`cement-twin` |
| `10-13-waveform` | 用户追加 | WaveForM 来源与协议冻结、可导小波/图算子、模型及正式实验 | P2 已交付运行时；独立研究后通过启动评审 | ETT、Weather、`cement-twin` |
| `09-28-p4-experiment-platform` | P4 | 回测、比较、搜索、报告、更多模型 | 比较使用 P1/P2；各模型矩阵分别满足前置 | 公共数据集、`cement-twin` |
| `09-28-p5-precision-serving` | P5 | 低精度、融合算子训练路径、HTTP 服务 | 第一版发布 | 公共数据集、`cement-twin` |

父子结构不表达依赖；各子任务的前置顺序写在各自 `prd.md` 中。

`00-bootstrap-guidelines`（spec 初始化）与本任务树无父子关系。现有 `.trellis/spec/` 为通用模板；建议在 `p0-foundation` 建立 Rust 工作区骨架后再执行该任务，使 spec 以真实代码为依据。

## 数据集矩阵

| ID | 路径 | 变量数 | 频率 | 任务 |
|---|---|---|---|---|
| `ett-h1` / `ett-h2` | `data/ETT-small/ETTh{1,2}.csv` | 7 | 1 h | 多变量预测 |
| `ett-m1` / `ett-m2` | `data/ETT-small/ETTm{1,2}.csv` | 7 | 15 min | 多变量预测 |
| `weather` | `data/weather/weather.csv` | 21 | 10 min | 多变量预测 |
| `exchange` | `data/exchange_rate/exchange_rate.csv` | 8 | 1 d | 多变量预测 |
| `ecl` | `data/electricity/electricity.csv` | 321 | 1 h | 多变量预测 |
| `traffic` | `data/traffic/traffic.csv` | 862 | 1 h | 多变量预测 |
| `ili` | `data/illness/national_illness.csv` | 7 | 1 周 | 多变量预测 |
| `cement-twin` | `data/阳泉回转窑联合训练孪生样本/data_in.csv` | 18 | 10 s 原始，1 min 聚合 | 多变量自预测 |

P1–P3 门槛验收：ETT 四个（D5）。P4 报告：`weather`、`exchange`、`ecl`、`traffic`、`ili`。`cement-twin`：P1–P4 全部参与，无官方基准，结果与基线并列报告。

## 跨子任务需求

- X1 私有数据隔离：`data/`、`runs/` 不进入 Git；测试夹具只能由合成数据或公共数据生成；报告与日志只输出统计量和指标，不输出原始行；私有数据文件只记录大小与 SHA256。
- X2 双数据来源实验：P1、P2、P3、P4 的验收实验必须同时包含公共数据集配置与 `cement-twin` 配置。
- X3 上游固定：所有上游代码与数据引用（Candle、TSLib、iTransformer、HF 数据快照）固定到完整提交 SHA，记录在 manifest 与模型卡中。
- X4 名称分离：复现版模型名与工程变体名分开；工程改动不得进入复现版默认配置。
- X5 实测环境矩阵：GPU 验收在本机 RTX 5090 Laptop 上执行；环境矩阵只记录实测的 OS、驱动、CUDA、Rust、Candle 版本组合。
- X6 结果可追溯：每个实验产物记录 resolved config、数据指纹、代码 SHA、随机种子、设备信息。

## Acceptance Criteria

- [ ] 子任务映射中的任务全部归档；或未完成子任务在本文件中标注延期原因与当前状态。
- [ ] 第一版发布范围（计划书第 15 节）内的功能全部通过对应子任务验收；P3 未通过时，发布说明标注 Informer 为 preview。
- [ ] 汇总结果表存在于 `docs/results/`，至少包含：ETT 四个数据集 × 预测长度 {96,192,336,720} × {DLinear, iTransformer, Informer} 的标准化空间 MSE/MAE（3 个种子均值 ± 标准差，并列官方 PyTorch 结果与 D4 判定）；`cement-twin` × H ∈ {15,30,60} × 同三个模型的分变量原始单位 MAE/RMSE；P4 运行的 5 个非 ETT 数据集结果（附公开参考数值）。
- [ ] `git ls-files` 输出中无 `data/`、`runs/` 路径，无由私有数据派生的文件。
- [ ] 最终集成评审记录（`review.md`）确认 X1–X6 在各子任务中均已满足。

## Out of Scope

- f-CaO 软测量及任何基于扩充样本的实验（D2）。
- 多 GPU 分布式数据并行训练（计划书第 10.2 节只做“一卡一实验”）。
- 桌面 UI。
- Burn、tch-rs 后端。
- 在线/增量训练。
- 分类、异常检测、短期预测（M4）任务。
