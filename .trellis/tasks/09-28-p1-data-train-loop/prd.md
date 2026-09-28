# P1 数据管线与基线训练闭环

父任务：`09-28-ferrots-implementation`。计划书对应：第 6 节、第 10 节、第 11 节，第 15 节 P1。

## Goal

本地 CSV → 数据校验与预处理 → 滑动窗口 → 基线模型训练 → 评价 → 预测，在 ETT 与 `cement-twin` 上都能在 CPU 与 `cuda:0` 完成，指标可以用独立脚本从预测表复算。

## 前置

- `09-28-p0-foundation` 完成（工作区、训练安全算子、StatefulAdam、本机 CUDA 构建）。
- `09-28-datasets-protocol` 完成（`ett-lsf-v1`、`cement-twin-v1` 参数固定，ETT 文件就位）。

## Requirements

### 数据读取与校验

- P1-R1 CSV 读取：UTF-8 表头（含中文列名，以及 `p (mbar)`、`% WEIGHTED ILI` 等含空格与符号的列名）；显式时间列与可配置时间格式（`exchange_rate.csv` 为 `1990/1/1 0:00`，其他文件为 `YYYY-MM-DD HH:MM:SS`）；按列类型解析；空值识别；文件不存在时报错，无远程下载。
- P1-R2 DataSpec（TOML，未知键报错）：时间列、采样间隔、排序与重复时间戳策略、输入列、目标列、future-known 列、切分方式（比例 / 显式行边界 / 日期边界）、段边界规则、缺失值模式。目标列只能按名称指定。
- P1-R3 DatasetReport（`ferrots data inspect`）：行数、起止时间、非递增与重复时间戳计数、各列缺失率、常量列、断档统计、各切分的有效窗口数。

### 预处理

- P1-R4 网格对齐与聚合：把不等间隔原始数据对齐到固定网格，支持均值聚合与“最少有效样本数”规则。
- P1-R5 段边界：连续缺失超过阈值时断段；可配置运行状态规则（列名 + 比较运算 + 阈值 + 最短持续时间）生成运行段；窗口不跨段。
- P1-R6 解析缓存：以数据文件指纹 + 预处理配置哈希为键缓存预处理结果；键不变时不再解析 CSV。
- P1-R7 时间特征：实现与 TSLib `time_features`（`timeenc=1`）数值一致的时间特征，频率参数与官方脚本一致。
- P1-R8 缺失值：复现模式遇到缺失值报错；工程模式使用受限因果前向填充（最大陈旧时间）并输出有效掩码。
- P1-R9 标准化：只用训练段拟合每列均值与尺度；常量列使用稳定尺度并记录标记；scaler 按列名保存。

### 窗口与训练

- P1-R10 WindowIndex：历史 `[o-L, o)`，目标 `[o+g, o+g+H)`；验证/测试样本可用前一切分末尾的历史；目标行必须全部属于被评估切分；训练窗口可在合法索引内打乱，验证/测试不打乱。
- P1-R11 掩码 MSE 损失：`sum(mask·(pred-target)²)/sum(mask)`；无效位置先置为有限值；分母为 0 时跳过更新并记录；整个 epoch 无有效标签时失败；梯度累计与评价按有效元素数加权。
- P1-R12 基线模型：LastValue、SeasonalNaive、Linear、DLinear（与 TSLib DLinear 语义一致：移动平均分解，`individual` 开关）。
- P1-R13 Trainer：Adam；按验证损失早停；测试集只在模型选定后评估；验证路径不记录计算图。
- P1-R14 基本检查点：`model.safetensors`、`scaler.json`、`data-schema.json`、`resolved-config.toml`、`manifest.json`（含文件校验和、数据指纹、代码 SHA）；best 与 last；先写临时目录再提交。
- P1-R15 指标：标准化空间与原始单位的 MAE/MSE/RMSE，按变量、按预测步长统计；MAPE 默认关闭；预测表列为 `run_id, series_id, forecast_origin, target_time, horizon, target_name, y_true, y_pred, valid`。
- P1-R16 CLI：`ferrots data inspect`、`ferrots train`、`ferrots evaluate`、`ferrots predict`；配置合并顺序为内置默认 < 配置文件 < CLI 覆盖。
- P1-R17 配置文件：`configs/` 下提供 `ett-lsf-v1` 四个数据集与 `cement-twin-v1` 的 DataSpec 与基线训练配置。

## Acceptance Criteria

- [ ] ETTh1，L=96，H ∈ {96,192,336,720}：训练/验证/测试窗口数与 TSLib `Dataset_ETT_hour` 的 `len()` 相同；ETTm1 同样与 `Dataset_ETT_minute` 相同。
- [ ] 时间特征与 TSLib `time_features` 输出的差异 < 1e-6（夹具由 ETT 数据生成）。
- [ ] DLinear 在 ETTh1/ETTh2/ETTm1/ETTm2 × 4 个预测长度上的标准化 MSE/MAE（3 个种子）满足父任务 D4 门槛（基准：本机运行官方 TSLib DLinear 3 个种子）。
- [ ] `exchange_rate.csv` 的时间列按配置格式解析，行数与起止时间与父任务 `research/public-datasets.md` 一致。
- [ ] `ferrots data inspect` 对 `cement-twin` 原始文件报告的行数、重复时间戳数、各列缺失数与父任务 `research/dataset-profile.md` 一致；聚合后报告运行段数量与各切分有效窗口数，所有窗口完全落在运行段内（测试断言）。
- [ ] `cement-twin-v1`：LastValue、SeasonalNaive、DLinear 在 CPU 与 `cuda:0` 上完成训练与评价，输出分变量原始单位 MAE/RMSE。
- [ ] 独立脚本从预测表复算 3 个变量的 MAE/RMSE，与 `ferrots evaluate` 输出差异 < 1e-6。
- [ ] 第二次加载 `cement-twin` 命中缓存且不解析 CSV（日志可见）；第一次与第二次加载耗时写入实验记录。
- [ ] 回归测试覆盖：尾批、全缺失标签、常量列、L=1、batch=1、重复时间戳、跨段边界、非法目标列、列重排、文件不存在、CUDA 不可用、损坏检查点；每个场景返回确定的错误或行为。
- [ ] 测试夹具不含水泥数据（父任务 X1 检查脚本通过）。

## Out of Scope

- iTransformer、Informer（P2、P3）。
- 完整恢复训练状态（优化器/调度器/RNG 恢复在 P2 完成）。
- 多文件合并、长表、Parquet、`series_id` 多设备（P4 或之后）。

## Notes

- 本子任务进入规划时补写 `design.md` 与 `implement.md`。
