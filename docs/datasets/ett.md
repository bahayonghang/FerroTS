# 协议 `ett-lsf-v1`

ETT 4 个文件上的长期预测协议。P1–P3 的验收门槛（父任务 D4）在本协议上执行。

## 1. 来源版本

| 简称  | 仓库                                         | 提交 SHA                                   | 本地路径                                     |
| ----- | -------------------------------------------- | ------------------------------------------ | -------------------------------------------- |
| TSLib | https://github.com/thuml/Time-Series-Library | `4e938a1767106324dd753b2a44832bf870a0252e` | `tools/reference/.cache/Time-Series-Library` |
| iT    | https://github.com/thuml/iTransformer        | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `tools/reference/.cache/iTransformer`        |

`tools/reference/fetch_upstream.py` 按上表 SHA 获取代码。下文 `TSLib:<路径>:<行号>` 与 `iT:<路径>:<行号>` 指上表提交中的文件与行号。脚本路径前缀：TSLib 为 `scripts/long_term_forecast/ETT_script/`，iT 为 `scripts/multivariate_forecasting/ETT/`。

## 2. 文件

| 数据  | 路径                       | 采样间隔 | 行数（不含表头） | 协议使用行数 |
| ----- | -------------------------- | -------- | ---------------- | ------------ |
| ETTh1 | `data/ETT-small/ETTh1.csv` | 1 h      | 17,420           | 14,400       |
| ETTh2 | `data/ETT-small/ETTh2.csv` | 1 h      | 17,420           | 14,400       |
| ETTm1 | `data/ETT-small/ETTm1.csv` | 15 min   | 69,680           | 57,600       |
| ETTm2 | `data/ETT-small/ETTm2.csv` | 15 min   | 69,680           | 57,600       |

- 时间列 `date`，格式 `%Y-%m-%d %H:%M:%S`。
- 变量共 7 个，按文件列序：`HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`。`features = M` 时取 `date` 之后的全部列（TSLib:data_provider/data_loader.py:68-70）。
- 字节数与 SHA256 见 `manifest.toml`。
- TSLib 在本地文件不存在时从 Hugging Face 下载数据（TSLib:data_provider/data_loader.py:57-61）。运行官方代码前必须确认本地文件存在，并通过 `tools/datasets/verify_manifest.py`。

## 3. 切分

行号为数据行（不含表头）从 0 开始的半开区间。L 为输入长度。

| 切分 | ETTh                 | ETTm                 |
| ---- | -------------------- | -------------------- |
| 训练 | `[0, 8640)`          | `[0, 34560)`         |
| 验证 | `[8640 − L, 11520)`  | `[34560 − L, 46080)` |
| 测试 | `[11520 − L, 14400)` | `[46080 − L, 57600)` |

- 来源：ETTh 为 TSLib:data_provider/data_loader.py:63-64，ETTm 为 TSLib:data_provider/data_loader.py:163-164；iT 相同（iT:data_provider/data_loader.py:47-48、135-136）。
- 边界按“每月 30 天”计算：训练 12 个月，验证 4 个月，测试 4 个月。
- 验证段与测试段向前多取 L 行。这 L 行只作为第一个窗口的输入历史。
- ETTh 第 14,400 行及以后、ETTm 第 57,600 行及以后不使用。

## 4. 标准化

- 每个变量独立做 z-score：`x' = (x − μ) / σ`。μ 与 σ 只用训练段计算（TSLib:data_provider/data_loader.py:74-77）。验证段与测试段使用同一组 μ 与 σ。
- 实现为 `sklearn.preprocessing.StandardScaler`（TSLib:data_provider/data_loader.py:8）：σ 为总体标准差（除以 n）；σ = 0 的变量按 σ = 1 处理。
- 时间特征不做标准化。

## 5. 窗口

- 样本 i（段内索引，从 0 开始）：
  - 编码器输入：段内行 `[i, i + L)`。
  - 预测目标：段内行 `[i + L, i + L + H)`。
  - 解码器已知段（Informer 使用）：段内行 `[i + L − label_len, i + L)`。
  - 来源：TSLib:data_provider/data_loader.py:101-112。
- 每段样本数 = 段长 − L − H + 1（TSLib:data_provider/data_loader.py:114-115）。
- L = 96 时各切分样本数（P1 窗口数单元测试使用下表）：

| H   | ETTh 训练 | ETTh 验证 | ETTh 测试 | ETTm 训练 | ETTm 验证 | ETTm 测试 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
| 96  | 8,449     | 2,785     | 2,785     | 34,369    | 11,425    | 11,425    |
| 192 | 8,353     | 2,689     | 2,689     | 34,273    | 11,329    | 11,329    |
| 336 | 8,209     | 2,545     | 2,545     | 34,129    | 11,185    | 11,185    |
| 720 | 7,825     | 2,161     | 2,161     | 33,745    | 10,801    | 10,801    |

## 6. 长度参数

| 参数      | 值                  | 来源                                                                                                       |
| --------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| L         | 96                  | TSLib:DLinear_ETTh1.sh:5,26,47,68；TSLib:Informer_ETTh1.sh:5,26,47,68；iT:iTransformer_ETTh1.sh:5,24,43,62 |
| label_len | 48                  | TSLib 两个脚本同上行；iT 脚本不设置，取默认值 48（iT:run.py:35），iTransformer 不使用该值                  |
| H         | {96, 192, 336, 720} | 同 L 的来源行                                                                                              |
| features  | `M`（7 进 7 出）    | 同 L 的来源行                                                                                              |

## 7. 时间特征

- `--embed` 默认 `timeF`（TSLib:run.py:72；iT:run.py:53），因此 `timeenc = 1`（TSLib:data_provider/data_factory.py:24；iT:data_provider/data_factory.py:18）。
- `--freq` 默认 `h`（TSLib:run.py:32；iT:run.py:29）。两个仓库的长期预测脚本都不设置 `--freq`（在 `scripts/` 下搜索 `--freq` 无结果）。因此 ETTm（15 min）也使用 `h`，不含分钟特征。
- `timeenc = 0` 时 ETTm 才增加 `minute // 15` 列（TSLib:data_provider/data_loader.py:188-189）。本协议不使用该分支。
- `freq = h` 对应 4 个特征，顺序固定（TSLib:utils/timefeatures.py:106；iT 同行）：

| 列  | 特征       | 公式                              | 来源（两个仓库相同）     |
| --- | ---------- | --------------------------------- | ------------------------ |
| 0   | HourOfDay  | `hour / 23 − 0.5`                 | utils/timefeatures.py:52 |
| 1   | DayOfWeek  | `dayofweek / 6 − 0.5`（周一为 0） | utils/timefeatures.py:59 |
| 2   | DayOfMonth | `(day − 1) / 30 − 0.5`            | utils/timefeatures.py:66 |
| 3   | DayOfYear  | `(dayofyear − 1) / 365 − 0.5`     | utils/timefeatures.py:73 |

- 特征值范围为 `[−0.5, 0.5]`。
- 时间嵌入输入维度：`freq_map['h'] = 4`（TSLib:layers/Embed.py:100；iT:layers/Embed.py:98）。
- iTransformer 把 4 个时间特征各作为 1 个变量 token，与 7 个变量拼接后输入编码器（iT:layers/Embed.py:140）；输出只保留前 7 个变量。

## 8. 训练设置

两个仓库的默认值相同，脚本未覆盖时使用下表：

| 参数       | 值                                                            | TSLib 来源                                                 | iT 来源                                                            |
| ---------- | ------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 优化器     | Adam，β 与 eps 为 torch 默认值，无权重衰减                    | exp/exp_long_term_forecasting.py:34                        | experiments/exp_long_term_forecasting.py:32                        |
| 损失       | MSE（标准化空间）                                             | exp/exp_long_term_forecasting.py:38                        | experiments/exp_long_term_forecasting.py:36                        |
| 学习率     | 1e-4                                                          | run.py:93                                                  | run.py:65                                                          |
| 学习率调整 | `type1`：第 e 轮（从 1 开始）学习率 = lr0 × 0.5^(e−1)         | utils/tools.py:14-15                                       | utils/tools.py:13-14                                               |
| 最大轮数   | 10                                                            | run.py:90                                                  | run.py:62                                                          |
| 早停       | 验证损失连续 3 轮不下降时停止；测试前加载验证损失最低的检查点 | run.py:92；exp/exp_long_term_forecasting.py:88,156,163-164 | run.py:64；experiments/exp_long_term_forecasting.py:94,172,181-182 |
| 批大小     | 32                                                            | run.py:91                                                  | run.py:63                                                          |
| 随机种子   | TSLib 固定 2021；iT 固定 2023                                 | run.py:10-13                                               | run.py:9-12                                                        |

- 两个仓库的种子写在 `run.py` 中，命令行无法修改。TSLib 的 `--seed`（run.py:118）只用于数据增强（TSLib:utils/augmentation.py:334,352）。3 种子基准由 `tools/reference/run_tslib.py` 替换 `fix_seed` 的值后运行。

## 9. 数据加载

| 切分 | TSLib                                  | iT                                  |
| ---- | -------------------------------------- | ----------------------------------- |
| 训练 | 打乱；`drop_last = False`；批大小 32   | 打乱；`drop_last = True`；批大小 32 |
| 验证 | 打乱；`drop_last = False`；批大小 32   | 打乱；`drop_last = True`；批大小 32 |
| 测试 | 不打乱；`drop_last = False`；批大小 32 | 不打乱；批大小 1                    |

- 来源：TSLib:data_provider/data_factory.py:26-28；iT:data_provider/data_factory.py:20-34。
- 两个仓库的测试都覆盖全部测试样本。
- iT 的验证集丢弃最后不足一个批次的样本。该差异只影响早停，不影响测试指标。

## 10. 指标

- 主指标在标准化空间计算：`--inverse` 默认关闭（TSLib:run.py:41；iT:run.py:81）。
- 模型输出取最后 H 步与全部 7 个变量（`features = M` 时 `f_dim = 0`，TSLib:exp/exp_long_term_forecasting.py:199-201）。
- `MSE = mean((ŷ − y)²)`，`MAE = mean(|ŷ − y|)`。平均范围为全部测试样本 × H × 7 个元素，元素等权（TSLib:utils/metrics.py:14-19；TSLib:exp/exp_long_term_forecasting.py:255）。
- 原始单位指标：对 ŷ 与 y 做逆标准化后用同一公式计算，作为附加报告。
- 父任务 D4 门槛只作用于标准化空间 MSE 与 MAE。

## 11. 模型配置

表中未列出的参数取第 8 节与各自 `run.py` 默认值。“派生”表示上游没有该数据集的脚本，配置由 ETTh1 脚本替换 `--data` 与 `--data_path` 得到。

### 11.1 DLinear（TSLib）

| 数据                | 配置                                                              | 来源                              |
| ------------------- | ----------------------------------------------------------------- | --------------------------------- |
| ETTh1               | L = 96，H 见第 6 节，`enc_in = 7`，`moving_avg = 25`（run.py:66） | TSLib:DLinear_ETTh1.sh:5,26,47,68 |
| ETTh2、ETTm1、ETTm2 | 同 ETTh1                                                          | 派生                              |

- DLinear 只读取 `seq_len`、`pred_len`、`moving_avg`、`enc_in`（TSLib:models/DLinear.py:17-26）。脚本中的 `e_layers`、`d_layers`、`factor` 对 DLinear 无作用。
- `individual` 取构造函数默认值 False（TSLib:models/DLinear.py:12）。命令行 `--individual`（run.py:151）不传入模型。
- DLinear 不使用时间特征。

### 11.2 Informer（TSLib）

| 数据                | 配置                                                                                                                                                                                                                      | 来源                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| ETTh1               | L = 96，label_len = 48，`e_layers = 2`，`d_layers = 1`，`factor = 3`，`enc_in = dec_in = c_out = 7`；`d_model = 512`、`n_heads = 8`、`d_ff = 2048`、`dropout = 0.1`、`activation = gelu`、`distil = True`（run.py:61-74） | TSLib:Informer_ETTh1.sh:5,26,47,68 |
| ETTh2、ETTm1、ETTm2 | 同 ETTh1                                                                                                                                                                                                                  | 派生                               |

### 11.3 iTransformer（iT）

| 数据  | H        | `d_model` | `d_ff` | `e_layers` | 来源                                |
| ----- | -------- | --------- | ------ | ---------- | ----------------------------------- |
| ETTh1 | 96、192  | 256       | 256    | 2          | iT:iTransformer_ETTh1.sh:5,24       |
| ETTh1 | 336、720 | 512       | 512    | 2          | iT:iTransformer_ETTh1.sh:43,62      |
| ETTh2 | 全部     | 128       | 128    | 2          | iT:iTransformer_ETTh2.sh:5,24,43,62 |
| ETTm1 | 全部     | 128       | 128    | 2          | iT:iTransformer_ETTm1.sh:5,24,43,62 |
| ETTm2 | 全部     | 128       | 128    | 2          | iT:iTransformer_ETTm2.sh:5,24,43,62 |

- 其余参数取 iT 默认值：`n_heads = 8`（run.py:43），`dropout = 0.1`（run.py:52），`activation = gelu`（run.py:55），`use_norm = True`（run.py:86），`class_strategy = projection`（run.py:82）。
- `factor` 传入 `FullAttention`（iT:model/iTransformer.py:30），全注意力不使用该值。

## 12. 参考数值

公开数值只作参考。验收基准为父任务 D4：本机官方 PyTorch 代码 3 种子均值。

来源：

| 简称 | 文献                                                                                  | 表格                                                                   | 输入长度 | 读取方式与日期                          |
| ---- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------- | --------------------------------------- |
| P-iT | Liu et al., iTransformer, ICLR 2024, arXiv:2310.06625v4                               | Table 10（长期预测全部结果）；Table 5（iTransformer 5 个种子的均值 ± 标准差） | 96       | arxiv.org HTML，2026-09-28              |
| P-TN | Wu et al., TimesNet, ICLR 2023, arXiv:2210.02186                                      | Table 13（长期预测全部结果）                                           | 96       | ar5iv HTML（页面未标版本号），2026-09-28 |

- iTransformer 取 P-iT Table 10。
- DLinear 取 P-TN Table 13；P-iT Table 10 的 DLinear 列与之相同。P-TN 对 DLinear 标注 ∗：作者使用官方代码，只修改输入与输出长度。
- Informer 取 P-TN Table 13（输入长度 96，由 TimesNet 作者运行）。

标准化空间 MSE / MAE：

| 数据  | H   | iTransformer  | DLinear       | Informer      |
| ----- | --- | ------------- | ------------- | ------------- |
| ETTh1 | 96  | 0.386 / 0.405 | 0.386 / 0.400 | 0.865 / 0.713 |
| ETTh1 | 192 | 0.441 / 0.436 | 0.437 / 0.432 | 1.008 / 0.792 |
| ETTh1 | 336 | 0.487 / 0.458 | 0.481 / 0.459 | 1.107 / 0.809 |
| ETTh1 | 720 | 0.503 / 0.491 | 0.519 / 0.516 | 1.181 / 0.865 |
| ETTh2 | 96  | 0.297 / 0.349 | 0.333 / 0.387 | 3.755 / 1.525 |
| ETTh2 | 192 | 0.380 / 0.400 | 0.477 / 0.476 | 5.602 / 1.931 |
| ETTh2 | 336 | 0.428 / 0.432 | 0.594 / 0.541 | 4.721 / 1.835 |
| ETTh2 | 720 | 0.427 / 0.445 | 0.831 / 0.657 | 3.647 / 1.625 |
| ETTm1 | 96  | 0.334 / 0.368 | 0.345 / 0.372 | 0.672 / 0.571 |
| ETTm1 | 192 | 0.377 / 0.391 | 0.380 / 0.389 | 0.795 / 0.669 |
| ETTm1 | 336 | 0.426 / 0.420 | 0.413 / 0.413 | 1.212 / 0.871 |
| ETTm1 | 720 | 0.491 / 0.459 | 0.474 / 0.453 | 1.166 / 0.823 |
| ETTm2 | 96  | 0.180 / 0.264 | 0.193 / 0.292 | 0.365 / 0.453 |
| ETTm2 | 192 | 0.250 / 0.309 | 0.284 / 0.362 | 0.533 / 0.563 |
| ETTm2 | 336 | 0.311 / 0.348 | 0.369 / 0.427 | 1.363 / 0.887 |
| ETTm2 | 720 | 0.412 / 0.407 | 0.554 / 0.522 | 3.379 / 1.338 |

- P-iT Table 5 报告 iTransformer 在 ETTh2 上 5 个种子的标准差：MSE 0.001–0.004，MAE 0.001–0.002。
- 论文数值与本机官方代码结果之间的差异原因未查明（论文未给出运行环境与种子）。父任务 D4 门槛不使用上表。

## P1 已验证的训练时序

固定官方实现的 type1 在 epoch 结束时调用。第 1、2 个 epoch 使用初始学习率 1e-4，第 3 个 epoch 使用 5e-5，随后逐轮减半。早停后不再推进下一轮。FerroTS 保留有效元素加权、固定顺序验证及训练中不访问 test 的契约。

初始化和训练采样已按用户批准对齐固定 PyTorch CPU RNG，详见 docs/models/dlinear.md。完整本机比较及 32 项 D4 判定见 docs/results/p1-dlinear-ett.md；公开论文表格仅用于背景参考。
