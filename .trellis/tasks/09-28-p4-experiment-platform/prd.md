# P4 通用实验平台

父任务：`09-28-ferrots-implementation`。计划书对应：第 7 节 P4 模型组、第 10.2 节、第 12 节，第 15 节 P4。

## Goal

所有模型共用数据协议与评价；提供回测、结果比较、超参数搜索与静态报告；扩充不同归纳偏置的模型；结果可追溯。

## 前置

- `09-28-p3-informer` 完成或以 preview 状态结束。

## Requirements

- P4-R1 rolling-origin 回测（`ferrots backtest`）：每折重新拟合 scaler 与训练状态；折边界落在段边界上；不混合不同 `forecast_origin` 对同一目标时间的预测。
- P4-R2 `ferrots compare --runs ...`：输出同协议下多个 run 的指标对照表；协议或数据指纹不同时拒绝比较。
- P4-R3 超参数搜索：网格与随机搜索；记录预算、种子、配置；先用验证集选择，再独立评估测试集。
- P4-R4 多 GPU 调度：一张卡一个独立实验；文档不称其为分布式训练。
- P4-R5 静态报告：Markdown/HTML，包含指标表、分变量与分步长统计、预测曲线；水泥数据报告不嵌入原始数据行。
- P4-R6 新模型：PatchTST、TCN、TSMixer 或 Transformer；每个模型附结构测试、梯度测试与能力表条目。
- P4-R7 性能剖析：CSV 解析、窗口打包、主机→设备传输、前向、反向、更新、验证、写盘分别计时；GPU 计时正确同步。
- P4-R8 结果索引：JSONL/CSV 产物可选索引到 SQLite。
- P4-R9 扩展数据集实验（父任务 D5）：按 `custom-lsf-v1` 在 Weather、Exchange、ECL、Traffic、ILI 上运行 DLinear、iTransformer、Informer 与本任务新增模型，3 个种子；报告结果并列公开参考数值，不设门槛。

## Acceptance Criteria

- [ ] 回测在 ETTh1 与 `cement-twin-v1` 上运行；每折 scaler 只用该折训练段拟合（测试断言）。
- [ ] `compare` 对协议不同的两个 run 返回错误；对同协议 run 输出对照表。
- [ ] 搜索任务的每个试验都有 resolved config、种子与验证指标记录；测试集只在最终配置上评估一次。
- [ ] 每个新增模型在 `ett-lsf-v1`（至少 ETTh1 的 4 个预测长度）与 `cement-twin-v1` 上完成训练与评价。
- [ ] 剖析报告给出各阶段耗时占比与峰值显存。
- [ ] P4-R9 的 5 个数据集结果写入 `docs/results/`，每组含均值 ± 标准差与公开参考数值来源。
- [ ] `docs/results/` 汇总表更新（父任务验收项）。

## Out of Scope

- 分布式数据并行、桌面 UI。
- Autoformer、FEDformer、TimesNet（计划书 P5 模型组，另立任务）。

## Notes

- 本子任务进入规划时补写 `design.md` 与 `implement.md`。
