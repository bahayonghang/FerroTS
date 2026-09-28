# P2 iTransformer 复现

父任务：`09-28-ferrots-implementation`。计划书对应：第 8 节、第 10.1 节、第 14.1 节，第 15 节 P2。

## Goal

在 Rust 中实现与官方 iTransformer（thuml/iTransformer，固定 SHA）语义一致的模型，逐层前向、梯度与单步更新对齐；提供完整恢复训练状态；在 ETT 与 `cement-twin` 上完成多种子实验。

## 前置

- `09-28-p1-data-train-loop` 完成（时间特征、窗口、Trainer、基本检查点）。

## Requirements

- P2-R1 模型结构：窗口归一化（均值 detach、总体方差、epsilon 与官方相同）→ 倒置嵌入（变量 token 与时间特征 token 拼接）→ Post-Norm 编码器（注意力、FFN、Dropout 位置、GELU 形式与官方相同）→ 投影到 H → 剔除时间特征 token → 反归一化。FFN 的 kernel=1 Conv1d 以 Linear 实现，权重与偏置映射正确。
- P2-R2 权重映射：PyTorch `state_dict` 名称 ↔ safetensors 名称的映射表；`tools/reference/` 从固定 SHA 的官方代码导出权重、逐层输出与梯度夹具（夹具只用合成数据或 ETT 生成）。
- P2-R3 形状约束：权重绑定 L、H 与变量数；加载时核对，不匹配时报错。
- P2-R4 配置校验：拒绝 Informer 专用参数（如 decoder 参数）；配置 future-known 变量时拒绝；`d_model % n_heads != 0` 报错。
- P2-R5 完整恢复状态：模型、优化器、缓冲区、trainer 状态、调度器状态、RNG 状态、scaler、数据 schema、resolved config、环境信息，按计划书第 10.1 节目录保存；在 optimizer-step 边界恢复。
- P2-R6 `ferrots resume --checkpoint <dir>`；加载时核对变量列顺序、shape、模型版本、数据协议、缺失参数。
- P2-R7 模型注册与能力表：名称、版本、来源仓库/SHA/文件、参数 schema、任务类型、时间特征支持、future-known 支持、mask 支持、设备与精度、复现测试状态。
- P2-R8 实验：`ett-lsf-v1` 4 个数据集 × 4 个预测长度 × 3 个种子；`cement-twin-v1` × H ∈ {15,30,60} × 3 个种子；设备为 `cuda:0`，另在 ETTh1 上做一次 CPU 对照。

## Acceptance Criteria

- [ ] 固定权重与输入下，嵌入、每层注意力输出、FFN 输出、归一化输出、最终输出与官方夹具的差异在 atol=1e-5、rtol=1e-4 以内（CPU 与 CUDA）。
- [ ] 全部可训练参数的梯度与夹具差异在同一容差以内；测试列出每个参数并断言梯度存在且不全为 0（使用非退化输入）。
- [ ] 一次 Adam 更新后的参数与夹具差异在 atol=1e-6 以内。
- [ ] 恢复测试（Dropout 开启）：连续 2N 步与“N 步 → 保存 → resume → N 步”的参数在 CPU 上逐位相同，在 CUDA 上差异在 atol=1e-6 以内。
- [ ] 损坏检查点（缺文件、校验和不符、变量顺序变化）加载时报错。
- [ ] ETT 16 组配置的标准化 MSE/MAE 满足父任务 D4 门槛（基准：本机运行官方 iTransformer 3 个种子）。
- [ ] `cement-twin-v1` 结果与 P1 基线并列报告（分变量原始单位 MAE/RMSE，均值 ± 标准差）。
- [ ] 记录每组实验的吞吐（样本/秒）、峰值显存、每 epoch 耗时；不设性能门槛。

## Out of Scope

- `itransformer_softsensor` 与任意目标软测量（父任务 D2，已移出范围）。
- FlashAttention、低精度（P5）。

## Notes

- 本子任务进入规划时补写 `design.md` 与 `implement.md`。
