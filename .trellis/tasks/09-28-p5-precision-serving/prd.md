# P5 低精度与部署服务

父任务：`09-28-ferrots-implementation`。计划书对应：第 4.5 节、第 12 节、第 13 节，第 15 节 P5。

## Goal

在第一版发布后，按独立验收逐项增加低精度训练、可训练融合算子与 HTTP 预测服务；每项改动不改变复现版默认配置。

## 前置

- 第一版发布（P0–P3 完成，P4 的回测与比较可用）。

## Requirements

- P5-R1 PrecisionPolicy：BF16/FP16 计算，FP32 master weights 与优化器状态，归一化与损失在 FP32；FP16 动态 loss scaling、非有限梯度检测与跳步状态保存。
- P5-R2 可训练融合 LayerNorm/Softmax：自定义 CUDA 前向 + 反向，与 `train_ops` 数值一致后才可启用。
- P5-R3 FlashAttention 反向（可选）：单独验收后才可进入训练路径。
- P5-R4 pinned-memory 与传输重叠：以 profiler 结果证明重叠后启用。
- P5-R5 HTTP 预测服务：加载冻结权重、scaler、数据协议；输入 schema 不兼容时拒绝请求；默认本地日志、无遥测。

## Acceptance Criteria

- [ ] BF16/FP16 在 ETTh1 与 `cement-twin-v1` 上的指标与 FP32 对照报告；FP16 在注入溢出时跳步并记录。
- [ ] 融合算子的前向与梯度与 `train_ops` 差异在记录的容差内；速度与显存对照写入报告。
- [ ] 服务对 schema 不匹配的请求返回错误码；对合法请求的输出与 `ferrots predict` 一致。

## Out of Scope

- 云端实验服务、强制遥测。

## Notes

- 第一版发布前不进入规划。进入规划时补写 `design.md` 与 `implement.md`。
