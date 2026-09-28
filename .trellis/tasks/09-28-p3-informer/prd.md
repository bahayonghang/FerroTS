# P3 Informer 复现

父任务：`09-28-ferrots-implementation`。计划书对应：第 4.3、4.4、9 节，第 15 节 P3。

## Goal

以 TSLib `models/Informer.py` 及其注意力、编码器文件（固定 SHA）为复现对象，在 Rust 中实现可训练的 ProbSparse Informer；GPU 路径不含隐式平方级内存或计算；在 ETT 与 `cement-twin` 上完成实验。

## 前置

- `09-28-p2-itransformer` 完成（夹具工具、完整恢复状态、注册表）。

## Requirements

- P3-R1 `PrefixSum` 自定义算子：CPU 线性扫描；CUDA 先实现正确的逐行扫描；反向为后缀累积和；前向与反向都不构造 n×n 矩阵。
- P3-R2 ProbSparse 注意力：`SampledQkScore`（不实体化广播后的五维张量，采样随机流可重建）→ `TopQuerySelect`（固定 tie 规则，batch=1、head=1 不丢维）→ 选中 query 与全部 key 计算注意力 → `ReplaceRows`（覆盖语义，被覆盖位置旧值梯度为 0，选中位置唯一性校验）。未选 query 使用官方规定的初始上下文（均值或 `PrefixSum`）。
- P3-R3 因果 mask 按 query 的原始时间位置构造。
- P3-R4 蒸馏层：CircularConv1d、BatchNorm（train/eval、运行均值/方差、momentum、缓冲区保存与恢复，缓冲区不进入优化器）、ELU、`MaxPool1d(kernel=3, stride=2)`（记录最大值位置，反向累加，重复最大值规则与参考实现一致）。
- P3-R5 Decoder：label 段取自已知历史，未来段为零占位；不从未来真实目标复制。
- P3-R6 保留官方实现细节：稀疏分数分母、mask 逻辑、注意力 dropout 位置；任何修正以工程变体单独命名。
- P3-R7 `informer_dense` 作为单独注册名用于消融，不作为 ProbSparse 复现版。
- P3-R8 GPU 路径不把 Q/K/V 复制到 CPU；关闭注意力矩阵导出。
- P3-R9 实验：`ett-lsf-v1` 4 × 4 × 3 个种子；`cement-twin-v1` × H ∈ {15,30,60} × 3 个种子；设备 `cuda:0`。

## Acceptance Criteria

- [ ] 每个自定义算子通过 CPU（f64）有限差分梯度测试与 CPU/CUDA 一致性测试。
- [ ] `PrefixSum` 在 n=1,000,000 时前向+反向峰值显存增量不超过输入大小的 4 倍（测量值写入测试输出）。
- [ ] 用共享的采样索引关闭随机性后，逐层输出与梯度与 TSLib 夹具差异在 atol=1e-5、rtol=1e-4 以内。
- [ ] BatchNorm 缓冲区在检查点保存与恢复后与连续训练逐位相同（CPU）。
- [ ] 复杂度测量：L ∈ {96, 192, 384, 768, 1536} 下记录编码器与解码器的峰值显存与耗时；ProbSparse 路径的注意力分数张量大小不随 L² 增长（测试断言）。
- [ ] `cuda:0` 上完整训练 ETT 16 组配置，结果满足父任务 D4 门槛（基准：本机运行官方 TSLib Informer 3 个种子）。
- [ ] `cement-twin-v1` 结果与 P1 基线、P2 iTransformer 并列报告。
- [ ] P3 任一验收项未通过时，能力表标记 Informer 为 preview，并在发布说明中写明。

## Out of Scope

- 分块并行 scan 等性能优化（按剖析结果在 P4/P5 决定）。
- Autoformer、FEDformer、TimesNet（计划书 P5 模型组）。

## Notes

- 本子任务进入规划时补写 `design.md` 与 `implement.md`。
