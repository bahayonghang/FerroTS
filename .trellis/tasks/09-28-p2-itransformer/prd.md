# P2 iTransformer 复现

2026-09-29 用户后续指示：验证优先 CUDA，不安排常规 CPU 测试/训练；仅在 CUDA 报错时按需运行约 3 个 epoch 的 CPU 诊断。原 CPU 完整对照保留为未执行要求，不记为通过；105 次 CUDA 正式矩阵及数值容差保持。详见 implement.md 的用户验证策略更新。
父任务：`09-28-ferrots-implementation`。计划书对应：第 8 节、第 10.1 节、第 14.1 节，第 15 节 P2。

## Goal

在 Rust 中实现与官方 iTransformer（thuml/iTransformer，固定 SHA）语义一致的模型，逐层前向、梯度与单步更新对齐；提供完整恢复训练状态；在 ETT 与 `cement-twin` 上完成多种子实验。

## 前置

- `09-28-p1-data-train-loop` 已完成并归档（时间特征、窗口、Trainer、基本检查点）；归档提交 `34a11fa`，journal 提交 `1f1b9ec`。正式验收与原失败证据保留在归档任务及原 runs 目录。
- 当前分支 `feat/p2-itransformer` 从 P1 收尾后的 `1f1b9ec` 创建。用户已批准实施，P2 于 2026-09-29 经 task.py start 进入 in_progress；已加载 trellis-before-dev 并按顺序实施。

## 执行约束

- 主会话内联执行，不派生子代理。每步完成相关验证再继续。
- 所有命令进程设置 RUSTUP_AUTO_INSTALL=0，固定 Rust 1.98.0；直接 Cargo 同时固定 RUSTC/RUSTDOC。Windows 原生 CUDA、MSVC、PATH、静态 CRT 复用 docs/environment.md 与 tools/validation/p1_env.ps1。
- 系统级安装先询问，保留 Rust 1.78.0；未经同意不删除文件/目录，不推送，不配置远程 CI。
- Python 复用 tools/reference 的 uv 环境；联网 HTTPS_PROXY=http://127.0.0.1:7897；PYTHONIOENCODING=utf-8。
- 保留 Exchange 7,588 行修正、已批准的固定 PyTorch RNG 对齐和 cement-twin-v1 四项协议调整。不改既定种子、超参数或数值门槛，不用局部测试替代完整正式矩阵。
- data/、runs/不提交；夹具仅合成/公共数据，报告仅统计量；提交前执行 bash tools/check_private_data.sh。

## Requirements

- P2-R1 模型结构：窗口归一化（均值 detach、总体方差、epsilon 与官方相同）→ 倒置嵌入（变量 token 与时间特征 token 拼接）→ Post-Norm 编码器（注意力、FFN、Dropout 位置、GELU 形式与官方相同）→ 投影到 H → 剔除时间特征 token → 反归一化。FFN 的 kernel=1 Conv1d 以 Linear 实现，权重与偏置映射正确。
- P2-R2 权重映射：PyTorch `state_dict` 名称 ↔ safetensors 名称的映射表；`tools/reference/` 从固定 SHA 的官方代码导出权重、逐层输出与梯度夹具（夹具只用合成数据或 ETT 生成）。
- P2-R3 形状约束：权重绑定 L、H 与变量数；加载时核对，不匹配时报错。
- P2-R4 配置校验：拒绝 Informer 专用参数（如 decoder 参数）；配置 future-known 变量时拒绝；`d_model % n_heads != 0` 报错。
- P2-R5 完整恢复状态：模型、优化器、缓冲区、trainer 状态、调度器状态、RNG 状态、scaler、数据 schema、resolved config、环境信息，按计划书第 10.1 节目录保存；在 optimizer-step 边界恢复。
- P2-R6 `ferrots resume --checkpoint <dir>`；加载时核对变量列顺序、shape、模型版本、数据协议、缺失参数。
- P2-R7 模型注册与能力表：名称、版本、来源仓库/SHA/文件、参数 schema、任务类型、时间特征支持、future-known 支持、mask 支持、设备与精度、复现测试状态。
- P2-R8 实验：`ett-lsf-v1` 4 个数据集 × 4 个预测长度 × 3 个种子，官方与 Rust 各48次；`cement-twin-v1` × H ∈ {15,30,60} × 3 个种子，共9次；设备为 `cuda:0`，另在 ETTh1/H96/seed2021 上做一次完整 CPU 对照。种子固定2021/2022/2023，总计106次完整训练。

## Acceptance Criteria

- [x] P2-AC1：固定权重与输入下，嵌入、每层注意力输出、FFN 输出、归一化输出、最终输出与官方夹具的差异在 atol=1e-5、rtol=1e-4 以内（CPU 与 CUDA）。2026-10-12 Linux quality 日志中 CPU 与 CUDA 全参数测试均通过；正式数值门 9 例为 CUDA。
- [x] P2-AC2：全部可训练参数的梯度与夹具差异在同一容差以内；测试列出每个参数并断言梯度存在且不全为 0（使用非退化输入；连接性夹具定义见 Technical Notes）。
- [x] P2-AC3：保留全参数真实 MSE 梯度门；用同一份官方真实 MSE 梯度输入两端 Adam，比较全部参数、一二阶矩和 step，atol=1e-6、rtol=0。各后端独立反向后更新的全参数差异单独记录为诊断，不作为 AC3 通过条件。2026-09-29 用户明确批准，证据见 research/numerical-gate-diagnosis.md 第 5 节。
- [ ] P2-AC4：恢复测试（Dropout 开启）：连续 2N 步与“N 步 → 保存 → resume → N 步”的参数在 CPU 上逐位相同，在 CUDA 上差异在 atol=1e-6 以内。CUDA 10/10 已在本套件通过；本 runtime source 的 CPU 逐位条款未重跑。
- [x] P2-AC5：损坏检查点（缺文件、校验和不符、变量顺序变化）加载时报错。
- [x] P2-AC6：ETT 16 组配置的标准化 MSE/MAE 满足父任务 D4 门槛（基准：本机运行官方 iTransformer 3 个种子），共32项双侧判定。使用总体标准差ddof=0，完整精度比较，原门槛不变。Linux 套件 32/32。
- [x] P2-AC7：`cement-twin-v1` 结果与 P1 基线并列报告（全部18变量原始单位 MAE/RMSE，P2三个种子均值 ± 总体标准差；既有P1单种子标明n=1，标准差不适用）。P1 数字来自已提交表，n=1。
- [x] P2-AC8：记录每组实验的吞吐（样本/秒）、峰值显存、每 epoch 耗时；标明显存计数来源、采样周期及测量边界，不设性能门槛。105 行均在 `docs/results/p2-itransformer-performance.md`。

## Out of Scope

- `itransformer_softsensor` 与任意目标软测量（父任务 D2，已移出范围）。
- FlashAttention、低精度（P5）。

## Technical Notes

- 固定官方代码的 key-projection bias 在最终 MSE 下可有数学零梯度；微小浮点残差不能证明连接性。全部真实 MSE 梯度保留原容差，并用相同中间投影探针证明连接性。AC3 按 2026-09-29 批准的同梯度优化器门执行；不修改生产 MSE、梯度、key bias 或 Adam 超参数。
- 完整恢复目录保留P0的optimizer-state.json，并增加计划书第10.1节要求的其余状态；v1五文件bundle继续只用于推理。
- 设计、顺序步骤与上下文分别记录在design.md、implement.md、implement.jsonl和check.jsonl。规划文件就绪不代表已获实施批准。
