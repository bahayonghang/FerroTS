# 后续任务执行顺序

本文件是未来实施计划，未勾选步骤不表示已执行。开发分支为 `dev`；父任务只协调，不直接承载模型实施。

1. [x] 将 P2 已提交结果通过合并提交纳入 `main`，清理对应 feat 分支，从 `main` 创建 `dev`。
2. [x] 保留 P2 `in_progress`，更新当前 Linux CUDA 状态与剩余验收条款；不把旧 Windows executor 状态作为当前进度。
3. [x] 补齐 P3 Informer 的 PRD、设计、实施计划及上下文。
4. [x] 建立独立 WaveForM 任务与一手来源证据；明确许可、协议、可导 IDWT 和长度边界门。
5. [x] 按可独立验证的增量细化 P4；候选模型与发布后 P5 不阻塞首个比较报告。
6. [x] 用户授权按顺序实施并启动 P3；复用已有 P2 技术交付，P2 剩余 CPU/设备条款继续保留，未归档或追认为通过。
7. [ ] 按 P3 `implement.md` 完成算子、持久化、模型闭环、105 次 CUDA 正式验收、独立复核与提交。
8. [ ] 按 WaveForM `implement.md` 先完成来源/许可/协议/算子门，再审批正式训练预算并运行 129 次矩阵。
9. [ ] 按 P4 `implement.md` 交付 compare/报告、扩展数据矩阵、回测与搜索；新增模型分别审批。
10. [ ] 各阶段更新能力表与 `docs/results/`；第一版集成评审后才考虑 P5。

## 本次规划验证

- `task.py validate` 检查父任务、P2、P3、P4、WaveForM 的上下文路径；逐项检查验收条款是否有设计机制和实施步骤。
- 检查矩阵算术、任务父子关系、分支字段与状态；未启动任务保持 `planning`。
- `bash tools/check_private_data.sh` 与 `git diff --check`。
- 确认运行时源码摘要等于现有 Linux suite；本轮仅规划变化不重跑已通过的训练或质量矩阵。
- 本地提交后验证 `main`/`dev` 与远端引用，并确认对应 feat 引用不存在。

## 后续实施验证

使用 `docs/environment.md` 的 Linux CUDA 命令与固定 Rust/CUDA 环境。各代码增量运行相关 CUDA 行为测试与 workspace fmt、CUDA Clippy/测试；训练和模型行为检查默认 CUDA。约三 CPU epoch 仅限 CUDA 失败后必要诊断，保留原错误且不替代验收。冻结新 suite 后，工具或运行时变化必须另建身份，不覆盖已接受证据。

当前 P3 开发门 AC1–AC6 已通过，正式0/105；下一关是 P3 implement.md 步骤6的实测预算审批。WaveForM、P4仍按顺序等待。
