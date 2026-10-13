# P3 Informer 复现

父任务：`09-28-ferrots-implementation`。开发分支 `dev`，状态 `in_progress`。

## Goal

复现固定 TSLib `4e938a1767106324dd753b2a44832bf870a0252e` 的长期预测 Informer，在现有 Rust 运行时实现可训练 ProbSparse、蒸馏与 decoder，在 ETT 与私有水泥数据上完成可追溯 CUDA 对照。

## 前置与约束

依赖 P2 已交付的模型派发、数据/训练合同、完整检查点与 Linux CUDA 技术证据，不重复其已通过矩阵。P2 整体仍为 in_progress，原 CPU、设备及文字条款单独跟踪，不因启动 P3 自动关闭。先审批本计划再 task.py start。

遵循 AGENTS 的 CUDA 优先要求：新模型行为与恢复验收在 CUDA 上执行，不安排常规 CPU 训练或 CPU-only 验收。旧 P3 规划中的 CPU f64/CPU bitwise 要求不作为本轮新 CUDA 计划的完成条件；这不追认 P2 未执行的 CPU 条款为通过。

## Requirements

- P3-R1：PrefixSum 前后向为线性空间；梯度为后缀累加，禁止构造三角 n×n 张量。
- P3-R2：ProbSparse 保留固定上游的采样分数、top-query、初始上下文及覆盖语义；encoder、decoder self/cross attention 均按该上游使用 ProbAttention。top-k 索引不求导，被选中的 Q/K/V 路径保持梯度。
- P3-R3：采样、score、mask 不实体化完整平方级张量或广播后的五维展开；因果 mask 直接从选中 query 的原位置构造。不将 Q/K/V 拷贝到 CPU，不输出完整注意力图。
- P3-R4：蒸馏包含 CircularConv1d、BatchNorm、ELU、重叠 MaxPool1d，前后向及 padding 与固定上游一致；BatchNorm buffer 不进入优化器。
- P3-R5：decoder 只从已知历史尾部和未来零值构造；监督标签不进入 ModelInput。时间特征对应拼接；初版要求 gap=0、0<label_len≤L，不支持 future-known 数值。
- P3-R6：训练、验证、预测的采样身份明确且可恢复；验证不消耗训练随机流。BatchNorm buffer 参与 snapshot、保存、加载、freeze、恢复和推理。
- P3-R7：复现版保持固定上游行为，包括未应用的 ProbAttention 权重 dropout；修正或 dense 消融单独命名，informer_dense 不充当复现版。本轮不强制实现 dense 消融。
- P3-R8：接入现有配置、静态派发、CLI、能力表及版本化检查点；不增加另一套训练框架。实验使用 FP32 与种子 2021/2022/2023。
- P3-R9：ETT 4数据集×4H×3种子，官方与Rust各48次；水泥 H15/30/60×3种子9次，共105次正式CUDA训练。一卡一个实验，cuda:0/cuda:1身份逐运行记录。
- P3-R10：报告包含复现结果、全部18个水泥变量原始单位指标、性能与失败项；数据和产物留在忽略目录，不生成私有原始行夹具。

## Acceptance Criteria

- [x] P3-AC1：CUDA 上新增算子、逐层输出和非退化真实损失梯度与固定权重/采样索引的官方夹具在 atol=1e-5、rtol=1e-4 内；检查预期参数梯度。top-k并列、B/head=1与L=1单独测试，不对离散索引作数值梯度比较。
- [x] P3-AC2：PrefixSum 在 n=1,000,000、FP32 时前向+反向峰值显存增量≤输入字节数4倍；口径见设计，输出实测峰值，不用低频全卡采样替代。
- [x] P3-AC3：L={96,192,384,768,1536} 时记录 encoder 与 decoder self/cross 的 score/mask shape、峰值和耗时；断言无完整Lq×Lk score/mask，采样路径不复制Q/K/V到主机。
- [x] P3-AC4：CUDA 独立进程连续2N步与N步→保存→恢复→N步，在dropout、稀疏采样、BatchNorm开启时，权重、Adam矩、浮点buffer和预测在atol=1e-6、rtol=0内；整数buffer、计数、随机状态精确一致。
- [x] P3-AC5：freeze/evaluate不改变训练权重、BatchNorm buffer或训练随机状态；缺失/额外/错shape buffer检查点整体拒绝。历史空buffer模型的现有检查点测试保持通过，不新增迁移层。
- [x] P3-AC6：decoder未来数值全零；保持ModelInput不变而更换监督标签，固定采样身份预测不变。训练、评价、CLI预测均输出[B,H,C]，B=1和尾批可用。
- [ ] P3-AC7：官方48与Rust48的ETT完整覆盖、正常退出和身份检查通过，32项MSE/MAE均满足父任务D4；ddof=0，不放宽门槛。
- [ ] P3-AC8：水泥9/9完成，18变量×3H×MAE/RMSE共108格均值/总体标准差，与P1/P2并列；H60/seed2021三变量18,574,560行预测独立复算6项误差均<1e-6。
- [ ] P3-AC9：105条性能记录、运行来源和正常退出可核验；模型卡、CLI、registry、配置和检查点身份一致。
- [ ] P3-AC10：正式门未全通过时保持preview；最终fmt、CUDA Clippy、CUDA workspace测试、参考工具测试、私有数据检查与独立评审通过，无隐瞒失败/忽略项。

## Out of Scope

CPU正式训练、低精度、FlashAttention、分布式训练、并行scan优化、Autoformer/FEDformer/TimesNet、新数据协议和软测量任务。
