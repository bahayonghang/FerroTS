# 后续模型与实验路线

## 当前基线与边界

`main` 已通过本地合并提交 `5b937b06c3143a58cbf7144e48df26acbf57db7d` 接收 P2 Linux CUDA 交付，开发规划在 `dev`。本地及远端 `feat/p2-itransformer` 已删除。分支合并不改变任务验收状态。

P2 于 2026-10-12 冻结的运行时 SHA256 为 `fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055`。该 Linux 套件完成官方 ETT 48、Rust ETT 48、水泥 9，共 105 次 CUDA 运行；D4 32/32、18,574,560 行独立复算、Rust 102/Python 55 项质量检查通过。证据入口为 P2 `research/acceptance.md`，原始套件在忽略目录 `runs/p2-acceptance/20261012-linux-cuda/`。此 SHA 是 P2 历史身份，不代表当前 P3 源码。

P2 保持 `in_progress`：完整 CPU 对照未执行、当前源码 CPU 逐位恢复条款未复验、原 RTX 5090 Laptop 条款未满足，且三个水泥运行在 `cuda:1`。这些是待明确处置的验收差异；不重写已通过的 CUDA 事实，不自动豁免，也不安排违背 CUDA 优先指令的例行 CPU 训练。

## 顺序与任务所有权

| 顺序 | 任务 | 最小交付 | 启动条件 |
| --- | --- | --- | --- |
| 1 | P2 收口 | 当前状态与剩余条款集中记录；对条款补验或延期作明确决定 | 不重复现有 105 次运行；归档须有明确处置 |
| 2 | P3 Informer | ProbSparse、蒸馏、decoder、BatchNorm 持久化、105 次 CUDA 对照 | P2 技术交付可复用；P3 计划审批后启动 |
| 3 | WaveForM | 来源/许可与协议研究、Haar 与图传播可训练性、129 次 CUDA 对照 | 计划审批后先研究；来源/许可/协议明确后实现；算子、恢复与预算通过后正式训练 |
| 4 | P4 比较与扩展实验 | 先 compare/静态报告，再五公共数据集矩阵、回测、搜索 | 比较可先消费 P1/P2；Informer/WaveForM 只有完成各自验收才计入完整模型矩阵 |
| 5 | 其他模型 | 首选 PatchTST；再评估 TSMixer 或 TimeMixer、TCN | 为选中的模型单独建任务，冻结上游与验收；不把所有候选绑进 P4 首个报告 |
| 6 | P5 | 原有低精度、融合算子与服务路线 | 维持第一版发布后的前置条件 |

Informer 与 WaveForM 共享 P2 运行时，不互相依赖模型内部实现。默认先 Informer；WaveForM 来源研究可以独立开展，若其许可或算子门未通过，保留阻塞证据并推进已审批的其他独立任务。P4 比较与报告无需等待所有候选模型实现。

## 层与接口

沿用当前五 crate：`ferrots-core` 管配置/模型描述，`ferrots-data` 管数据协议与窗口，`ferrots-nn` 管可导算子/静态模型派发，`ferrots-train` 管训练/评价/检查点，`ferrots-cli` 管命令。继续扩充 `ModelSpec`、`ForecastModel` 与能力表，不增加插件系统或第二套训练运行时。

Informer 的持久化 buffer、随机采样和 decoder 在现有层内实现；标签不进入 `ModelInput`。WaveForM 复用数据、掩码监督和评价，新增内容限于小波与图模型自身。P4 消费同一运行产物合同，不重做训练状态持久化。每个模型的正式验收必须覆盖训练、保存、恢复、评价和预测入口。

## 实验量与证据复用

| 范围 | CUDA 正式训练数 | 计数说明 |
| --- | ---: | --- |
| Informer | 105 | ETT 4×4×3×2 实现=96；水泥 3×3=9 |
| WaveForM | 129 | ETT 96；Weather 4×3×2 实现=24；水泥 9 |
| P4 首批三个模型 | 180 | DLinear/iTransformer/Informer 各 5 数据集×4 H×3 种子=60 次 Rust 运行 |

P4 若纳入已通过验收的 WaveForM，总矩阵为 240 个 Rust 实验格；其同协议 Weather 12 格可复用，只有在来源、配置、数据、种子、设备与窗口合同都匹配时计入，若这 12 格均可复用，P4 新运行 228 次；不能复用则为 240 次，不重复计数。不同训练日程/模型身份的结果不能因数据名相同就复用。正式训练前先测 CUDA 短运行耗时、显存与所需磁盘，提交预算评估。

所有正式种子沿用 `{2021,2022,2023}`。ETT 使用项目既定协议；ILI 使用 L36/H24、36、48、60。私有数据使用全部 18 变量、L96/H15、30、60 与稳定运行段协议。优先单卡 CUDA，一卡一个独立实验；双卡并行不称为分布式训练。

运行必须绑定源码、二进制、配置、数据、工具及上游指纹，保留失败与中断 attempt。P1 仅有单种子基线时标 n=1；跨环境的 P1 数值仅为历史参考，不宣称同机公平速度或优劣结论。P2 报告可直接作为当前同机结果入口。

## 候选模型依据

- PatchTST：优先补充时间 patch、通道独立的监督预测基线；来源为 [作者仓库](https://github.com/yuqinie98/PatchTST) 与 [论文](https://arxiv.org/abs/2211.14730)。下一任务先核实固定 SHA、许可及归一化/patch/head 语义，不引入预训练范围。
- TSMixer 与 TimeMixer 是不同模型，不能混名。当前 P4 的 TSMixer 候选保留；TimeMixer 的多尺度 MLP 作为另一候选，来源为 [作者仓库](https://github.com/kwuking/TimeMixer) 与 [论文](https://openreview.net/forum?id=7oLshfEIC2)。正式选型前冻结各自来源，不同时承诺实现。
- TCN 保留为卷积基线候选。Autoformer/FEDformer/TimesNet 不纳入本轮启动范围。

## 审批与完成边界

本设计建立时的授权为合并、分支整理及规划。此后 P3 Informer 已获实施与旧正式套件预算授权，状态为 `in_progress`；新的正式 suite 仍须先审查当前源码身份、局部证据及预算。WaveForM 与 P4 保持 `planning`，不得由 P3 进度自动启动。P2 旧条款的补验/延期需要单独明确，不能用分支整理代替。WaveForM 源码复用资格未确定前不复制其实现或将其声明为已有许可代码。
