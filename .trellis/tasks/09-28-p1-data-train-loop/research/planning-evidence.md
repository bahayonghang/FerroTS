# P1 规划证据与接口核对

日期：2026-09-28。核对方式：本地仓库、固定 SHA 的上游缓存、P0 已提交验收记录。未运行 P1 训练或把已有 smoke 当成正式验收。

## 1. 状态

- 起始工作区干净，feat/p0-foundation，HEAD=134737dfd6c3055c76c369903e66f68e9f4f431d，与用户交接一致。
- P0 research/review-b.md、implement.md 和 PRD 验收项均记录完整通过；本次重跑 bash tools/check_private_data.sh 退出 0。未重复运行已通过的 Rust/CUDA 测试。
- task.py archive 生成 99d13f1；add_session.py 用 P0 实施 SHA 记录 session 2，生成 f840dfa。随后从 feat/p0-foundation 创建 feat/p1-data-train-loop。
- P1 保持 planning，无活动任务指针。本轮仅编写规划与上下文清单。datasets-protocol 未重复归档；bootstrap-guidelines 和其他子任务未变更。

## 2. 来源与使用边界

| 来源 | 读取范围 / 作用 |
| --- | --- |
| FerroTS_Rust_Implementation_Plan.md:105-167 | 五 crate 边界；HostBatch、窗口、scaler、masked loss 与有效元素加权 |
| FerroTS_Rust_Implementation_Plan.md:169-185 | ModelInput 与 Supervision 分离；模型能力校验 |
| FerroTS_Rust_Implementation_Plan.md:231-289 | 冻结验证、检查点、指标长表、配置合并 |
| FerroTS_Rust_Implementation_Plan.md:313-345 | 四层验证、回归场景、P1/P2 范围 |
| 父任务 prd.md | D1–D6、X1–X6；D4 是同机官方三种子门槛 |
| 父任务 research/upstream-verification.md | Candle/TSLib/iTransformer 的历史核对；平台状态由 P0 环境记录更新 |
| 父任务 research/dataset-profile.md | 原始水泥统计；f-CaO 内容仅属历史，不进入 P1 |
| 父任务 research/public-datasets.md | 九个公共数据文件指纹；Exchange 格式、行数与时间范围 |
| docs/datasets/README.md、manifest.toml | 协议索引、私有数据规则与公共指纹 |
| docs/datasets/ett.md §3–11 | ETT 行边界、窗口数、scaler、freq=h、训练与模型参数 |
| docs/datasets/cement-yangquan.md §2.1–2.6 | 四项已批准调整及统计验收值 |
| docs/datasets/custom-lsf.md §1–5 | 特殊表头、Exchange 解析；其余大矩阵留 P4 |
| docs/environment.md | 固定 Rust、MSVC、CUDA PATH、静态 CRT、uv 环境 |
| .trellis/spec/backend/training-foundation.md | P0 public API、无梯度路径、设备失败、优化器状态契约 |

总计划超过单文件注入提示阈值；上下文清单使用本证据摘要，实施前仍按 implement.md 的启动清单读取总计划相关节。内联执行不依赖子代理注入。

## 3. 固定上游核对

本地 git -C tools/reference/.cache/Time-Series-Library rev-parse HEAD 返回 4e938a1767106324dd753b2a44832bf870a0252e，与协议一致。以下锚点均属于该提交。

| 路径 | 已核实行为 | 规划后果 |
| --- | --- | --- |
| models/DLinear.py:12-49 | individual 由构造函数参数决定；两支 weight=1/L；bias 未覆盖 | 默认 shared；明确支持 individual；不把 bias 置零冒充官方初始化 |
| layers/Autoformer_EncDec.py:21-53 | 首尾复制填充、AvgPool1d stride=1，seasonal=x-moving_mean | 用可微基础运算保持边缘语义；L=1 必须测试 |
| data_provider/data_factory.py:24-28 | timeenc=1 来自 timeF；val 也 shuffle；drop_last=False | ETT 保留尾批；FerroTS 按父计划保留顺序验证 |
| utils/timefeatures.py:99-120 | h 四特征、t 五特征 | ETTm 继续使用 h；cement 用 t |
| exp/exp_long_term_forecasting.py:42-74 | 验证按 batch mean 再均值 | FerroTS 采用有效元素加权；报告披露差异 |
| exp/exp_long_term_forecasting.py:151-164 | 每轮测试、val 选 best，epoch 结束后调 lr | Rust 测试隔离；type1 调用时点必须精确实现 |
| utils/tools.py:12-29 | type1 在函数调用时设置 lr0*0.5^(epoch-1) | 训练 epoch 1/2 都使用 lr0，epoch 3 使用 lr0/2 |
| utils/tools.py:42-55 | 相等验证损失会重存 best 并清零计数 | Rust 严格下降早停；差异公开且 D4 不放宽 |
| exp/exp_long_term_forecasting.py:255-266 | stdout 未限定小数位，metrics.npy 保存完整指标 | 现有 wrapper 的 float 解析可用，不需为精度重写官方模型 |
| run.py:88-103 | epochs=10、batch=32、patience=3、lr=1e-4，num_workers 默认 10 | 正式运行显式设置关键值及 Windows num_workers=0 |

docs/datasets/ett.md §8 的 type1 表述没有说明调用发生在 epoch 结束。本设计按固定源码明确实际 epoch 序列。实施时只补充文档的时点说明，不改变协议版本或参数。

## 4. 现有代码的复用与缺口

- crates/ferrots-data/src/lib.rs 当前只有骨架，P1 数据语义尚未实现。
- crates/ferrots-train/src/params.rs:15-18 拒绝空参数；:59-85 加载前校验完整参数集。无参数基线需独立分支，不能放宽既有契约。
- crates/ferrots-train/src/optim.rs:75-79 的 groups 为私有，:119-121 仅返回只读切片。type1 需要最小学习率 setter，不能每轮重新 new optimizer。
- crates/ferrots-cli/src/main.rs 仅有 doctor；新增命令需要统一错误入口，不能继续把所有错误标为 doctor failed。
- tools/reference/run_tslib.py 已校验上游 HEAD 和本地数据文件，按 --seed 替换 fix_seed，写 stdout.log、metrics.json 与 checkpoints。保留该入口；验收工具在外层补齐机器/数据/配置/运行覆盖证据。
- tools/reference/pyproject.toml、uv.lock、.python-version 和 .venv 是参考环境权威；不新增第二个环境。
- tools/datasets/profile_cement.py:52-71 给出段生成语义；:154-169 给出不重置陈旧时钟的填充；:185-208 给出先按完整运行段分配、再拟合可用训练分钟的顺序。

## 5. 规划选择与未验证事实

- 规划种子为 2021/2022/2023，D4 标准差 ddof=0；这些是待本次规划批准的可重复参数。
- cement SeasonalNaive period=60、DLinear 使用 Adam/lr=1e-4/batch=32/max_epochs=10/patience=3，作为固定基线设置；不宣称这些参数最优。
- 正式 D4 是否通过、全部训练耗时、缓存收益、Windows 检查点故障行为尚未验证。
- 原始报告和协议中的水泥数字是已有数据卡统计，本轮未重新解析 1 GB 文件。实施验收将重新校验文件 SHA 和这些统计。
- 完整矩阵保持 96 次 ETT 训练与 18 次水泥运行；额外 CPU 闭环、夹具和回归测试不能替代矩阵。
- P0 编译/测试证据没有替代 P1 的任何验收项。

## 6. 规划检查记录

- task.py validate：implement.jsonl 15 条、check.jsonl 13 条，均通过；引用文件存在、无重复、每文件不超过 32768 字节。
- PRD 的 17 个需求全部保留。R1 把“其他文件”明确为其他公共文件，并补充数据卡已规定的 cement T 分隔/微秒格式；R2–R17 未改。10 个 AC 原文保留，仅增加 AC1–AC10 标识，全部未勾选。
- implement.md 包含顺序唯一的 Step01–Step19；14 个 PowerShell 代码块通过语法解析。
- 矩阵算术检查：ETT 4*4*3=48 次/实现，共96次；16*2=32项 D4；水泥3*3*2=18次；三变量导出103192*60*3=18574560行。
- 单步 DLinear 优化更新测试放在 train，避免 nn 反向依赖 train；已有 ferrots 二进制命名保持不变。
- git diff --check 与私有数据检查通过。P1 保持 planning，未设置当前任务，未启动产品实施或正式训练。
- 本轮变更仅为 P0 收尾记录及 P1 规划；未安装工具链、未删除数据/运行目录、未推送。P1 规划文件暂未提交。
