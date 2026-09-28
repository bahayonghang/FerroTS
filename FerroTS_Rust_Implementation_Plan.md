# FerroTS：Rust 原生时序预测平台完整实施方案

**方案版本：0.1 / 核查日期：2026-09-28**

**交付性质：架构与实施设计，不是已完成的软件。** 本文中的命令、配置、模块和支持矩阵属于拟议接口与验收目标；没有在用户显卡上编译或训练，也没有取得 GPU 性能实测结果。上游事实来自访问时的官方 main 分支源码，实施前必须固定提交 SHA；本次未取得可核验的三个仓库提交 SHA，不以版本号代替提交指纹。

## 1. 总体决策与命名

推荐项目名 **FerroTS**，GitHub 仓库名 `ferrots`，命令行程序 `ferrots`，Rust crate 前缀 `ferrots-*`。中文定位为“Rust 原生多模型时序预测与实验平台”。这个名字不绑定 Candle、某一个模型或水泥行业；名称可用性、商标及 crates.io 注册状态尚未确认。

项目不是把 TSLib 的 Python 文件机械翻译成 Rust，也不是仅对 Python 训练脚本增加一个 Rust 启动器。目标是：本地 CSV/Parquet → 数据校验与时间切分 → 滑动窗口 → Rust 原生模型 → CUDA 自动求导与优化 → 可恢复检查点 → 回测、预测和本地部署。

**主路线：Candle + 自建训练安全算子层 + 统一实验运行时。** 不重新实现矩阵乘法、显存管理和整套自动求导；只补齐时序模型确实需要、上游路径不满足要求的算子、状态管理和任务协议。生产训练与推理不依赖 Python 或 LibTorch。CUDA 仍依赖 NVIDIA 驱动、工具链和相应底层库，不能宣传成“整个软件依赖树全部是 Rust”或“任何机器复制一个二进制即可使用 GPU”。Candle 的官方训练示例与 CUDA feature 为这条路线提供基础。[S14][S16][S17]

## 2. 两个官方仓库的参考分工

| 参考来源 | 优先继承 | 不直接照搬 |
|---|---|---|
| thuml/Time-Series-Library | data_provider、models、layers、exp 分层；统一训练/评估；模型脚本与基线结构 | 脚本式全局参数、特定 CSV 列约定、批次平均评价、训练期间反复查看测试集、仅权重检查点 |
| thuml/iTransformer | 倒置嵌入、变量 token、编码器、预测投影、窗口归一化与反归一化 | 为接口兼容而接收却未使用的解码器参数；默认等同于任意输入到任意目标的软测量模型 |

TSLib 官方于 2026 年 4 月说明不再积极增加新功能，并提示旧基准未必适合衡量当前研究进步。因此把它作为代码与模型语义参考，而不是自动沿用全部排行榜和评价设定。[S01]

### 2.1 已核查的上游约束与改进

1. `Dataset_Custom` 使用 `date` 列，将一个目标列移到末尾，并按约 70%/10%/20% 划分。它已经只用训练段拟合标准化器，这一点应保留；新增显式时间列、输入列、目标列和切分配置，而不是把上游错误描述为全数据标准化。[S06]
2. 当前数据工厂对非测试阶段开启 shuffle；长期预测验证器对每批平均损失再平均。当前 `drop_last=False`，尾批大小不同会使这种统计不等于逐样本平均。新实现验证/测试不打乱，按有效元素总和与总数聚合。验证打乱本身不是未来信息泄漏。[S07][S08]
3. 长期预测训练器每轮计算并打印测试损失，早停实际上仍依据验证损失。新实现训练期只接触训练/验证集，测试集在选定模型后独立使用；问题是人工调参偏倚风险，不能直接说上游把测试集用于梯度更新。[S08]
4. 当前早停工具只保存 `model.state_dict()`。新实现区分“最佳推理权重”和“完整恢复训练状态”。[S09]
5. 当前 MAPE 直接除以真实值，接近零的工业变量不适合无保护地使用。默认报告 MAE、RMSE；MAPE 必须配置分母阈值、有效覆盖率和单位语义。[S10]
6. `Dataset_Custom` 本地路径缺失时存在远程数据回退逻辑。面向本地工业数据，默认离线、文件不存在即报错；下载功能必须显式启用。[S06]
7. 模型层支持不等于训练入口、GPU 和任务全部支持。每个注册模型附能力表，只有通过相应测试后才标记“可训练”。

## 3. 技术栈与边界

### 3.1 推荐选型

| 部分 | 首版建议 | 取舍 |
|---|---|---|
| 张量/自动求导/CPU/CUDA | candle-core、candle-nn | 固定同一版本和提交；训练只能使用经过验证的路径 |
| 自定义 GPU 算子 | Candle CustomOp；必要时通过匹配版本的 cudarc | 复用 Candle device、CUDA context 和 stream；不能只按 GPU 编号任意交换指针 |
| CSV | csv | 首版依赖较小；显式解析数据类型、空值、时间列 |
| 复杂表格与 Parquet | 可选 Polars/Arrow | 不让所有用户为高级表格功能承担依赖与编译成本 |
| 配置与 CLI | serde + toml + clap | 类型化参数；未知配置项报错 |
| CPU 数据并行 | rayon / 有界 channel | 与 GPU 计算线程分离；预取有上限 |
| 模型与优化器张量 | safetensors | 元信息单独保存 JSON/TOML |
| 事件与结果 | JSONL + CSV/Parquet，后期可索引到 SQLite | 本地优先，不强制云实验服务 |
| 展示 | 首版 CLI 与静态报告；后期 Rust 服务 API | 不以开发桌面 UI 阻塞训练内核 |

csv、Polars 都提供 CSV 读取入口；Candle 的 VarMap 提供 safetensors 保存基础，但它不等于完整训练状态。[S25][S26][S27]

本轮查看到的 Candle main 工作区声明 `0.11.0`，这不代表本文确认了该版本已经作为稳定发行版发布。立项时选定经过测试的 tag/commit，所有 Candle 子 crate 一起锁定；不直接把 `main` 作为长期依赖。[S15]

### 3.2 为什么不是一开始同时支持三个后端

数据、配置、切分、指标和产物格式保持后端无关；`ferrots-nn`、`ferrots-train` 首版明确使用 Candle Tensor。不要写一个覆盖所有张量操作的抽象框架，也不要声称未来把 Candle 换成 Burn 无需移植模型。

Burn 有面向训练与推理的 CUDA/自动求导能力，可作为阶段门失败后的替代底座；但启用它时需要模型或适配实现，不只是改配置。tch-rs 仍是允许 LibTorch 依赖时的备选，不作为本方案默认运行时。[S29]

## 4. Candle 的关键问题及解决办法

### 4.1 训练安全算子层是必需模块

**能在 GPU 上完成前向计算，不等于能正确训练。** 当前 `softmax_last_dim` 使用 `apply_op1_no_bwd`，优化版 `layer_norm` 使用 `apply_op3_no_bwd`；这些接口建立的结果没有反向路径。`LayerNorm::forward` 还会在部分连续张量情况下进入优化路径。[S18][S19][S21]

建议建立 `nn/train_ops/` 与 `nn/infer_ops/`：

| 算子/模块 | 首版训练路径 | 推理加速策略 |
|---|---|---|
| Softmax | 普通可微 softmax；核查所有掩码行均有效 | 梯度不需要时才使用 fused softmax |
| LayerNorm | `layer_norm_slow` 或等价基础运算；显式初始化缩放/偏移参数 | 验证数值一致后使用优化版 |
| 注意力 | QK 矩阵乘法 + 稳定 Softmax + AV | FlashAttention 作为后期推理路径 |
| Dropout | 明确 train/eval 标志；可恢复随机流 | 推理关闭 |
| 窗口方差 | 显式实现总体方差 | 不替换为默认无偏方差 |
| 参数更新 | 自建可序列化 Adam/AdamW，FP32 状态 | 不适用 |

当前 FlashAttn 的 CustomOp3 实现没有提供 `bwd`；默认自定义算子反向会返回“不支持”。因此不能因为启用 `flash-attn` feature 就把它放进训练主干。后期可实现匹配的反向，但必须单独验收。[S21][S22]

梯度测试不仅检查最终 loss 能调用 backward，还要检查输入嵌入、Q/K/V、前馈层、归一化仿射参数和输出层是否获得预期梯度。梯度为零不一定是错误：使用非退化测试输入，并区分“合法零梯度”和“根本没有梯度”。

### 4.2 方差默认值不能照搬

官方 iTransformer 使用 `unbiased=False`，即平方偏差除以窗口长度 L；Candle 当前 `var_keepdim` 则除以 L-1。首版自行实现总体方差，常量列、L=1 和 epsilon 都需测试。[S03][S23]

全局标准化与模型内窗口归一化是两个不同步骤：前者只从训练段估计统计量，后者对当前已知历史窗口计算。不得不加区分地合并为通用 RevIN；复现版还需保持官方 mean detach 等具体语义。

### 4.3 cumsum 可能抵消 Informer 的稀疏优势

Candle 当前 cumsum 构造长度为 n 的 n×n 三角矩阵，再进行矩阵乘法。Informer 的因果初始上下文使用 cumsum。由源码可推断，直接套用会引入平方级结构，即使注意力主体是稀疏的，也不能据此宣称整个实现具有稀疏复杂度。[S12][S23]

解决方式：提供 `PrefixSum` 自定义算子，CPU 先用线性扫描；CUDA 先实现正确的逐行扫描，再优化为分块并行 scan。反向是反方向的累积和：某个输入会影响它之后的所有输出，因此它的梯度等于后续输出梯度之和。前向、反向都不能构造 n×n 矩阵。

### 4.4 Informer 蒸馏模块需要专门测试

TSLib 蒸馏包含 circular Conv1d、BatchNorm、ELU，以及 kernel=3/stride=2 的 MaxPool1d。不能用不支持对应反向的池化接口硬替代；Candle 当前 MaxPool2D 反向对 kernel 与 stride 不等的情况有限制。[S13][S20]

实现 `MaxPool1d` 时记录最大值位置，并把上游梯度累加回对应输入；相邻窗口可能指向同一个位置，必须累加。重复最大值要定义与参考版本一致的选择规则。BatchNorm 的 train/eval、运行均值/方差、动量和缓冲区恢复都单独验证；不要根据一段过时注释推断整个模块不支持训练。

### 4.5 优化器、精度和恢复训练

Candle 当前 AdamW 的 moment 状态随参数 dtype 创建，且公开 Optimizer 接口没有完整状态导入/导出契约。建议自建 `StatefulAdam`、`StatefulAdamW`，维护稳定参数名、逐参数 step、FP32 一阶/二阶矩和参数组；不能只包一层已有优化器却拿不到其内部状态。[S24]

首版只正式支持 FP32。BF16/FP16 后期以 PrecisionPolicy 管理：FP32 master weights 与优化器状态，关键归一化/损失在 FP32，低精度计算使用可微 cast。FP16 另做动态 loss scaling、非有限梯度检测与跳步状态管理；BF16 也不应默认等于数值安全。首版遇到未验证 dtype 直接拒绝。

## 5. 仓库与总体架构

```text
ferrots/
├── Cargo.toml / Cargo.lock / rust-toolchain.toml
├── crates/
│   ├── ferrots-core/      # 配置、任务/变量语义、指标、产物协议；不依赖 Candle
│   ├── ferrots-data/      # CSV、时间校验、scaler、窗口索引、HostBatch
│   ├── ferrots-nn/        # Candle train_ops、layers、models、registry
│   ├── ferrots-train/     # Trainer、优化器、调度器、恢复状态
│   └── ferrots-cli/       # doctor/train/evaluate/predict/backtest
├── configs/              # reference 与 engineering 分开
├── tests/fixtures/       # 固定权重、输入、逐层输出和梯度
├── tools/reference/      # 仅开发期使用的官方 Python 数值基准
├── docs/                 # 模型卡、数据协议、算子能力、环境矩阵
└── runs/                 # 默认不纳入 Git
```

首版不拆成十几个空 crate。自定义内核规模扩大后，再分出 `ferrots-kernels`；服务与桌面层也放在后续阶段。

数据流：CSV → 类型化 SeriesTable → 切分与预处理元信息 → WindowIndex → HostBatch → GPU DeviceBatch → 模型 → 损失 Tensor → backward → 梯度检查/累计/裁剪 → 优化器 → 日志与检查点。

CPU 工作线程不创建 CUDA 张量；GPU worker 持有自己的 context 和执行队列。共享数据表只读、批次缓冲区可复用，有界队列防止预取耗尽内存。数据预取与模型计算是否真正重叠，需要用 profiler 验证，不能仅因用了线程就认定实现了异步传输。

## 6. 数据协议与防泄漏设计

### 6.1 支持的数据形式

首版：一行一个时间点、一列一个变量的宽表 CSV；必须有可配置时间列。后续：多设备 `series_id`、长表、Parquet、多文件与多采样率流。

DataSpec 明确：时间列、时区、采样间隔、排序与重复策略、输入列、目标列、未来已知列、有效值标记、单位、设备/批次边界。不要根据列名猜测目标，不允许目标因列位置改变而被静默替换。

初次读取输出 DatasetReport：行数、起止时间、乱序/重复数、缺失率、常量列、异常时间间隔、有效窗口数。默认不自动下载数据，不默默删除异常行。

### 6.2 时间窗口的定义

设 o 是第一个待预测行的索引，L 是历史长度，H 是预测长度，g 是预测前的跳过步数。历史使用 `[o-L, o)`；目标使用 `[o+g, o+g+H)`，左闭右开。g=0 表示直接预测后续 H 个采样点。

例如 L=96、H=24、g=0，历史为 0～95 行，目标为 96～119 行。验证样本可以使用训练区间末尾的历史作为上下文，因为这些值在预测时已知；所有目标行必须属于所评估的那个目标区间。

训练、验证和测试先确定时间边界，再生成索引。窗口不能跨设备、跨生产批次或跨越被标记为断开的长停机区间。训练窗口可以在合法索引集合内打乱，不等于可以随机切分全部窗口。

### 6.3 数据实际可获得时间

扩展数据契约记录 `event_time` 与 `available_at`：前者是样品/测量对应时间，后者是结果进入系统的时间。历史特征必须在相应预测时刻已经可得；训练标签必须在该回测训练截止时刻已发布。未来真实目标只参与损失与评价。

对于 f-CaO，不能把化验结果回填到历史后再当作当时已知的输入。可采用有上限的因果前向填充、距离上次有效观测的时间和缺失掩码，但这些会改变模型输入协议，应放入工程扩展模型。真实缺失的质量标签不能用填充值假装成监督真值。

### 6.4 标准化与缺失值

训练段拟合每列均值/尺度；验证与测试只 transform。输出 scaler 按目标名保存，不能通过复制目标预测来适配输入 scaler 维数。常量列设置稳定尺度并记录标记。

复现模式默认缺失值报错；工程模式支持受限因果填充、最大陈旧时间和有效掩码。需要缺失掩码但模型不支持时，不允许静默忽略：改用明确的适配器/模型变体，或拒绝配置。

### 6.5 统一损失

对于有效标签，使用掩码均方误差：

`loss = sum(mask * (prediction - target)^2) / sum(mask)`

mask 与预测形状相同，有真实标签处为 1，其余为 0。该式来自“只对实际观察到的目标取平均”，不让填充值影响监督。实际实现先把无效位置变成有限值或显式选择有效位置，再平方与求和；不能依赖 NaN 乘零消除缺失。

分母为零时跳过优化更新并记录原因；整个 epoch 无有效标签时直接失败。梯度累计与评价都按有效元素总数加权，不能对大小不同的 microbatch 均值再简单平均。多目标加权时同时保存每个目标的原始单位指标，避免总分掩盖某一个质量指标恶化。

## 7. 模型接口、注册与算法范围

ModelInput 只包含预测时已知的字段；Supervision 单独包含未来 target、mask 与权重。模型不得接收 Supervision。需要 decoder 的模型从已知历史构造 decoder 上下文，再拼零或协议规定的未知占位，而不是从未来真实目标复制。

统一外部形状 `[B,L,C_in] → [B,H,C_out]`。B 为批次样本数，L 为历史长度，H 为未来步数，C_in/C_out 分别是输入与输出变量数。模型内部可转置，但返回时恢复统一布局。

模型注册条目包括：名称、版本、来源仓库/SHA/文件、参数 schema、任务类型、目标模式、时间特征支持、future-known 支持、mask 支持、训练精度与设备矩阵、复现测试状态。新增模型通过静态注册或类型化 enum 接入；不使用未定义稳定 ABI 的动态 Rust 插件。

| 规划顺序 | 模型 | 作用 | 进入支持列表前的门槛 |
|---|---|---|---|
| P1 | LastValue、SeasonalNaive、Linear/DLinear | 管线与基线 | 索引、损失、CPU/GPU、保存加载 |
| P2 | iTransformer | 第一优先核心模型 | 官方逐层前向与梯度对齐 |
| P3 | Informer | 第二优先核心模型 | ProbSparse、前缀和、索引写回、蒸馏完整训练 |
| P4 | PatchTST、TCN、TSMixer/Transformer | 扩充不同归纳偏置的模型 | 各自结构与任务语义测试 |
| P5 | Autoformer、FEDformer、TimesNet 等 | 频域/分解扩展 | FFT/复数路径与反向、复杂度、平台测试 |

不把模型文件存在或某次 loss 下降作为“算法已支持”的充分条件。新增参数对既有模型无意义时应报错，例如 iTransformer 配置中误写 Informer 专用 decoder 参数。

## 8. iTransformer 实现细节

官方基本路径为：历史输入 → 可选窗口归一化 → 变量维 token 嵌入 → Encoder → 每个变量的预测投影 → 剔除辅助时间 token → 反归一化。[S03][S04]

例：B=32，L=96，C=33，d_model=128，H=24，不使用辅助时间 token：

```text
[32,96,33] → [32,33,96] → [32,33,128]
           → 编码器 → [32,33,24] → [32,24,33]
```

即每个传感器的整段历史成为一个 token，注意力在变量之间运行，不能误写成时间点间的普通 Transformer。启用 K 个时间特征 token 后，编码器 token 数为 C+K，最后只保留过程变量输出。

复现需要保持：总体方差、epsilon、均值 detach、归一化与反归一化顺序、Post-Norm 的残差顺序、Dropout 位置、GELU 具体形式、投影 bias、变量/时间 token 排列。编码器 FFN 的 kernel=1 Conv1d 可以改写成数学等价 Linear，但权重维度与偏置必须正确映射。[S03][S04][S05]

官方结构的时间嵌入和输出投影分别依赖固定 L 和 H；同一个权重文件不应宣称随意接受任意历史长度/预测长度。设备变更时变量顺序和 scaler 映射仍必须检查。

### 8.1 过程变量预测与软测量要分开

`itransformer` 复现版输出输入变量对应的未来序列，允许按输入中的目标索引选择子集。若输入只有温度、电流、喂料量，目标却是未作为输入变量的 f-CaO，不能靠设置 `c_out=1` 获得原版结构的严格复现。

后期提供明确命名的 `itransformer_query` / `itransformer_softsensor`：变量编码器产生多变量表示，目标专用查询或跨变量读出产生任意目标序列，并使用独立 target scaler。这是待验证的新工程变体，不称为官方原版，也不预先保证优于原版。

官方基础模型没有真正利用未来解码器输入进行外生信息融合；配置 future-known 变量时，基础版应拒绝或要求显式融合变体，而不是接收后静默忽略。[S03]

## 9. Informer 实现细节

以 TSLib 的 `models/Informer.py`、注意力和编码器文件为具体复现版本，完整记录来源。目标不是将所有长序列 Transformer 都归入 Informer 名称。[S11][S12][S13]

ProbSparse 实现顺序：采样 key → 计算 query 稀疏性评分 → 选择 query 索引 → 对选中的 query 与全部 key 计算注意力 → 更新上下文。未选 query 使用参考版本规定的初始上下文。索引选择本身不可微，但选中值以及上下文更新的梯度必须保留。

关键模块：

- `SampledQkScore`：避免把 broadcast view 实体化为完整五维大张量；采样随机流可重现。
- `TopQuerySelect`：定义重复值 tie 规则；测试 batch=1、head=1 等场景，避免无参 squeeze 丢掉语义维度。
- `PrefixSum`：前向线性扫描、反向后缀扫描，避免三角矩阵。
- `ReplaceRows`：对选中位置做覆盖而不是相加。反向中，被覆盖的旧上下文梯度为零，更新值取得所选位置梯度；选中位置唯一性须验证。
- `CircularConv1d/MaxPool1d`：保留 padding、序列长度变化和重复最大值规则；BN 缓冲区不参与优化器更新但要保存。
- 因果 mask 按 query 的原始时间位置构造；不能只按 top-query 列表的新位置构造。

正确性阶段允许小尺寸慢实现，但生产 GPU 路径不得把完整 Q/K/V 复制到 CPU 做选择，也不允许内部偷偷换成 dense attention。显式 `informer_dense` 可以用于消融，但它不是 ProbSparse 复现版。还要保留上游某些具体实现细节，例如稀疏分数分母、mask 逻辑、注意力 dropout 实际使用位置；任何修正单独标注为工程变体。

对于长预测序列，检查 decoder 中所有路径的复杂度，而不只测 encoder；关闭大注意力矩阵导出，避免诊断输出本身占用平方级内存。

## 10. 训练器与实验管理

训练循环：构造批次 → 前向 → 有效标签损失 → backward 返回梯度 → 累计并按实际有效数归一化 → 检查非有限值 → 可选全局范数裁剪 → optimizer step → scheduler step → 释放该批计算图。

返回损失必须仍是 Tensor；可以读取标量做日志，但不能只保留 f64 再尝试反向。不要把带图的输出存入跨 epoch 日志集合。

复现 profile 使用与固定官方配置匹配的 Adam、学习率计划、dropout 等；工程 profile 可默认 AdamW、裁剪与 warmup/cosine。AdamW 不应未经声明替换官方 Adam；零 weight decay 时也要核对 epsilon、bias correction 和状态更新语义。

验证需要独立的 train=false 与无梯度图路径；关闭 Dropout 不等于自动停止参数的计算图记录。首版可构造冻结 Tensor 的只读模型视图/快照执行验证，避免伪造一个并不存在的框架级 no_grad API。服务推理使用完整导出的冻结权重。

### 10.1 检查点

```text
checkpoint/
  manifest.json
  model.safetensors
  optimizer.safetensors
  buffers.safetensors
  trainer-state.json
  scheduler-state.json
  rng-state.json
  scaler.json
  data-schema.json
  resolved-config.toml
  environment.json
```

manifest 记录模型名/版本、参数映射、上游 SHA、当前代码 SHA、数据指纹、文件校验和、dtype、窗口、目标顺序等。优化器记录每个参数的 moment 与 step。BN 等缓冲区独立于可训练参数。

首版承诺在 optimizer-step 边界恢复；不承诺半个 microbatch 内精确恢复。随机性覆盖数据顺序、Dropout、ProbSparse 采样和初始化；必要时采用与 run/step/layer/sample_id 绑定的可重建随机流，而不是声称存一个 seed 就够了。

检查点先写临时目录，验证完整性后提交 manifest；保留 best 与 last 两种用途。加载时核对变量列顺序、shape、模型版本、数据协议和缺失参数；不静默接受不完整权重。

### 10.2 评价与实验搜索

输出原始单位与标准化空间的 MAE/RMSE/MSE，分变量、预测步长、设备/工况统计。全局分数明确是元素加权还是变量等权；多量纲变量不能只给一个原始单位混合总分。

预测表保留 `run_id, series_id, forecast_origin, target_time, horizon, target_name, y_true, y_pred, valid`。同一个目标时间来自不同 forecast_origin 的预测，不默认混合平均。

支持固定切分和 rolling-origin 回测；每折重新拟合 scaler 与训练状态。超参数搜索先提供 Rust 网格/随机搜索，逐任务记录预算、种子和配置；先验证集选择，再独立最终测试。多 GPU 优先做“一张卡一个独立实验”，不能把它称为分布式数据并行训练。

## 11. 配置与命令行

本包提供两个 TOML **接口设计模板**，不是当前可执行程序的配置。实现后拟支持：

```bash
ferrots doctor --device cuda:0 --training-smoke-test
ferrots models list --task forecast --device cuda:0
ferrots data inspect --config configs/itransformer.example.toml
ferrots train --config configs/itransformer.example.toml
ferrots train --config configs/informer.example.toml
ferrots evaluate --run runs/<run-id> --split test
ferrots predict --checkpoint runs/<run-id>/best --input data/recent.csv
ferrots resume --checkpoint runs/<run-id>/last
ferrots backtest --config configs/itransformer.example.toml
ferrots compare --runs runs/<run-a> runs/<run-b>
```

配置合并顺序：内置默认值 < 配置文件 < 明确 CLI 覆盖。把完整展开后的最终配置写入产物，配置 hash 参与 run_id。模型参数与任务参数分区；`d_model % n_heads != 0`、非法时间区间、缺少 future-known 数据、CUDA 不可用等均在正式训练前报错。

## 12. 性能优化路线

先建性能剖析：CSV 解析、窗口打包、CPU→GPU 传输、前向、反向、参数更新、验证、写盘分别计时；GPU 计时要正确同步或使用事件。记录峰值显存、主机内存和稳态吞吐，不只记录第一轮耗时。

数据不预先复制全部重叠窗口。保留连续只读数据与窗口索引，每批打包；对大表按分片/Parquet row group 或二进制缓存处理，避免反复解析 CSV。缓存由数据指纹、列配置、预处理版本和切分配置共同标识。

按剖析结果优化 contiguous 布局、批次缓冲区、日志频率、预取、pinned-memory 与传输重叠；后两者在兼容底层 API 和生命周期测试通过后启用。自定义 kernel 与 Candle 共享 context/stream，避免额外跨后端拷贝。[S28]

AMP、可训练融合 LayerNorm/Softmax、梯度检查点均是后期特性，不能作为首版性能承诺。改为 Rust 不保证 GPU 训练快于 PyTorch；性能目标以同权重形状、dtype、batch、数据和评价协议的实测对照为依据。

OOM 不应静默改变实验。批量大小探测在正式训练前进行并恢复初始随机状态；训练中失败则保存失败原因，显式重新启动新的配置或从一致状态恢复。

## 13. 平台与部署

首个 GPU 验收环境建议为 Linux x86_64 + 单张 NVIDIA GPU；Windows 原生作为明确测试目标，同时保留 WSL2 路线。NVIDIA 提供 Windows CUDA 和 WSL CUDA 支持文档，但这不等于所有 Candle 依赖组合已经在用户设备验证。[S30][S31]

`doctor` 报告 Rust/目标平台、driver/toolkit、GPU 与算力、运行时 DLL/SO、Candle build features；再运行一个真实的前向→反向→参数更新测试。显式 `cuda:0` 失败即退出，不静默改为 CPU。

CPU-only 构建不要求 CUDA 工具链；CUDA 与可选高级内核通过 feature 控制。环境矩阵只填写实测版本，不给未经测试的“最低 CUDA 版本”承诺。WSL 使用文档规定的 Windows 驱动桥接方案，不在 WSL 内随意安装另一套显示驱动。

部署产物包含冻结权重、scaler、数据协议、模型配置与版本。批量预测先提供 CLI；后续通过 Rust HTTP 服务提供单设备/多设备预测。默认本地文件、本地日志，无强制遥测；线上数据 schema 不兼容时拒绝请求。

## 14. 测试与验收

### 14.1 四层数值验证

1. 算子级：forward、有限差分/参考梯度、形状、dtype、边界、非连续布局。
2. 模型级：固定输入与相同权重，逐层比较倒置嵌入、attention、FFN、归一化和输出。
3. 更新级：比较一次 loss、全部预期参数梯度和一次 Adam/AdamW 更新。
4. 实验级：多个种子的训练/验证曲线、原始单位指标、检查点恢复与预测一致性。

Python 官方仓库只用于开发期产生可版本化的测试夹具，不参与生产训练。正常用户运行不需要 Python。相同随机种子不保证 Rust 与 PyTorch 得到相同随机数；初始权重、Dropout 掩码和 ProbSparse 索引应显式共享或在确定性测试中关闭随机性。

FP32 初始建议容差可从 atol=1e-5、rtol=1e-4 开始，随后按算子、设备和参考误差校准并固定；不能用不断放宽容差掩盖结构错误。CUDA 的非确定性与近零梯度另做说明，不承诺跨硬件位级一致。

### 14.2 回归场景

覆盖尾批、全缺失标签、常量列、L=1、batch=1、mask 全屏蔽、重复时间戳、跨设备边界、长停机、非法目标列、列重排、读取失败、无 GPU、OOM、损坏检查点、恢复时数据变更。保留一组能检测训练梯度被 no_bwd 算子切断的负面测试。

### 14.3 CI

CPU CI 每次提交执行 fmt、clippy、单元/数据协议测试；受控自托管 GPU runner 执行 CUDA smoke、梯度和模型测试。不能只用普通 CPU CI 的通过状态宣传 CUDA 可用。性能回归在固定硬件跑，不执行不可信 PR 的任意 GPU 代码。

## 15. 实施阶段与完成定义

| 阶段 | 交付 | 完成标准 |
|---|---|---|
| P0 版本与算子验证 | 固定依赖、doctor、训练安全 Softmax/LayerNorm、FP32 | CPU/CUDA 都完成梯度与更新；不可微路径被检测 |
| P1 数据与训练闭环 | CSV、窗口、scaler、基线、基本检查点 | 本地 CSV 完成 train/evaluate/predict，指标手工可核对 |
| P2 iTransformer | 官方复现模型、权重映射、完整状态恢复 | 逐层/梯度/单步更新对齐，固定数据集多种子验证 |
| P3 Informer | ProbSparse、scan、ReplaceRows、pool、decoder | GPU 完整训练，复杂度与显存无隐式平方级退化 |
| P4 通用实验平台 | 更多基线、回测、比较、搜索、报告 | 所有模型共用数据协议与评价，结果可追溯 |
| P5 工业与性能扩展 | 稀疏标签、任意目标读出、低精度、服务 | 每项独立数值/回测/性能验收，名称与默认值不混入复现版 |

**第一版正式发布范围建议：本地 CSV、CPU/单卡 CUDA、FP32、DLinear/iTransformer/Informer、完整检查点、评价与批量预测。** P3 未通过时可以发布 preview，但必须公开显示 Informer 尚未正式支持，不能用名称占位冒充实现完成。

## 16. 风险控制与最终结论

最大风险是梯度链路被优化算子切断、算子复杂度退化、恢复训练状态不完整，以及数据可获得时间建模错误；不是 TOML/CLI 或模型下拉框本身。

处理方式是先做小而完整的训练核心，再扩充算法；工程改进先以独立变体验证，不能悄悄改变官方结构后仍宣称严格复现。上游两个仓库的许可证均为 MIT，应在移植文件、NOTICE 和模型卡记录来源并保留相应声明；后续依赖单独检查。[S32][S33]

一句话：**FerroTS 应当成为以官方模型语义为基准、以训练正确性和时序数据协议为核心的 Rust 时序实验平台，而不只是 TSLib 的语言翻译版。**

关键记忆点：GPU 前向≠可训练；同名方差函数≠同一统计定义；变量预测≠任意质量软测量；保存权重≠恢复训练；切换算法必须同时验证数据能力与算子能力。

自测问题：一个模型 loss 在下降，但输入嵌入与注意力 Q/K 没有梯度，应从哪层排查？把窗口长度从 96 改到 192 后，原 iTransformer 权重为何不一定可直接加载？为什么验证窗口可以使用训练段末尾的历史，但不能用尚未发布的化验值？

---

## 官方资料索引

本文件的 [Sxx] 均指向官方代码、官方项目文档或维护者发布的 API 文档；main 链接会变化，实际开发需将每个引用冻结到完整 SHA。

- [S01] **TSLib 官方仓库与维护说明**
- [S02] **iTransformer 官方仓库**
- [S03] **iTransformer 官方模型**
- [S04] **iTransformer 倒置嵌入**
- [S05] **iTransformer 编码器**
- [S06] **TSLib 自定义数据集**
- [S07] **TSLib DataLoader 工厂**
- [S08] **TSLib 长期预测训练器**
- [S09] **TSLib 检查点与学习率**
- [S10] **TSLib 评价指标**
- [S11] **TSLib Informer**
- [S12] **TSLib 注意力实现**
- [S13] **TSLib 编码器与蒸馏模块**
- [S14] **Candle 官方仓库**
- [S15] **Candle 工作区依赖**
- [S16] **Candle CUDA 功能开关**
- [S17] **Candle 官方训练示例**
- [S18] **Candle Softmax 与归一化算子**
- [S19] **Candle LayerNorm 模块**
- [S20] **Candle 自动求导**
- [S21] **Candle 自定义算子与反向接口**
- [S22] **Candle FlashAttention**
- [S23] **Candle 张量方差与前缀和**
- [S24] **Candle 优化器**
- [S25] **Candle 参数映射与保存**
- [S26] **Rust csv 文档**
- [S27] **Polars CSV 文档**
- [S28] **cudarc 文档**
- [S29] **Burn 官方仓库**
- [S30] **NVIDIA Windows CUDA 安装文档**
- [S31] **NVIDIA WSL CUDA 文档**
- [S32] **TSLib 许可证**
- [S33] **iTransformer 许可证**

[S01]: https://github.com/thuml/Time-Series-Library "TSLib 官方仓库与维护说明"
[S02]: https://github.com/thuml/iTransformer "iTransformer 官方仓库"
[S03]: https://raw.githubusercontent.com/thuml/iTransformer/main/model/iTransformer.py "iTransformer 官方模型"
[S04]: https://raw.githubusercontent.com/thuml/iTransformer/main/layers/Embed.py "iTransformer 倒置嵌入"
[S05]: https://raw.githubusercontent.com/thuml/iTransformer/main/layers/Transformer_EncDec.py "iTransformer 编码器"
[S06]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/data_provider/data_loader.py "TSLib 自定义数据集"
[S07]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/data_provider/data_factory.py "TSLib DataLoader 工厂"
[S08]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/exp/exp_long_term_forecasting.py "TSLib 长期预测训练器"
[S09]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/utils/tools.py "TSLib 检查点与学习率"
[S10]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/utils/metrics.py "TSLib 评价指标"
[S11]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/Informer.py "TSLib Informer"
[S12]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/layers/SelfAttention_Family.py "TSLib 注意力实现"
[S13]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/layers/Transformer_EncDec.py "TSLib 编码器与蒸馏模块"
[S14]: https://github.com/huggingface/candle "Candle 官方仓库"
[S15]: https://raw.githubusercontent.com/huggingface/candle/main/Cargo.toml "Candle 工作区依赖"
[S16]: https://raw.githubusercontent.com/huggingface/candle/main/candle-core/Cargo.toml "Candle CUDA 功能开关"
[S17]: https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/mnist-training/main.rs "Candle 官方训练示例"
[S18]: https://raw.githubusercontent.com/huggingface/candle/main/candle-nn/src/ops.rs "Candle Softmax 与归一化算子"
[S19]: https://raw.githubusercontent.com/huggingface/candle/main/candle-nn/src/layer_norm.rs "Candle LayerNorm 模块"
[S20]: https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/backprop.rs "Candle 自动求导"
[S21]: https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/custom_op.rs "Candle 自定义算子与反向接口"
[S22]: https://raw.githubusercontent.com/huggingface/candle/main/candle-flash-attn/src/lib.rs "Candle FlashAttention"
[S23]: https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/tensor.rs "Candle 张量方差与前缀和"
[S24]: https://raw.githubusercontent.com/huggingface/candle/main/candle-nn/src/optim.rs "Candle 优化器"
[S25]: https://raw.githubusercontent.com/huggingface/candle/main/candle-nn/src/var_map.rs "Candle 参数映射与保存"
[S26]: https://docs.rs/csv/latest/csv/ "Rust csv 文档"
[S27]: https://docs.pola.rs/user-guide/io/csv/ "Polars CSV 文档"
[S28]: https://docs.rs/cudarc/latest/cudarc/ "cudarc 文档"
[S29]: https://github.com/tracel-ai/burn "Burn 官方仓库"
[S30]: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html "NVIDIA Windows CUDA 安装文档"
[S31]: https://docs.nvidia.com/cuda/wsl-user-guide/index.html "NVIDIA WSL CUDA 文档"
[S32]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/LICENSE "TSLib 许可证"
[S33]: https://raw.githubusercontent.com/thuml/iTransformer/main/LICENSE "iTransformer 许可证"
