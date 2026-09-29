# P1 数据管线与基线训练闭环设计

状态：in_progress，用户于 2026-09-28 批准。接口按 implement.md 顺序实现和验证。

## 1. 目标、依据与范围

完成 CSV → 数据报告 → 预处理 → 窗口 → 基线训练 → 检查点 → 评价/预测闭环。实现范围由本任务 P1-R1–P1-R17 和 AC1–AC10 定义。验收覆盖 ETT、Exchange 解析与 cement-twin；执行清单见 implement.md。

约束优先级：本次用户指令 → 父任务已确认的 D1–D6、X1–X6 → 已批准数据协议 → P1 PRD → 总计划。父任务早期研究中的平台未验证、runs 尚未忽略、f-CaO 数据描述属于历史记录；当前环境以 docs/environment.md 为准。

- 起点：P0 实施提交 134737dfd6c3055c76c369903e66f68e9f4f431d；归档 99d13f1；journal f840dfa。
- 分支：feat/p1-data-train-loop，从完成收尾的 feat/p0-foundation 创建。
- 内联实施与内联 trellis-check；批准后才运行 task.py start 和 trellis-before-dev。
- 生产路径只使用 Rust；Python 仅生成公共/合成夹具、运行官方基准、执行独立验收。
- P2/P3 模型、完整训练恢复、Parquet、多设备、多文件、远程 CI、低精度、系统工具链安装不属于 P1。

源文件与已核实差异见 research/planning-evidence.md。仓库没有 CONTEXT.md 或嵌套 AGENTS.md；本任务不创建这些文件。

## 2. 模块归属与依赖

| 模块 | 负责内容 | 权威对象与边界 |
| --- | --- | --- |
| ferrots-core | 类型化配置、数据/模型元信息、指标累计与产物 schema | 不依赖 Candle；未知字段和非法组合在解析后、读取大文件前拒绝 |
| ferrots-data | CSV、原始报告、网格/运行段、切分、scaler、缓存、窗口、HostBatch | DataSpec 决定数据语义；纯 CPU 主机数据，不创建 CUDA 张量 |
| ferrots-nn | LastValue、SeasonalNaive、Linear、DLinear、冻结模型 | 只接收 ModelInput；返回完整输入变量预测，再按目标名选择；不依赖 train/data |
| ferrots-train | DeviceBatch、masked MSE、梯度累计、Adam/LR、Trainer、检查点协调 | 复用 NamedParams/StatefulAdam；依赖 data、nn、core；持有 Candle device |
| ferrots-cli | 参数覆盖、命令分派、错误上下文和退出码 | 默认 < TOML < 显式 CLI；调用业务接口，不另写预处理/指标算法 |
| tools/reference、tools/validation | 独立参考、矩阵执行、结果比较、证据汇总 | 开发工具；公共/合成夹具可提交，实验产物留在 runs/ |

数据流：ResolvedConfig → SourceFingerprint → RawDatasetReport → PreparedSeries → SplitPlan/Scaler → WindowIndex → HostBatch → ModelInput + Supervision → 预测/损失 → 指标/检查点。未来标签不进入 ModelInput。

nn 的模型接口提供 named_parameters() 的名称/Var 列表，由 train 包装为 NamedParams，保持 train → nn 的依赖方向。无参数模型返回空列表，Trainer 跳过优化器创建；保留 P0 NamedParams 拒绝空参数集的契约。

新增依赖限于 CSV、日期解析、TOML、SHA256、流式 gzip 及已有 safetensors。实现时通过固定工具链解析依赖并提交 Cargo.lock；不升级 Candle/cudarc 或另建 Python 环境。

## 3. 配置与数据契约

### 3.1 配置结构

TOML 分为 data、window、model、train、output。DataSpec 是 data 节的类型化值；无需设计第二种配置继承系统。所有结构使用严格反序列化；嵌套未知键、无关模型参数、空目标、重复列名、负长度或不支持的 dtype 均报错。

- data：protocol_id、path、timestamp_column、timestamp_format、timezone、interval_seconds、数值列类型、input_columns、target_columns、future_known_columns、排序/去重、聚合、运行规则、missing、split、time_features。
- window：lookback L、horizon H、gap g、stride，要求 L/H/stride > 0、g ≥ 0，所有边界采用半开区间。
- split：比例、显式行边界、日期边界、运行段累计长度分配四种类型。比例规则采用 floor(n*train)、floor(n*test)、余数为 val；日期边界用配置时间格式解析，禁止重叠与倒序。
- model：类型化 enum，含 individual、moving_avg 或 seasonal_period 等本模型字段。
- train：seed、device、FP32、Adam 参数、lr_schedule、batch_size、accumulation_steps、max_epochs、patience。
- output：run 根目录、指标输出、可选预测导出。CLI 覆盖使用 Option，未显式传入的默认值不覆盖 TOML。

P1 时间戳采用无夏令时转换的显式 naive 时间语义；现有协议均为本地无时区字符串。非空 IANA 时区配置在 P1 明确拒绝，避免隐式时间转换。所有输入/目标按列名定位，按配置顺序构造张量。目标必须是输入变量的非空子集；独立软测量目标属于后续阶段。

future-known 字段可解析、校验与写入数据 schema。P1 四种基线不消费未来已知变量；非空 future-known 的训练配置在模型能力校验处失败。禁止接受后静默忽略。

### 3.2 CSV 与原始报告

使用 csv 按 UTF-8 表头读取。列名保留空格、符号、中文和上游 U+FFFD，不 trim/改名；拒绝重复表头。时间与数值列按 DataSpec 解析；ETT 使用空格分隔时间，Exchange 使用不补零的月/日/时，cement 使用 T 分隔符与六位小数秒。解析器必须保留微秒精度，以合成时间夹具检查格式字符串在所选 Rust 库中的实际含义。空 token 列表显式配置；非空非法数字、无限值、错误日期返回行号/列名及错误类型，日志不输出私有单元格内容或整行。文件缺失直接失败，无网络路径。

原始报告在排序/去重/填充前计算：原始行数、最小/最大时间、按文件相邻行的非递增数、重复时间戳组数、重复组涉及行数、去重移除数、每列缺失数/率、非数值数、常量标记、断档数量与最大间隔。统计命名区分水泥数据的 7 个重复组、14 个涉重行和 7 个移除行。

稳定排序保留原文件行序作为同时间戳的次序；重复策略支持 error、keep_first、keep_last、preserve。ETT 使用行序、不重采样；工程配置才启用固定网格。DatasetReport 同时包含 raw 和 prepared 两部分以及各切分的窗口数，禁止用聚合后的行数替代原始行数。

## 4. 预处理、切分和有效性

### 4.1 通用路径

1. 原始统计 → 显式排序/去重 → 可选网格聚合。网格从 floor(min_timestamp, interval) 至 floor(max_timestamp, interval)；空格点保留。每变量按有效样本数求均值，少于 min_samples 则缺失。
2. 根据未填充的状态列、比较运算、阈值和连续缺失上限生成段。支持 gt/ge/lt/le/eq/ne，拒绝非有限阈值。停止分钟结束段，过长缺失结束段。
3. 段内因果前向填充；陈旧时间从最后一次实际有效观测计算，填充结果不能重置时钟；不跨段。
4. 先固定切分，再拟合 scaler/生成窗口。运行段切分的分母为填充和完整分钟筛选之前的运行段长度，防止缺失比例改变切分。

保留 observed_mask、available_mask、imputed_mask 三种语义。通用工程模式对输入允许受限填充，目标监督只计实际观测；缺失目标可留在窗口中参与掩码损失。P1 基线不能消费缺失输入，历史中仍缺失的窗口被显式排除并计数。复现模式遇到配置数值列缺失直接报错。

### 4.2 cement-twin-v1 的固定语义

严格保留 docs/datasets/cement-yangquan.md §2.2 的四项已批准调整：18 列都填充；填充后仍缺失的分钟不进入任何窗口；重复时间戳保留第一行；按运行段累计分钟数与段中点分配 70/10/20。

- 仅 data_in.csv；1 min 均值，每变量最少 3 个有效原始样本。
- 窑头煤反馈均值 > 5 为运行、≤ 5 为停止、缺失为 unknown。运行段以运行分钟开始和结束；unknown 连续超过 5 min 则在最后一个运行分钟之后结束；两端 unknown 不计入段长；段长至少 30 min。
- 在每个运行段内填充全部 18 列，最大陈旧时间 10 min。
- 按完整运行段长度的中点确定切分，然后取填充后 18 列全可用的连续子段。窗口的历史、gap 和目标范围必须在同一可用子段。
- 此已批准协议把填充后全可用值作为协议目标，target_valid 全为 1；同时保留 imputed 统计。通用工程模式的 observed-only 监督规则不覆盖此协议。
- scaler 只拟合训练切分的 303,009 个可用分钟；均值/总体标准差按数据卡的 1e-3 相对容差检查。

验收必须复现 48 个运行段、529,531 段分钟、37/4/7 个切分段；H=15/30/60 的训练窗口 299819/299384/298514，验证 57806/57746/57626，测试 103417/103342/103192。这些值只作为外部验收期望，产品实现不得硬编码。

### 4.3 scaler 与时间特征

在 CPU 用 f64 计算每列均值和总体方差，构造 batch 时转换为 FP32。scaler.json 记录版本、列名/次序、fit 切分、有效计数、mean、scale、constant 标记及协议指纹；常量列 scale=1。通用模式只使用训练切分的实际有效观测；cement 使用上述协议可用分钟。加载按列名和已存 schema 校验，不重新拟合。

ETT 固定 timeenc=1、freq=h，含 HourOfDay/DayOfWeek/DayOfMonth/DayOfYear 四列，分钟级 ETT 也使用 h。cement 使用 freq=t，首列增加 MinuteOfHour。P1 实现 h/t 两种已使用频率；其他频率明确拒绝。周一=0，年日序从 1 开始；闰年仍按 TSLib 除以 365。时间特征不做标准化。

### 4.4 缓存

缓存放在 runs/cache/data/<key>/，不提交。key = SHA256(源文件内容 SHA256 + 规范化 DataSpec/预处理配置 + 缓存版本)。不依赖 mtime/大小作为唯一指纹。首读一次解析，二读仍可流式计算文件哈希，但 CSV parse_count 必须为 0。

缓存保存未标准化的处理表、时间戳、各掩码、段、原始/处理统计和列 schema，使用独立 safetensors API 与 JSON 元信息；data 不引入 Candle。拟合 scaler 和窗口索引在缓存读取后按当前 train/window 配置构造。文件校验和、形状、版本与 key 全部通过才使用；缓存损坏报错，显式 bypass 可以重算，禁止静默使用损坏表。发布到新目录，保留失败临时目录供诊断，不自动清理。

## 5. 窗口和批次

WindowIndex 只保存合法 origin/segment/split，不预展开所有 [L+H,C] 窗口。历史 [o-L,o)，目标 [o+g,o+g+H)，合法条件含连续网格、段、完整目标归属和掩码策略。验证/测试可读取前一切分历史，但不能跨被断开的运行段。训练起点步长为配置 stride，ETT/cement 均为 1。

ETTh 目标切分 [0,8640)/[8640,11520)/[11520,14400)；ETTm 边界乘 4；末尾额外行不使用。按 2026-09-28 用户批准，初始化与训练采样匹配固定 PyTorch 2.14.0 的 CPU RNG。训练顺序由 seed、模型初始化后的状态和 epoch 重建；参考每轮五个 int64 RNG 消耗独立建模，不访问 test。val/test 永不打乱；所有阶段保留尾批。详见 research/d4-diagnosis.md。

HostBatch 具有真实 B、历史、时间特征、目标、目标 mask、origin 元信息。train 将历史与时间特征构成 ModelInput，将标签/mask 构成独立 Supervision。仅当前批次上传设备；P1 使用同步有界批次读取，不引入未测的异步预取。

## 6. 基线模型与数值证据

统一形状 [B,L,C] → [B,H,C]，随后按已解析目标列名选择 [B,H,C_target]。类型化模型 enum 记录版本、上游 SHA、能力、参数 schema；未验证能力不标注支持。

- LastValue：重复历史最后一个值；无可训练参数。
- SeasonalNaive：对 period=P，预测第 k 步取历史索引 L-P+((g+k) mod P)，k 从 0 开始；要求 1≤P≤L。ETTh 默认 P=24，ETTm P=96，cement P=60；cement 的 P=60 是本次已批准的比较基线参数，不代表已证实的生产周期。
- Linear：沿时间轴做 L→H 的仿射映射；支持共享/individual 参数；不做隐藏归一化。
- DLinear：复制首尾值填充的奇数 moving_avg（默认 25，允许 L=1）；trend 为居中移动平均，seasonal=x-trend；分别沿时间轴做带 bias 的 L→H 投影后相加。默认 individual=false（共享投影）；所有投影权重初始化为 1/L，bias 保留 U(-1/sqrt(L),1/sqrt(L)) 语义。仅使用训练安全基础操作。

Linear/DLinear 对配置的 g 学习对应目标；时间特征通过统一接口保留，DLinear 明确不使用。模型来源与 MIT 许可记录到移植文件、NOTICE 与模型卡。

同名参数采用稳定名字，individual 的变量身份由 schema 绑定。训练模型持有 Var；freeze_for_eval 产生 detached 且独立复制的只读 Tensor 权重，不复用带图输出。验收断言所有冻结权重及输出 track_op=false，且冻结快照不随后续更新变化。

夹具由固定 TSLib 的 DLinear 类直接生成：合成输入、显式共享初始权重、trend/seasonal、输出、MSE、每个参数梯度及一次 Adam 更新；覆盖 individual 两值、L=1、batch=1、非退化常规输入。CPU/CUDA FP32 atol=1e-5、rtol=1e-4；CPU 优化器单步 atol=1e-6。数值不通过时检查实现，不放宽容差。初始化与采样序列的一致性仅针对固定 PyTorch 版本和已验证夹具；不扩展到其他版本或模型。

## 7. 损失、优化器与 Trainer

masked MSE 先用 mask 把无效位置的预测/标签替换为有限值，再平方求和。有效位置非有限值报错。返回 loss_sum Tensor 和 valid_count；valid_count=0 跳过 backward/update 并记录，整个训练 epoch 或整个验证集合无有效标签直接失败。

每个 microbatch 对 loss_sum 反向；按参数累计 detached 梯度和有效元素总数，组末按总数归一化一次再更新，尾部不足 accumulation_steps 的组也更新。全空组不增加 optimizer step。禁止平均 microbatch 均值。更新前检查全部预期梯度是否存在及有限；合法零梯度保留。

复用 StatefulAdam，beta=(0.9,0.999)、eps=1e-8、weight_decay=0。P0 只读 groups() 无学习率修改入口；P1 在 StatefulOptimizer 增加校验后更新学习率的窄接口，保持矩状态、逐参数 step 与现有存储格式兼容，不重建优化器。

ETT：batch=32、accumulation=1、max_epochs=10、patience=3、lr0=1e-4、type1。官方在 epoch e 结束后设下一轮 lr=lr0*0.5^(e-1)，实际各轮使用 lr0,lr0,lr0/2,lr0/4,…；必须测试调用时点。

验证损失使用有效元素总数加权，且验证不打乱。best 采用严格下降；相等损失计入 patience。训练中只迭代 train/val，选定 best 后由 evaluate 访问 test。官方基准保留其验证批均值、val shuffle、每轮 test 日志行为，并在比较报告披露差异；FerroTS 保留父任务批准的改进。D4 数值门槛保持不变。

LastValue/SeasonalNaive 的 train 命令执行数据准备、冻结模型构造、验证和检查点发布，optimizer_steps=0，不伪造 loss 训练曲线。Linear/DLinear 执行完整循环。所有日志仅含统计量，不保存计算图或原始私有行。

## 8. 基本检查点与推理

每个已发布 bundle 必含 model.safetensors、scaler.json、data-schema.json、resolved-config.toml、manifest.json。manifest 记录格式/模型版本、参数名形状、协议和配置哈希、输入/目标顺序、dtype、seed、设备/环境、上游完整 SHA、数据内容指纹、代码 HEAD、dirty 标记及被测源码快照哈希；列出其余文件的大小和 SHA256，不递归包含自己的校验和。

无参数模型保存合法的空张量映射，manifest 标记参数为空；不调用 NamedParams::new 空集。P1 bundle 标记 inference_only，不提供 resume 命令或声称恢复 optimizer/RNG；last 是最后完整 epoch 的推理快照。

采用不可变 generation：先写 runs/<id>/.staging/<generation>/，flush 并校验全部文件，最后写 manifest；同卷 rename 到 checkpoints/<generation>/。best/ 与 last/ 各保存按序编号的引用 JSON；引用先写临时文件再 rename 到新的编号，不覆盖非空目录。逻辑路径 runs/<id>/best 和 last 解析最新已提交引用，再加载 bundle。

失败时旧引用仍有效；尚未发布 staging 不可见。最新正式引用或 bundle 损坏时明确失败，不静默回退旧权重。全部版本保留，不自动删除文件。测试覆盖各发布阶段失败、部分文件、hash/schema/shape/列变化，以及验证完成前不修改当前模型。

evaluate 只使用 bundle 内的 scaler/schema；输入 CSV 列重排时按名字重排，缺列/重复列/协议不符报错。evaluate 复现实验要求数据指纹匹配；predict 接受新文件，但要求兼容 schema，显式记录新指纹。

predict 的回测模式按合法 test 窗口导出；未来模式只用尾部完整历史，生成未来目标时间，不要求未来 y_true。forecast_origin 表示 o 的时间边界，历史时刻严格早于该边界；horizon=1..H，target_time=t[o+g+horizon-1]。未来未知 y_true 为空且 valid=false。

## 9. 指标、输出与资源约束

在 CPU 用 f64 累计有效数量、绝对误差和平方误差，最后求 MAE/MSE/RMSE。标准化与原始单位使用相同有效位置。按变量、horizon 与总体分别维护标准化统计；cement 的原始单位只按变量/步长报告，不输出跨量纲混合总分。MAPE 默认关闭；P1 暂不支持开启。

预测表固定列：run_id,series_id,forecast_origin,target_time,horizon,target_name,y_true,y_pred,valid。P1 series_id 为单一固定序列 ID。保留不同 origin 对同一目标时刻的独立预测，不合并重叠预测。流式写 CSV/CSV.gz，数值使用可往返的精度。

常规训练/评价流式累计指标，默认不导出完整预测。cement H=60 一次全变量导出为 103192*60*18=111447360 行；矩阵不可默认保存所有运行的长表。正式独立复算选择 DLinear/cuda:0/H=60/seed=2021 的全部 test origin，导出窑喂料反馈、窑头煤反馈、分解炉内温度1反馈三列对应的 18574560 行 CSV.gz。输出列过滤只影响导出，不改变模型目标或评价指标。

独立 Python 脚本只读预测表并按 valid 过滤，复算三个变量的原始单位 MAE/RMSE，与 evaluate 的未舍入 JSON 差异必须 <1e-6；不得调用 Rust 指标实现。补充在合成和公共表上测试非法键、重复记录、空有效集与非有限值。日志/可提交报告只输出汇总统计；私有预测表与缓存均在 runs/。

## 10. 正式矩阵、门槛与产物

### 10.1 ETT D4

- 四数据集 × H={96,192,336,720} × seeds={2021,2022,2023}：Rust CUDA 48 次，固定官方 TSLib CUDA 48 次，共 96 次训练。两者同一机器、GPU、数据指纹与协议。
- 另外完成 ETTh1/L96/H96/seed2021 的 Rust CPU 完整训练、evaluate、predict，证明 CPU 闭环；合成集上四基线均做 CPU/CUDA 路径验证。
- 官方使用现有 run_tslib.py，--itr 1、--num_workers 0、GPU 0；其余模型/优化器参数来自 ett-lsf-v1。num_workers=0 为 Windows 执行设置。正式运行前固定完整参数；不减少 epoch、训练样本或预测长度。
- D4 对 16 个组合的 MSE、MAE 分别检查，共 32 项：abs(mean_rust-mean_official) ≤ max(2*std_official,0.02*abs(mean_official))。本计划固定总体标准差 ddof=0，并同时报告 Rust 标准差与双方三个种子原值。
- 比较读取未舍入 metrics；缺运行、重复 seed、运行失败、非有限指标、SHA/配置/数据不匹配均失败。已有一次 ETTh1 smoke 不计入本轮 48 次基准。

### 10.2 水泥与数据门槛

- LastValue、SeasonalNaive、DLinear × H={15,30,60} × device={cpu,cuda:0} × seed={2021}：18 次正式运行。DLinear 完整 max_epochs=10/patience=3；无参数模型按第 7 节执行。报告全部 18 变量的原始单位 MAE/RMSE。
- raw 统计与父任务研究/数据卡一致；运行段、窗口、scaler 与数据卡一致；第二次读缓存 parse_count=0，首读与二读耗时均记录，不设未批准的速度门槛。
- ETT 窗口与实际官方 Dataset.len() 对齐；ETT 时间特征误差 <1e-6；Exchange 行数 7588，时间范围 1990-01-01 00:00 至 2010-10-10 00:00。

### 10.3 证据与收尾

新增验收工具保存命令、配置/代码/数据指纹、开始结束时间、退出码、关键计数和指标到 runs/p1-acceptance/<suite-id>/。完整矩阵以清单校验覆盖率，不能由已存在文件数推断通过。源码或配置变更使相关运行证据失效，必须重跑受影响正式条目。

docs/results/p1-dlinear-ett.md、p1-cement-baselines.md 保存统计表；本任务 research/review.md 保存 AC 与最终质量检查证据。最后运行 fmt、CPU/CUDA Clippy -D warnings、两套工作区测试、doctor 正反例、私有数据检查及内联 trellis-check。全部通过后才本地提交；不推送、不设置 CI。

## 11. 风险与停止条件

1. D4 可能暴露模型/优化器/加载语义差异。先用显式共享权重和单步夹具定位；保留父任务明确要求的加权验证与 test 隔离，不通过时停在该门槛修复，不调宽容差或替换验收。
2. 96 次训练、水泥 CPU 训练与预测导出耗时较长。逐次记录进度，先估算剩余磁盘空间；资源不足时保留已完成产物并报告，不删除旧目录。
3. Windows 原子发布采用新 generation 与新引用名，需真实 Windows 故障注入检查，不能用 POSIX 覆盖目录的假设。
4. Rust 1.98.0、静态 CRT 与现有 MSVC/CUDA 设置按 docs/environment.md 使用。任何系统工具链安装先询问；不卸载保留的 1.78.0。
5. 故障修复保持当前步骤未完成；涉及需求、协议或统计门槛变化时回到规划审批。普通实现错误在已批准范围内修复并重跑。
6. 回滚只使用已保存的提交/文件快照恢复本任务变更；不执行 reset --hard、clean、自动删除 runs/data 或无关文件。
