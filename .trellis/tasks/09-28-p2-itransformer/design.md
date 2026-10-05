# P2 iTransformer 技术设计

2026-09-29 用户后续指示：验证优先 CUDA，不安排常规 CPU 测试/训练；仅在 CUDA 报错时按需运行约 3 个 epoch 的 CPU 诊断。原 CPU 完整对照保留为未执行要求，不记为通过；105 次 CUDA 正式矩阵及数值容差保持。详见 implement.md 的用户验证策略更新。
状态：用户已批准实施；2026-09-29无代理前缀的正常启动命令执行成功，任务为in_progress。实施基点：`1f1b9ec`，分支 `feat/p2-itransformer`。本文件描述目标行为，不代表 P2 已实现或已通过验收。需求来源为本任务 `prd.md`、父任务 D3-D6/X1-X6、计划书第 7/8/10.1/14.1 节。代码与上游核对见 `research/planning-evidence.md`。

## 1. 目标、边界与权威来源

交付官方语义的 `itransformer`、逐层/梯度/单步 Adam 对照、完整 optimizer-step 边界恢复、模型能力表，以及完整 ETT/水泥实验。生产运行只依赖 Rust。参考 Python 只用于开发与正式官方基准。

固定来源：iTransformer `c2426e68ca13f74aaec08045c5c724d8ad328124`；Candle `0.11.0` / `31f35b147389700ed2a178ee66a91c3cc25cc80d`；TSLib 时间特征/数据协议 `4e938a1767106324dd753b2a44832bf870a0252e`；PyTorch CPU RNG `08187d9e0fba026dc8217405802ab5381dc88d90`。继续使用 Rust 1.98.0 与仓库锁文件。

不包含 Informer、任意目标软测量、FlashAttention、低精度、跨硬件逐位恢复、半个 microbatch 内恢复、搜索/回测平台、远程 CI。保留现有数据协议、P1 RNG 对齐和所有历史失败证据。

| 所有者 | 权威对象与拟变更 | 验证边界 |
| --- | --- | --- |
| core | `ModelSpec`、严格配置、能力描述的无张量字段、ReferenceRng 状态 | serde/维度/模型参数适用性；不引入 Candle |
| data | 当前 Dataset/scaler/WindowIndex/HostBatch | 复用 P1；模型不得改变切分、填充或数据指纹 |
| nn | `ForecastModel` 静态 enum、iTransformer、可微层、冻结模型、参数映射 | 输入不含 Supervision；CPU/CUDA 官方夹具 |
| train | Trainer 状态机、累积边界、Adam、scheduler、完整 checkpoint | 恢复不漏步/重步；加载先完整校验后构造可运行 session |
| CLI | models list/train/resume/evaluate/predict | 薄参数与输出适配；复用同一配置、注册和加载逻辑 |
| validation | 固定配置矩阵、官方 wrapper、收据、D4、统计报告 | 完整覆盖、独立复算、无旧 smoke 混入 |

依赖方向保持 cli → train → nn → core，data → core。本次为模型实现、训练状态扩展及必要接口改造；不做相邻清理。

## 2. 配置与模型接口

### 2.1 严格 schema

新增 `ModelSpec::Itransformer`：`d_model`、`n_heads`、`d_ff`、`e_layers`、`dropout`、`activation`、`use_norm`。复现配置显式写出全部有效值。支持官方 GELU/Relu 类型化枚举，GELU 使用 erf 形式，`approximate=none`。`factor`、decoder/label_len/d_layers/dec_in/c_out/distil 等无作用或其他模型专用参数不进入该 variant；未知字段直接拒绝。

在创建 run 目录和 GPU 张量前检查：正维度、`d_model % n_heads == 0`、有限 `0 <= dropout < 1`、合法激活、空 future-known 列集、targets 为 inputs 的有序子集。构批时若仍有 unavailable history，直接报错，不用0填补未支持的输入mask。正式实验固定全变量输出。模型输出先是所有 C 个输入变量；train/evaluate 的现有 `select_targets` 再按名称对应索引选取目标。

新增明确的训练尾批策略，默认保留尾批。ETT iTransformer reference profile 显式丢弃 train 尾批以匹配官方；验证/测试保留全部窗口。cement 保留全部合法 train 尾批。P1 现有配置保持原语义。新增字段采用默认值且默认值不序列化，避免无关地改变旧配置 hash；v1 checkpoint 的原始 config/schema 哈希按存储字节校验。

窗口 L、H、C、输入/目标列序、K 个时间特征及频率都写入绑定元数据。虽然 token 轴权重不显式依赖 C，加载仍严格核对训练时的 C 和列序。权重不得隐式跨 L/H/C/时间特征协议复用。

### 2.2 静态 dispatch

新增 `nn::model`，由 `ForecastModel::{Baseline,ITransformer}` 持有具体模型。公共入口覆盖 construct、named_parameters、weights/buffers、load、freeze_for_eval、forward 和 sampling_rng。迁移 `ModelInput` 到公共模块，并保留原路径 re-export。P1 的 baseline 专属实现继续保留。

`ForwardContext` 显式传入 train/eval 和训练前向计数。模型只接收 history 与可选 history_time；future_time 是既有接口的已知日历字段，iTransformer 不使用 future_time。配置 future-known 数值变量始终报错。冻结视图对参数执行 detach 后 copy，避免共享可变存储；冻结输出不得 track_op。

模型能力表为静态注册数据，至少包含名称/版本、来源仓库/SHA/文件、参数 schema、forecast 任务、同输入变量目标模式、history-time 支持、future-known=false、缺失输入 mask=false、监督 mask 由 train 支持、FP32、编译可用设备和已验证复现状态。`ferrots models list --task forecast --device cpu|cuda:0` 使用同一注册信息；构建支持与实测通过分开记录。

## 3. 模型数值路径

设 X=[B,L,C]，T=[B,L,K]，D=d_model，F=d_ff，A=n_heads，d=D/A，V=C+K。ETT K=4（包括 ETTm 的 freq=h）；cement K=5（freq=t）。无时间特征的合成案例使用 K=0。

1. `mean=detach(mean_L(X))`；`centered=X-mean`；`std=sqrt(pop_var_L(centered)+1e-5)`；`Z=centered/std`。std 不 detach。关闭 use_norm 时跳过完整归一化/反归一化路径。
2. 转置 Z 为 [B,C,L]，把未做窗口归一化的 T 转置后接在变量 token 后，得到 [B,V,L]。Linear(L,D)+bias，然后 embedding dropout。无位置嵌入。
3. 每层 Q/K/V Linear(D,D)+bias，整形为 [B,A,V,d]；scores=QKᵀ/sqrt(d)；train_ops softmax；attention dropout；乘 V，合并 heads，out Linear(D,D)+bias。
4. `x1=LN1(x+dropout(attention_output))`；`y=dropout(GELU(Linear(D,F)(x1)))`；`y=dropout(Linear(F,D)(y))`；`x2=LN2(x1+y)`。LayerNorm epsilon=1e-5，总体方差，末轴归一化。
5. 所有层后执行 encoder final LayerNorm。Linear(D,H)+bias，转置到 [B,H,V]，只取前 C 个变量 token，再按变量 std/mean 反归一化，得到 [B,H,C]。

train_ops 继续禁用无反向的 fused softmax/LayerNorm。新增 GELU 路径先核对 Candle 实际前向/反向支持，以 CPU f64 有限差分与固定 torch 梯度验证。张量转置后的 matmul 明确布局；不通过 detach 或主机数值往返修复梯度错误。

## 4. 参数映射、初始化与随机性

参数名保留官方完整路径，i 为从 0 开始的层号。

| 官方 state_dict / Rust safetensors 名 | 官方 shape | Rust shape / 映射 |
| --- | --- | --- |
| `enc_embedding.value_embedding.weight/bias` | [D,L] / [D] | 原样 |
| `encoder.attn_layers.i.attention.{query,key,value,out}_projection.weight/bias` | [D,D] / [D] | 原样；计算使用 xWᵀ |
| `encoder.attn_layers.i.conv1.weight/bias` | [F,D,1] / [F] | squeeze kernel 轴为 [F,D]；导出回官方时 unsqueeze |
| `encoder.attn_layers.i.conv2.weight/bias` | [D,F,1] / [D] | [D,F] / [D] |
| `encoder.attn_layers.i.norm{1,2}.weight/bias` | [D] | 原样 |
| `encoder.norm.weight/bias` | [D] | 原样 |
| `projector.weight/bias` | [H,D] / [H] | 原样 |

映射版本、名称集合、shape、dtype、源 SHA 和文件校验和一同保存。未知、多余、缺失、重复或不兼容 tensor 均拒绝。iTransformer 当前无持久 buffer，但必须生成合法的空 `buffers.safetensors` 并校验空键集合。

初始化遵守官方构造顺序，使用 P1 固定版本 ReferenceRng；Linear/Conv1d 的均匀范围按 fan_in，LayerNorm 为 weight=1/bias=0。先消耗官方顺序的 RNG，再排序 NamedParams；禁止按字典序重新安排随机初始化。保存初始化后的 RNG state，用于训练采样。fixture 比较三个种子的参数位和前十个 epoch 的排列。

Dropout 继续使用独立 ChaCha8 契约，不声明与 PyTorch CUDA dropout 天然相同。`RngStream.run_step` 对新模型定义为训练前向 microbatch 的单调计数，不能用 optimizer_step 替代：梯度累积中每个 microbatch 必须产生独立 mask。每个 embedding/attention/residual/FFN dropout 站点有稳定 ID，保存在模型/RNG 格式版本契约中。空监督 microbatch 同样消费前向计数；eval 不推进训练流。

## 5. 官方夹具与数值验收

`tools/reference/gen_p2_fixtures.py` 从固定官方类生成合成数据或公共 ETT 夹具，记录完整 config、seed、版本、源 SHA、输入、权重、每层嵌入/attention/FFN/norm/最终输出、loss、逐参数梯度及一次 Adam 后参数/矩/计数。矩阵至少覆盖 K=0/4/5、norm on/off、batch1、L1、常量通道、非连续布局和非退化标准形状。退化案例检查有限性与定义好的输出，不要求退化输入产生非零梯度。

- 主夹具关闭 dropout，固定同权重/输入，比较 CPU/CUDA 全模型和最终 MSE 梯度：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 每个可训练参数都必须在清单中，梯度必须存在、有限，并通过同一容差；不允许只检查最终投影。
- 官方 MSE 下 key bias 数学上可有零梯度，浮点残差不能证明连接性。另用明确记录的中间投影探针损失产生非零 key-bias 梯度；对其余参数也检查至少一个非退化夹具的非零梯度。两端执行完全相同的探针，生产 MSE 不变。
- 一次 Adam 对照使用同一份官方真实 MSE 梯度输入两端，比较全部参数、FP32 一二阶矩和 per-parameter step，atol=1e-6、rtol=0，不排除 key bias。全模型真实 MSE 梯度及连接性探针继续使用原门槛；各后端独立反向后的参数和矩更新差异完整保存为诊断。该 AC3 修订于 2026-09-29 经用户明确批准，详见 research/numerical-gate-diagnosis.md 第 5 节。
- Dropout 开启的训练前向/梯度用显式共享 mask 夹具对照；mask 注入只存在测试/参考适配器，正式训练使用既定 RNG。
- checkpoint reload 后 eval 与 reload 前一致，时间 token 输出没有混入 C 个预测变量；训练不能经由 infer_ops 绕过反向。

## 6. 完整训练状态与恢复

### 6.1 唯一可恢复状态

Trainer 拆出可序列化运行状态及单一推进函数，fresh fit 和 resume 使用相同循环。状态包括：当前 epoch（明确从 1 开始）、下一动作 phase、当前 epoch 的 next_batch_offset、训练排列校验和、global_optimizer_step、next_forward_id、有效元素累计 sum/count、空批计数、split access、已完成 epoch 记录、early stopping 的 best/best_epoch/stale/patience、best generation 身份和结束原因。非有限状态必须拒绝。

phase 至少区分 `train`、`validate`、`complete`。累积组完成并清空后才允许持久化；尾组也必须执行一次真实 step。保存时断言累积 tensor 集和有效数均为空。全空监督组不是 optimizer step；显式 stop-after-step 计数仅计真实更新。

- epoch 中途：记录下一个尚未消费的批次，phase=train。恢复前按保存的 post-init MT19937 状态/epoch 重建排列并验证 hash；从 cursor 继续，禁止重放训练前向。
- epoch 最后一个更新后、验证前：phase=validate。恢复后先做验证，再更新 early stopping 和 scheduler。
- epoch 边界：先完成验证/early stopping，未结束时推进 type1，再提交 phase=train、下一 epoch/cursor0 的状态。终止时 phase=complete，不再推进 scheduler。
- scheduler-state 保存类型、initial_lr、已应用 epoch、当前组 LR，必须与 optimizer groups 一致。P1 的第 1/2 轮 lr0、第 3 轮 lr0/2 时序保持不变。
- rng-state 保存格式/算法版本、完整 post-init MT19937 state（624个u32与index）、采样策略、seed、next_forward_id 和 dropout 站点版本。采样排列由这份不可变状态、trainer中的epoch和cursor重建；不再维护相互独立的第二份采样游标，不以单个seed代替完整状态。

### 6.2 文件与版本

沿用不可变 generation、同文件系统 staging→验证→rename、manifest 完成标志以及 append-only best/last 引用。v2 完整目录为：

```text
manifest.json
model.safetensors
optimizer.safetensors
optimizer-state.json
buffers.safetensors
trainer-state.json
scheduler-state.json
rng-state.json
scaler.json
data-schema.json
resolved-config.toml
environment.json
```

`optimizer-state.json` 是在计划书第 10.1 节基础上保留 P0 已有格式的显式补充；包含 optimizer kind、groups、逐参数 shape/step。manifest 对其余 11 个文件记录 bytes/hash，记录模型/映射/schema/RNG 版本、数据与源快照、L/H/C/K、dtype、设备、变量顺序、父 generation、训练可恢复能力和 checkpoint 边界。首轮验证前的 last 允许 validation_loss/best 为 null；一旦存在观测值就必须有限，不用0伪造验证结果。best 引用只在真实验证改善后出现。

v1 inference-only bundle 继续支持 evaluate/predict，resume 给出明确不支持错误。v2 可用于 evaluate/predict 和可训练模型 resume；无参数基线继续 inference-only。格式读取先按 version/capability dispatch，不把 v1 默认解释成完整状态。

所有输入先加载到临时对象：核对文件集合、路径局限于该 generation、hash、名称/shape/dtype、columns、protocol/data/config、优化器组、scheduler/cursor/RNG、代码与环境身份，再建立可运行 session。任何错误不修改原 session 或已提交 checkpoint；不静默回退到旧引用。严格续训要求原数据指纹、训练参数、运行设备/环境一致；evaluate/predict 的既有设备覆盖与新文件预测规则保留。

### 6.3 发布与 CLI

默认每轮完成时发布 last，验证严格改善时发布 best，避免每个 step 写完整模型。`train --stop-after-steps N` 在第 N 个真实 optimizer step 后发布可恢复 last 并正常停止，用于可验证的暂停；N 不改动 max_epochs 或学习率配置。中断发生在累积组中时，只能恢复最近已提交边界，不声称保存未提交梯度。

`ferrots resume --checkpoint <generation|last|best>` 验证完整状态并创建新 continuation run。保留原 run 与全部文件，写 parent_run/parent_generation/manifest hash 链接。若已存在 best，复制并验证继承的 best generation，使新 run 的 best 自包含并重新发布本地引用；首轮验证前保持 best 不存在。新 run identity 与执行元数据单独保存，原训练 config 字节不改写。运行代码以 runtime source hash 校验；相同源码从 dirty 工作区进入 Git 提交时，记录新旧 Git SHA，不把提交动作解释为算法改变。已完成状态不给出虚构训练步骤。

最小 CLI 不接受恢复时改 seed/batch/模型/数据/优化器/L/H 的覆盖。错误在执行更新前返回非零；不可用 CUDA 不回退 CPU。

### 6.4 恢复证明

同一已验证环境中，dropout=0.1；连续 2N 步对比 N 步→新进程加载→继续 N 步。至少覆盖 N=3 的 epoch 中途案例和跨 epoch/scheduler 的 N=7 案例，累积1/2、尾批、尾累积组、空监督批、最佳轮早于最后轮、patience 临界点。合成数据覆盖全部状态，不使用私有夹具。

CPU 参数/矩/下一预测逐位相同；CUDA 参数/矩/下一预测 `atol=1e-6, rtol=0`；整数状态、排列/cursor、dropout mask、optimizer step、epoch 记录数与 early stopping 决策一致。比较模型和优化器的所有名称。新进程 CLI 案例排除仍持有内存状态造成的假恢复。

负例覆盖缺/多文件、校验和错误、shape/dtype/version、列重排、数据改变、缺参数、矩/组不一致、非法 cursor、RNG index 越界、scheduler LR 不一致、未清空梯度边界，以及各发布阶段注入失败。证明此前已提交引用仍可用，损坏最新引用直接报错。

## 7. 数据、完整实验与报告

### 7.1 固定矩阵

| 组 | 配置与种子 | 正式次数 |
| --- | --- | --- |
| 官方 CUDA ETT | ETTh1/ETTh2/ETTm1/ETTm2 × H96/192/336/720 × 2021/2022/2023 | 48 |
| Rust CUDA ETT | 同上 | 48 |
| Rust CUDA cement | H15/30/60 × 2021/2022/2023 | 9 |
| Rust CPU 对照 | ETTh1/H96/seed2021 完整 train/evaluate/predict | 1 |

共 106 次完整训练，另做数值/恢复/回归测试。CPU 对照不设新的官方指标门槛。CUDA 正式设备固定 cuda:0；官方使用同一物理 GPU。

ETT：L96、全部7变量、h/timeF、heads8/layers2/dropout0.1/GELU/use_norm=true，Adam β=(0.9,0.999)、eps1e-8、weight_decay0、lr1e-4、batch32、accumulation1、epochs≤10、patience3、epoch-end type1、FP32、num_workers0、itr1。ETTh1/H96/192 的 D=F=256；ETTh1/H336/720 的 D=F=512；其余数据集 D=F=128。`label_len=48` 仅传给官方接口，Rust 模型配置不接受 decoder 参数。

官方保留 train/val shuffle+drop_last、test batch1、原验证计算及每轮 test 日志。Rust reference 训练采用 drop_last，validation/test 保留所有窗口、固定顺序、按有效元素加权；Trainer 不访问 test。记录这些协议差异，不修改 D4 门槛。

cement：复用原 TOML 数据处理、L96、18变量、freq=t 与训练参数；模型使用官方未调参默认 D512/F2048/heads8/layers2/dropout0.1/GELU/use_norm=true。不根据 ETT 或测试结果调参。严格保留全部四项已批准协议调整、48运行段、37/4/7段切分、303,009个 scaler-fit 分钟及各 H 的完整窗口数。Exchange 7,588行修正继续受回归检查约束。

D4 的 32 项检查按 `abs(Rust均值-官方均值) <= max(2*官方总体标准差, 0.02*abs(官方均值))`，均值/标准差均用三个指定种子，ddof=0，MSE/MAE 独立检查。比较完整精度值；显示舍入不参与判定。

### 7.2 基准与可追溯性

新增 `run_itransformer.py`，直接调用固定官方 run.py，只替换唯一 fix_seed 赋值并显式传参；拒绝 dirty/错误 SHA、本地文件缺失与不可用 GPU。记录 wrapper hash、源码 hash、resolved args、数据 hash、版本、设备、环境和退出码，保留官方训练/验证/测试日志与结果数组。性能观测不得消费 RNG、改梯度或改模型选择。

P2 suite 使用新 `runs/p2-acceptance/<id>/`；每个 run 有唯一键、命令收据、二进制 hash、dirty source inventory/hash、config hash、checkpoint manifest hash、指标与覆盖统计。正式入口拒绝重复、缺失、错模型、错种子、错 horizon、缩短 epoch budget、伪成功或过期源快照。

P1 官方 DLinear run 不可复用为 P2 官方基准。P2 重试保留失败记录；仅在完整身份与每个产物 hash 校验相同时复用本次正式 suite 的已完成项。复用必须有显式 receipt，不导入 smoke。运行中出现 OOM 不自动缩 batch、不改 H/D/F。

### 7.3 指标、性能和报告

报告路径：`docs/results/p2-itransformer-ett.md`、`docs/results/p2-itransformer-cement.md`、`docs/models/itransformer.md`。仅写统计量。ETT 表列每个种子与均值±总体标准差、32项 bound/difference/pass；cement 表列全部18变量原始单位 MAE/RMSE、三个种子均值±总体标准差，与已验证 P1 基线并列。P1 单种子结果显式标 n=1/std不适用。

独立指标核查沿用流式长表：选择 P2 cement H60/seed2021 的所有103,192个测试 origin，导出 P1 已选三变量、全部60步，共18,574,560行；Python 独立复算6项原始单位 MAE/RMSE，差异<1e-6。全部18变量在九次正常评价中完整计量；导出列过滤不改评价分母。其他八次不生成重复巨型长表。

每次正式运行记录每 epoch 用时、实际处理样本数、稳态 train samples/s、验证和写盘用时。CUDA 阶段边界显式同步。显存报告提供同设备 `nvidia-smi` 50ms采样的最大 used-memory、空闲基线、测量时间段、样本数及丢样数；名称明确为“设备显存采样峰值”，包含其他进程影响，不能声称是精确模型分配峰值。官方另外记录 torch allocator 的 max_memory_allocated/max_memory_reserved。可用的进程计数另列，不把 WDDM 不可用计数填0。CPU-only对照的显存标为不适用。正式预检必须确认设备采样可用；失败时停止并报告，不伪造性能值。不设性能门槛。

## 8. 兼容、质量门与回退

P1 的 v1配置/模型权重/五文件 inference bundle、train/evaluate/predict、tail/weighted metrics、固定初始化/采样 RNG、scaler/cache 保持可读可运行。以合成/公共 v1 fixture 验证旧 bundle 与现有正式结果读取；不修改 archived reports。

P1 test suite 与数据门、CPU/CUDA完整 build/test/Clippy、fmt、Python tests、真实 cuda:0 smoke/cuda:9 拒绝、Trellis 上下文和隐私检查都进入最终门。新 P2 的完整矩阵独立验收；不能用局部通过替代。P2 改动若改变任何已验收 P1数值路径，必须解释差异并完整重跑受影响正式矩阵，保留原记录。

失败时停止依赖步骤，保存失败收据并诊断；禁止放宽阈值、改种子/超参或关闭既有测试。需要系统安装、范围变化或算法语义调整时先征求用户意见。回退通过限定代码修复或经授权的 Git revert；不 reset、不删除原数据、run、checkpoint 或工具链。

本机命令严格复用 `docs/environment.md` 和 `tools/validation/p1_env.ps1`；CUDA MSVC/PATH/静态CRT不另建配置来源。每个命令进程设置 RUSTUP_AUTO_INSTALL=0、Rust1.98.0；直接Cargo固定RUSTC/RUSTDOC。保留Rust1.78.0。Python使用tools/reference的uv环境，HTTPS_PROXY=http://127.0.0.1:7897，PYTHONIOENCODING=utf-8。

## 9. AC 到设计机制

| AC | 机制 | 主要证据 |
| --- | --- | --- |
| AC1 层前向 | 第3/4/5节逐层trace和映射 | CPU/CUDA逐层逐shape误差清单 |
| AC2 参数梯度 | 第5节完整名称对照、MSE与连接性探针 | 每个参数presence/finite/nonzero/error |
| AC3 Adam | 第5节同一官方真实MSE梯度、全部参数/矩/step | CPU/CUDA atol1e-6；独立反向更新差异另记诊断 |
| AC4 恢复 | 第6节状态机与进程重启 | dropout开启的连续/恢复差异及所有状态 |
| AC5 损坏拒绝 | 第6.2/6.4节完整校验及发布负例 | 每类错误非零、原状态未改 |
| AC6 ETT/D4 | 第7.1/7.2节完整96次ETT训练 | 32/32双侧判定、三个种子、身份收据 |
| AC7 cement | 第7节9次全窗口和18变量统计 | 原始单位表、P1样本数、独立复算 |
| AC8 性能 | 第7.3节明确测量口径 | 每run/epoch统计、带采样定义显存峰值 |

P2-R3/R4由配置与load负例证明；P2-R5/R6由完整状态和CLI恢复证明；P2-R7由注册表及models list过滤测试证明；P2-R8包括完整CPU对照。X1-X6由隐私、双数据矩阵、固定来源、名称、实测环境和receipt共同证明。
