# P2 iTransformer 顺序实施计划

状态：2026-09-29按用户指定的无代理前缀命令启动成功，任务已由planning进入in_progress，当前任务指向P2。此前三次拒绝保留在research/planning-evidence.md；实现和验证进展见research/implementation-evidence.md。分支 `feat/p2-itransformer`，基点 `1f1b9ec`。执行方式固定为主会话内联；不派生子代理。

## 2026-09-29 用户验证策略更新

用户要求后续尽量不执行 CPU 测试，优先使用 CUDA；仅在 CUDA 报错后按需使用约 3 个 epoch 的 CPU 训练辅助诊断。此要求覆盖下文原计划中的常规 CPU 重跑与第 15 步 CPU 完整对照安排。保留此前已获得的 CPU 证据；未执行的 CPU 正式对照不得记为通过。48 次官方 CUDA ETT、48 次 Rust CUDA ETT 和 9 次 CUDA cement 共 105 次正式训练及所有 CUDA 数值/D4/完整性门保持不变。短 CPU 诊断不计入正式矩阵。以下原 106 次要求作为原始基线保留，最终报告必须列出本次用户要求导致的 CPU 对照未执行项。

## 0. 阶段门与命令环境

先让用户审阅本任务 PRD/design/implement/上下文清单与规划摘要。只有用户随后明确批准，才执行 `task.py start`，随后读取 `trellis-before-dev`，然后按编号顺序实施。任何步骤失败先保存证据、诊断并修复；通过该步骤验证后再进入下一步。新的范围或验收语义调整需要重新审阅。

Rust和参考计算命令使用以下环境变量；本地Trellis启动命令单独执行，不添加无需联网的代理前缀。禁止系统安装、自动下载 Rust、卸载1.78.0、删除文件/目录、推送或配置远程CI。

```powershell
$env:RUSTUP_AUTO_INSTALL='0'
$env:RUSTUP_TOOLCHAIN='1.98.0'
$env:PYTHONIOENCODING='utf-8'
$env:HTTPS_PROXY='http://127.0.0.1:7897'
$env:UV_PYTHON_DOWNLOADS='never'
```

Rust 命令先按模式 dot-source `. ./tools/validation/p1_env.ps1 -Mode cpu` 或 `-Mode cuda`，使用 `Invoke-P1Cargo`；该函数调用固定Cargo，helper同时固定RUSTC/RUSTDOC。模式切换使用独立进程，CPU与CUDA现有target目录不混用。CUDA完整复用MSVC、PATH、CL、NVCC_APPEND_FLAGS、RUSTFLAGS和目标平台设置。改变CRT前停止，不复用旧对象。

Python统一使用 `uv run --frozen --project tools/reference python ...`。日志保存在忽略的 `runs/p2-development/` 或正式suite；现有helper生成的 `runs/p1-development/` 日志也可引用，但报告注明实际路径。任何命令返回非预期退出码立即停止依赖命令。

新增runner、fixture生成器、配置及CLI选项均是下列步骤的拟交付接口；本次规划不执行尚未实现的命令。

## 1. 批准后激活与加载规范

- [x] 用户已回复“批准实施”；已核对分支feat/p2-itransformer、基点1f1b9ec、仅P2规划文件未提交及P1归档，保留既有规划文件。
- [x] `uv run --frozen --project tools/reference python .trellis/scripts/task.py start .trellis/tasks/09-28-p2-itransformer`。2026-09-29单独执行，exit=0。
- [x] 读取 `.agents/skills/trellis-before-dev/SKILL.md`；按backend索引加载training-foundation/data-pipeline/training-loop及两个thinking guides。主会话手工加载两份jsonl中的规范和research。
- [x] 验证：task状态为in_progress、当前任务指向P2、分支仍为feat/p2-itransformer；没有更新依赖或安装工具链。

## 2. 建立固定官方夹具与参考入口

- [x] 新增 `tools/reference/gen_p2_fixtures.py`、`run_itransformer.py`，复用现有环境、upstream pin和收据约定，不改官方checkout。
- [x] 导出合成/公共FP32权重、每层输出、MSE梯度、Adam单步后参数/矩，写映射表和夹具来源说明；确认空buffer格式可被Rust读取。
- [x] 为每个参数定义连接性探针；key bias不用浮点舍入残差证明连接性。保留官方输入requires_grad=True失败记录及生产输入下38个参数梯度的研究结论。
- [x] 验证：同seed重导出tensor内容一致；完整名称/shape/dtype清单、源SHA、依赖版本、三个初始化种子与十轮采样序列一致；输入不含私有值。空buffer先做Python往返，Rust读取在步骤10验证。运行相关Python单测和隐私检查。

## 3. 严格配置与能力描述

- [x] 修改 `ferrots-core/src/config.rs`，增加iTransformer variant、有效参数范围、future-known拒绝及reference训练尾批策略。
- [x] 定义模型注册描述所需的无张量类型；所有P1配置默认值与序列化身份保持兼容。
- [x] 验证：合法默认/显式config；非法heads/D、维度、dropout、激活、decoder参数、未知字段、future-known、目标列；P1原config字节/hash及错误语义回归。
- [x] 执行CPU `Invoke-P1Cargo test -p ferrots-core`，以及受影响配置Python检查。

## 4. 统一模型接口并回归基线

- [x] 在nn新增公共ModelInput、ForwardContext、ForecastModel静态enum和registry；保留baseline原路径re-export。此步enum先接通Baseline，步骤6再加入已实现的ITransformer分支；不得把未实现模型标为可用。步骤3新增的ModelSpec在旧Baseline构造器中明确拒绝，保持中间步骤可编译。
- [x] 把Trainer、checkpoint与evaluation/CLI中的Baseline硬编码改为统一dispatch；已有baseline实现与参数名保持原样。
- [x] 验证：四基线的forward、参数/冻结、train/evaluate/predict行为不变；冻结快照不共享可变参数；现有CPU workspace tests全通过。
- [x] 此阶段不引入第二套Trainer或数据加载器。

## 5. 训练安全层与GELU

- [x] 在nn中实现/复用biased Linear、LayerNorm、多头投影与精确GELU；只补缺失能力。
- [x] 验证CPU f64有限差分、CPU/CUDA FP32前向与梯度，覆盖非连续布局、常量、L1、B1和shape错误。
- [x] 保留no_bwd源检查与负面梯度测试。两个设备相关测试均运行；CUDA不可用不得skip。

## 6. iTransformer模型、映射与初始化

- [x] 新增nn模型，按design第3节实现归一化→倒置嵌入→Post-Norm encoder→final norm→projector→slice→反归一化。
- [x] 按第4节名称表加载/导出权重；实现固定官方构造顺序的ReferenceRng初始化，保存post-init采样状态。
- [x] 验证K0/4/5、use_norm on/off、L/H/C/K绑定、目标子集与时间token切除、Conv1d squeeze/unsqueeze；逐项核对初始化参数位与三个种子。
- [x] CPU/CUDA模型前向和已有baseline测试通过后进入下一步。

## 7. 完整数值门

- [x] 运行所有官方逐层夹具，两设备均检查每层embedding/attention/FFN/norm/final输出。
- [x] 输出全部参数的gradient presence/finite/nonzero-probe及MSE误差清单；任何未列出参数视为失败。
- [x] 按2026-09-29批准的AC3修订，以同一份官方真实MSE梯度执行两端Adam，比较全部参数、矩和计数，atol1e-6/rtol0。全模型前向/真实MSE梯度保留atol1e-5/rtol1e-4。独立反向更新差异完整记录为诊断，不改变生产梯度或超参数。
- [x] 验证：全部案例通过。失败时保存原fixture与误差，诊断映射/算子/顺序；未通过不得以正式训练替代数值门。

## 8. Dropout与可恢复随机流

- [x] 把ForwardContext接入各dropout站点；区分microbatch前向计数与optimizer计数，保留P0 ChaCha8编码。
- [x] core增加严格版本化ReferenceRng状态读写，保持P1算法和既有位级夹具。
- [x] 验证共享mask的官方前向/反向；train/eval行为、不同站点/微批、空监督批、累积1/2、RNG非法状态；同context的CPU/CUDA mask完全一致。

## 9. Trainer状态机与调度恢复

- [x] 把fit局部状态收敛为一份TrainingState，fresh/resume使用同一推进逻辑；持久化design第6.1节所有字段。
- [x] GradientAccumulator暴露可证明为空的提交边界；尾组更新、空组跳过与前向计数独立记录。
- [x] scheduler明确next-action，支持epoch中途、最后更新后待验证、下一轮和complete四种时序；首轮前best可为空。
- [x] 验证：无需文件系统的连续/暂停内存状态机对照；type1不重复；test访问次数恒为0；strict best/tie/patience/尾组行为保持。

## 10. v2完整checkpoint事务

- [x] 扩展checkpoint版本dispatch，复用P0 optimizer-state格式，写design第6.2节12个文件；空buffers文件真实存在。
- [x] 完整读取校验先于session构造；v1保持inference-only可读，v1 resume明确拒绝。
- [x] 同卷staging和append-only refs；保存失败保留staging，最新引用损坏直接报错。
- [x] 验证文件集合/哈希/名称/shape/dtype/协议/数据/列序/组/cursor/RNG/LR负例；各发布阶段故障注入；此前generation仍可用，活状态不部分更改。

## 11. resume CLI与续训产物

- [x] 增加 `resume --checkpoint` 和 `train --stop-after-steps`；恢复前拒绝不兼容数据、模型、环境和训练参数。
- [x] 新建continuation run并记录父manifest；存在历史best时复制并验证，没有best时不制造虚假引用；原run保持完整。
- [x] 接通v2的evaluate/predict，继续按存储scaler处理推理文件。
- [x] 验证CPU CLI端到端、非法参数、首轮前暂停、completed状态、损坏路径；错误均非零，调用者不承担模型状态组装。

## 12. CPU/CUDA进程重启恢复门

- [x] 合成数据dropout0.1，连续2N与N+resume+N；N3/N7、累积1/2、尾批/尾组、空监督、跨epoch/LR、非最后best/patience。
- [x] 两侧使用相同初始参数与全部数据顺序；中断分支退出进程，从磁盘启动新CLI。
- [x] CPU参数/矩/预测逐位相同；CUDA atol1e-6/rtol0；整数状态、RNG/mask/cursor/下一批、选择结果完全一致；时间与run id不参与位级比较。
- [x] P0原低层resume与P1全部checkpoint测试继续通过。没有部分微批恢复承诺。

## 13. 固定配置、正式runner与性能观测

- [x] 新增16个展开后的ETT iTransformer reference配置、3个cement H配置及小型synthetic配置；不能只改horizon而遗漏ETTh1 D/F变化。
- [x] 接通 `models list` 的任务/设备过滤，生成模型卡、映射/许可证/NOTICE；复现状态只在验收后标为通过。
- [x] 新增 `tools/validation/p2_acceptance.py`、`p2_quality.ps1` 和runner测试；复用收据/hash/D4模式，既有P1工具保持兼容。
- [x] 设计runner命令：preflight、numerics、resume、data、ett-reference、ett-rust、d4、cement、recompute、final，各接收明确 `--suite`。receipt允许验证后续跑，禁止无记录跳项。
- [x] 增加同步分阶段计时、稳态样本吞吐、50ms设备显存采样及官方allocator峰值；测量不消耗随机流。不支持的进程显存标不可用，不记0。
- [x] 验证runner能拒绝错seed/遗漏/重复/短训练/坏hash/改source/改config；OOM保存失败且不缩批。冻结正式配置，形成suite preflight收据。

## 14. 正式实验前完整回归

- [x] 按2026-09-29用户更新执行CUDA workspace build、Clippy --all-targets -- -D warnings、workspace test；未执行独立CPU-target重跑；fmt通过。Python执行完整unittest。
- [x] 运行真实cuda:0 doctor training smoke与cuda:9拒绝。
- [x] 复查P1数据门：公共manifest、ETT全窗口/time/scaler、Exchange7,588、cement四协议调整/48段/37-4-7/303,009 fit及全部H窗口；缓存二次parse_count=0。
- [x] 读取公共/合成v1 bundle并完整evaluate/predict，保持P1数值/尾批/RNG回归；如发现P1数值路径改变，先诊断，再完整重跑受影响正式矩阵。
- [x] `task.py validate`、隐私与diff检查通过。此后如运行源码改变，旧收据必须失效并重跑受影响正式项。

## 15. 完整CPU对照

- [ ] CPU-only构建完成ETTh1/L96/H96/seed2021 iTransformer，max_epochs10/patience3；训练到既定正常结束，随后best evaluate与全部test predict。
- [ ] 验证完整窗口覆盖、checkpoint/summary/best一致、无训练test访问、设备/性能记录和指纹；这是完整训练，不能替换为几步smoke。

## 16. 官方CUDA ETT完整矩阵

- [x] 固定官方iTransformer源码，4数据集×H96/192/336/720×seed2021/2022/2023，共48次，单卡逐次执行。
- [x] 保留官方train/val drop_last、test batch1、type1及best选择；只按已批准protocol设置num_workers0/GPU0/itr1/seed，不改超参。
- [x] 核验每项完整参数/数据/环境/代码身份、正常终止、best权重、完整测试shape和MSE/MAE、epoch/吞吐/显存；汇总必须48/48。
- [x] 不使用P1的DLinear官方实验、论文数字或历史smoke补项。

## 17. Rust CUDA ETT完整矩阵与D4

- [x] 同一配置与种子集完成48次Rust训练及best evaluate，保留所有epoch和收据。
- [x] 对16配置分别计算MSE/MAE：三种子均值/总体标准差，32项双侧D4全部满足原门槛。
- [x] 生成 `docs/results/p2-itransformer-ett.md`，含每seed、bound、difference、pass与正式suite身份。
- [x] 失败先保留原矩阵并诊断，不换种子、不改hyperparameter、不放宽门槛；若修复影响其他项，按身份规则重跑。

## 18. 水泥完整矩阵与独立复算

- [ ] cuda:0，H15/30/60×seed2021/2022/2023，共9次完整训练/evaluate；每次验证全部18变量、原始单位与全测试窗口。
- [ ] 报告三个种子的均值±总体标准差，并列P1已验收基线；P1单种子标n=1/std不适用，不能构造假标准差。
- [ ] H60/seed2021导出指定三变量的全部103,192 origins×60步×3变量=18,574,560行CSV.gz；不采样。
- [ ] 独立Python流式复算6项MAE/RMSE，绝对差<1e-6；产物保留在runs/。报告只含统计量。

## 19. 最终质量门与内联trellis-check

- [ ] 按已批准的CUDA优先策略运行完整fmt/CUDA build/Clippy/workspace tests、Python全部测试、真实doctor两项、Trellis验证、隐私和diff检查；测试失败/忽略均为0。基线测试数CPU65/CUDA71/Python9不得因删除测试下降，新用例另计。
- [ ] 最终suite重新读取全部105次CUDA训练项（完整CPU对照保持未执行）、32个D4判定、9次cement的18变量、全行独立复算、数值门和恢复门；验证每份receipt/source/binary/config/manifest hash及集合完整性。
- [ ] 按 `trellis-check` 内联审查P2-R1至R8、AC1至AC8、X1至X6及跨层流；写research/acceptance.md与review.md，记录实际通过/失败/忽略数和任何未测边界。
- [ ] 更新backend model/resume规范、模型卡/能力状态及环境记录，只把已实测组合记为支持；不要改P1历史验收结论。
- [ ] 全部门通过后才允许进入本地提交。失败保留事实，不宣告P2完成。

## 20. 本地交付与后续收尾

- [ ] 按Trellis Phase3.4和git-commit约定准备仅P2范围的本地提交；提交前与staging后执行 `bash tools/check_private_data.sh` 和diff检查。
- [ ] 核对已测runtime source与提交的规范化Git blob；data/、runs/和私有派生值不进入提交。禁止推送。
- [ ] 更新task的实施commit及真实验收状态，给出交付摘要；finish-work归档/journal按用户下一次明确指示执行。

## 验收对应表

### 可复用验证命令

每个进程先执行第0节环境前缀。以下CPU/CUDA两组在不同进程运行，任一命令失败即停。

```powershell
. ./tools/validation/p1_env.ps1 -Mode cpu
Invoke-P1Cargo fmt --all --check
Invoke-P1Cargo build --workspace
Invoke-P1Cargo clippy --workspace --all-targets -- -D warnings
Invoke-P1Cargo test --workspace
uv run --frozen --project tools/reference python -m unittest discover -s tools/validation/tests -p 'test_*.py'
uv run --frozen --project tools/reference python .trellis/scripts/task.py validate .trellis/tasks/09-28-p2-itransformer
bash tools/check_private_data.sh
git diff --check
```

```powershell
. ./tools/validation/p1_env.ps1 -Mode cuda
Invoke-P1Cargo build --workspace --features cuda
Invoke-P1Cargo clippy --workspace --all-targets --features cuda -- -D warnings
Invoke-P1Cargo test --workspace --features cuda
Invoke-P1Cargo run -p ferrots-cli --features cuda -- doctor --device cuda:0 --training-smoke-test
```

cuda:9是预期失败检查，不用会对任何非零立即throw的Invoke-P1Cargo包装：使用已固定的 `$script:P1Cargo` 直接执行相同doctor命令，捕获日志，明确断言exit=1以及实际device initialization/CUDA_ERROR_INVALID_DEVICE错误。这项检查不能把编译失败当作预期成功。

正式runner拟定调用形式为 `uv run --frozen --project tools/reference python tools/validation/p2_acceptance.py <stage> --suite <suite-dir>`。stage按步骤13-19取preflight/numerics/resume/data/ett-cpu/ett-reference/ett-rust/d4/cement/recompute/final；suite-dir由一次正式preflight创建并保持不变。p2_quality.ps1必须复用既定环境helper，Python命令统一经过uv，任务路径指向P2；不得直接调用仍绑定旧P1任务路径的p1_quality.ps1来宣称P2质量门通过。

### 需求映射

| 需求 | 顺序步骤 | 完成证据 |
| --- | --- | --- |
| R1 / AC1 | 2,5,6,7 | CPU/CUDA逐层fixture全部通过 |
| R2 / AC2,AC3 | 2,6,7,8 | 名称/shape映射、所有参数MSE梯度与非零探针、Adam更新 |
| R3 | 3,6,10,11 | L/H/C/K/columns绑定及加载拒绝 |
| R4 | 3,13 | 严格模型schema与future-known/decoder/heads负例 |
| R5 / AC4 | 8,9,10,12 | 完整状态、dropout开启的新进程连续/恢复对比 |
| R6 / AC5 | 10,11,12 | resume CLI、损坏状态矩阵、原状态不变 |
| R7 | 3,4,13,19 | 同一注册表供校验、构造、列表及模型卡 |
| R8 / AC6 | 13,15,16,17 | CPU1、官方CUDA48、RustCUDA48、D4 32/32 |
| R8 / AC7 | 18 | CUDA9、全部18变量、三seed统计、全行复算 |
| AC8 | 13,15,16,17,18 | 全106 run的逐epoch计时/吞吐/标明口径的显存统计 |
| X1-X6及兼容 | 14,19,20 | 隐私/双数据/来源/命名/环境/receipt及P1完整回归 |

## 关键停止条件与恢复点

- 步骤7数值失败：停在模型/映射边界；不得进入正式矩阵。
- 步骤12恢复失败：停在状态/发布边界；不得缩短N或关闭dropout来获取通过。
- 步骤14发现P1回归：修复并验证P1受影响行为，再冻结正式suite身份。
- 步骤16-18失败：保留原run和原因，以严格收据续跑；不得删除或覆盖失败记录。
- 环境需安装、验收需改变、正式OOM需改超参：停止相关步骤，说明证据并请求用户决定。
- 代码回退只操作已授权范围；不得删除数据、旧工具链、run或他人改动。

## 2026-09-29 20:44 America/Chicago execution checkpoint

Step 17 remains active: official CUDA 48/48 and Rust CUDA 9/48 were revalidated. Rust ETTh1/H96/H192/H336 completed all seeds. Existing PID 43756 owns ETT followed by D4; PID 40828 waits for the complete D4 pass before cement, recompute, final CUDA quality and final acceptance. Both creation times match the handoff. ETTh1/H720/seed2021 is active.

No stage was restarted. Steps 17-19 remain unchecked until their full completion conditions pass. The original CPU control remains unexecuted. The final main-session review and report reconciliation remain pending. See research/acceptance.md and the appended evidence/review entries. No commit or archive is authorized in this continuation.

## 2026-09-29 cement tooling finding

Step 18 has a confirmed consumer defect: three functions in p2_acceptance.py assume a list where the Rust/P1 metric contract supplies a dictionary. A candidate-only correction passed 8/8 synthetic regression methods. The candidate is outside the frozen tools and is not counted as formal acceptance.

Continue the existing ETT process. Preserve the original suite and receipts. Before applying the correction, resolve the explicit tool-identity transition described in research/cement-metric-contract-diagnosis.md. Do not overwrite suite.json or relax hash verification. Final quality, final acceptance and final inline trellis-check remain open.

## Verification continuation: 2026-09-29 21:20 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 18/48 passed frozen entry validation. 12/12 complete-configuration diagnostic comparisons satisfy the unchanged D4 bound. Full D4 remains pending. The original shell PID 43756 and continuation PID 40828 retain their recorded creation times. Active run: rust-ETTh2-itransformer-cuda-h336-s2021-20260929-211846-941822. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/verification-update.json; runs/p2-development/continued-verification-20260929-211647-8010923/process-latest.json.

The cement candidate passed an additional real-artifact report integration check: nine accepted P1 entries, 18 variables per entry, 108 report rows and 324 baseline metric cells. P2 inputs were synthetic; zero formal CUDA runs were added. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/cement-integration.json. The earlier 8 synthetic methods remain passed. The production tool and old suite/receipts are unchanged. Evidence-transition selection remains pending.

Reports and the model card now show the verified snapshot. The registry remains pending. Final CUDA quality and final inline trellis-check remain open. No duplicate training, CPU training, task restart, commit, archive or push occurred.

## Approved evidence transition, 2026-09-29

The user explicitly selected versioned migration with full verification of unaffected ETT receipts and reruns of affected gates and final quality. See research/tooling-transition.md. The idle downstream waiter PID40828 was deliberately stopped; original ETT/D4 PID43756 remains active. The original waiting status is historical. The new suite and corrected production validator remain pending until parent ETT48 and D4 32/32 finish. No training was interrupted or duplicated.

## Half-matrix checkpoint: 2026-09-29 21:43 America/Chicago

All 48 official and 24 Rust ETT entries were revalidated, including identities, full coverage, best checkpoints and receipts. All 16 available diagnostic comparisons meet the original bound. Full D4, cement, recomputation and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-midpoint-20260929-214323-0225823.json. ETT/performance/model-card snapshots now reflect 24 Rust entries and 72 measured runs. No formal performance gate was generated.

Versioned migration is explicitly selected. Replacement executor PID32276 waits for the original ETT/D4 PID43756. All 52 repository plus staged Python tests passed, with zero failures/errors/ignored. The same-thread heartbeat ferrots-p2-cuda provides continued observation and recovery; it does not create a subagent or duplicate training. Frozen production files remain unchanged.

## Retained interruption and scoped recovery: 2026-09-29 23:47 America/Chicago

The original ETT process tree exited between the 23:40:34 observation and the successor's 23:41:18 D4 check. ETTm1/H336/seed2023 retained four epoch records but has no final training summary or command receipt. The final exit code and cause are unknown. No matching Application or System level 1-3 event was found in the inspected 23:40:00-23:42:30 interval. The current CUDA doctor passed; this diagnostic does not establish the cause of the earlier exit. No CUDA failure was established and no CPU diagnostic was scheduled.

The old successor PID32276 stopped at verify-original-d4 because d4.json was absent. No dependent gate ran and no frozen tool was installed. Original logs and their copies/hashes are retained under runs/p2-development/interruption-20260929-234423-4955379. The earlier empty interruption-20260929-234348-4775294 directory is a failed evidence-copy attempt; its path composition was corrected without changing original evidence.

After confirming that all original worker processes were absent, the main session started a separate hidden PowerShell execution process, PID67224, created 2026-09-30T04:45:58.0116320Z. The new script has no wait on an old PID. It revalidates existing entries, completes missing ETT and original D4, then performs the already approved versioned transition and remaining gates. Script: runs/p2-development/tooling-transition-20260929-2129/recover-versioned-20260929-2345.ps1. SHA256: 32fbcabfd92aa51d1689b64cb7c8002353b297ec4096a46b8cd0bf792cb49e7f. Launch receipt: runs/p2-development/tooling-transition-20260929-2129/recovery-launch-20260929-234557-9868650/launch.json. Current ownership is recorded in active-executor.json in that staging directory. The process was observed alive after the launching shell returned; creation-time checks use UTC ticks.

Official CUDA ETT remains 48/48. Rust ETT has 32 verified completed entries. The latest complete-configuration diagnostic has 20 passes and zero failures; full D4 remains pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/configuration-diagnostics-20260929-230734-090871.json. The missing H336/seed2023 is a new attempt; the interrupted attempt is not counted. Runtime, binary, seeds, configuration and thresholds remain unchanged. P2 remains in_progress; no commit, archive or push occurred.

## ETT and migration checkpoint: 2026-09-30 02:54 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 48/48 completed with normal exits. Both original and successor D4 passed 32/32 with ddof=0, zero failed comparisons and unchanged thresholds. The complete parent audit revalidated all 96 entries, prerequisite gates, quality and original receipts. Successor suite runs/p2-acceptance/20260929-cuda-v3 contains 96 byte-identical imports with bound original receipt paths. The tested cement/provenance correction is installed. Runtime source, CUDA binary, profiles, seeds and tolerances remain unchanged. Current successor quality passed Rust 102 and Python 52, with zero failed or ignored tests. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-transition-checkpoint-20260930-025430-462084.json.

PID67224 stopped after ETT and original D4 passed. Its final error was MSVC initialization failed; retained stderr reports The input line is too long. Repeated initialization expands PATH: an isolated reproduction passed from lengths 4045 and 5802, then failed from length 7559. Two fresh-process probes passed with the required Rust 1.98.0, CUDA 13.3 and static-CRT flags. No CUDA failure was established. The frozen environment helper remains unchanged. Evidence: runs/p2-development/tooling-transition-20260929-2129/msvc-path-reproduction-20260930-024540-7128072/diagnosis.json and runs/p2-development/tooling-transition-20260929-2129/isolated-env-check-20260930-024757-6311963/result.json.

Replacement executor PID73220 was created at 2026-09-30T07:48:38.2326413Z. It keeps an uninitialized parent environment and creates a new child process for every stage. Python stages load p1_env once. Quality stages let p2_quality load the environment once. Script: runs/p2-development/tooling-transition-20260929-2129/continue-isolated-20260930-0246.ps1; SHA256 fd28baca7f56f8fc9c6235d5ac50380d5354110b3a42ca539b69817782040f8c. Ownership remains in active-executor.json. No ETT training was repeated. Cement is active; independent recomputation, final quality, final acceptance and final main-session sign-off remain pending. P2 remains in_progress. No commit, archive or push occurred.

## Cement interruption and recovery: 2026-09-30 03:11 America/Chicago

Executor PID73220 and training PID38688 exited between the 02:52:38 and 02:56:15 observations on September 30, 2026. The first H15/seed2021 attempt has zero completed epochs and no final command receipt. The exit code and cause remain unknown. The interval overlaps a host application restart; causality was not established. No CUDA failure was established. Original logs and hashes remain in runs/p2-development/tooling-transition-20260929-2129/host-interruption-20260930-030009-2678822/interruption.json.

After verifying that all prior P2 process trees had ended, the main session launched executor PID67188 at 2026-09-30T08:06:42.4105976Z through Windows CIM. The observed parent is WmiPrvSE.exe. The executor was verified alive after the launch command returned; survival through an application restart was not tested. The executor retains an uninitialized parent environment and creates one child environment per stage. Only cement, independent recomputation, final CUDA quality and final acceptance are scheduled. ETT96, the audited versioned migration, current quality and D4 were reused.

At 2026-09-30T03:11:56.2431998-05:00, cement has 0/9 complete entries. Active run: rust-cement-itransformer-cuda-h15-s2021-20260930-030647-900688; training PID25580, completed epoch records 0. Current owner: runs/p2-development/tooling-transition-20260929-2129/active-executor.json. Checkpoint: runs/p2-development/tooling-transition-20260929-2129/cement-recovery-checkpoint-20260930-031156-482696.json. The task remains in_progress. Independent recomputation, final quality, final acceptance and final main-session Trellis sign-off remain pending. No CPU training, commit, archive or push occurred.

## First cement epoch: 2026-09-30 05:44 America/Chicago

The active CUDA H15/seed2021 run completed epoch 1: 299,819 training windows, 9,370 optimizer steps, zero skipped empty batches, train MSE 0.10933995531677243 and validation MSE 0.10041589545685217. Training took 9,299.952 seconds and validation took 7.171 seconds; the combined epoch time was 9,307.124 seconds (155.1 minutes). Checkpoint timing was 10.181 seconds. Steady throughput was 32.239 windows/s. Best and last checkpoint directories are present. These observations cover one epoch of an unfinished run; they do not establish final checkpoint validity or add a completed formal entry. Cement remains 0/9; executor PID67188 and training PID25580 remain active. The full matrix, independent recomputation, final quality and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/cement-first-epoch-20260930-054416-417122.json.

## Performance repair continuation: 2026-10-01

User-requested training stop and scoped CUDA performance repair are complete. The repaired development build passed the full CUDA quality suite. Synthetic and bounded real-data diagnostics are retained under runs/p2-development/gpu-performance-repair-20260930. Formal P2 acceptance remains in_progress because the runtime identity changed and the full matrix was not restarted during repair.

## Repaired-runtime v4 suite: 2026-10-02 09:16 America/Chicago

The performance repair changed the runtime source to 513a8c4ac49d04d5c528c2b4e1be6abcd349bfce5b9a533391c191cd049951b2 and the CUDA binary to a8cc02c50065b714cecd96a55edc2a018a6854461a484d47382add1173585379. The runner binds every receipt to the suite source and binary, and an import requires the same identity. The v3 evidence therefore cannot satisfy current-source acceptance.

The user selected a full v4 suite with unchanged tooling: runs/p2-acceptance/20261002-cuda-v4. Official ETT48 is rerun under the new suite identity. Executor PID22708 was launched through Windows CIM at 2026-10-02T14:16:33.5628392Z. Script: runs/p2-development/p2-v4-20261002/execute-v4.ps1, SHA256 678c4228f2d52a93ace9dc3819c91e9e3d6f8aa7bd1ce3251d1a342bae969230. Each stage runs in a fresh child process. Preflight passed. Remaining stages: initial quality, data, numerics, resume, official ETT48, Rust ETT48, D4, cement9, recompute, final quality and final acceptance. Steps 16-19 are re-executed for v4. Main-session report reconciliation and final review remain required. v2/v3 suites remain retained historical evidence. No commit, archive or push.

v4 numerics attempt 1 failed at 2026-10-02T14:22:14Z. The CUDA numerics test passed (1 passed, 0 failed), but the report path was captured with a trailing ANSI reset code. Cause: PowerShell 7.6.6 rendered native stderr lines with ANSI codes in the CIM-launched environment. A direct reproduction showed NO_COLOR=1 removes the codes. The executor stage header now sets NO_COLOR=1; validation tooling is unchanged. Failed evidence remains in runs/p2-development/p2-v4-20261002/executor-20261002-091634-8818359 and the v4 numerics command receipt. Executor PID63504 was relaunched through Windows CIM at 2026-10-02T14:23:37.7407807Z, script SHA256 099601ad3dfdc9975b3184118adb1c8ca59033839056ecaf31c40f7822cab4c4. Existing preflight, quality and data gates were skipped; numerics passed.

v4 resume attempt 1 failed at 2026-10-02T14:24:28Z. The CUDA recovery test passed (1 passed, 0 failed; ten cases). The runner could not decode stdout as UTF-8: cargo replayed cached Chinese MSVC linker warnings and the nested PowerShell wrote them through the GBK console code page. The frozen binary SHA256 remained a8cc02c50065b714cecd96a55edc2a018a6854461a484d47382add1173585379. The executor stage header now also sets the console input/output encoding to UTF-8; validation tooling is unchanged. Failed evidence remains in runs/p2-development/p2-v4-20261002/executor-20261002-092338-4911385 and the v4 resume command receipt. Executor PID48028 was relaunched through Windows CIM at 2026-10-02T14:26:00Z, script SHA256 afe421e882ab0ad89b1bd5f0cd938bd2b2c0637aa1cfa81d41fc5cccec103a8b. Resume passed; official ETT48 started at 14:26:44Z.

## Registry status and v5 suite: 2026-10-03 07:39 America/Chicago

The v4 suite passed at 2026-10-03T10:47:37Z with registry status "pending" in the tested source. Step 13 requires the pass status after acceptance; step 20 requires the committed source to equal the tested source. The status string is compiled into the binary and covered by the runtime source hash. The user selected the status change plus a full v5 rerun.

crates/ferrots-nn/src/registry.rs now sets iTransformer reproduction_status to "p2_cuda_formal_pass"; crates/ferrots-nn/tests/model.rs asserts the same value. No other source changed. New runtime source: fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055. New CUDA binary: be54cf2e1f6b393d20cadae41faf7bb3e73a4af463f7347e038762488747e4ad. fmt and the CUDA workspace build passed before launch.

Suite: runs/p2-acceptance/20261003-cuda-v5. Executor PID24056 was launched through Windows CIM at 2026-10-03T12:38:58Z. Script: runs/p2-development/p2-v5-20261003/execute-v5.ps1, SHA256 b30f7ad2a2e42e8276bf8790a2e514b7235a79ffc1672878ec88e96194c42ed6; it differs from execute-v4.ps1 only in suite, staging path, expected source and labels. Launch receipt: runs/p2-development/p2-v5-20261003/launch.json. Preflight passed. v4 and older suites remain retained historical evidence. Steps 16-19 are re-executed for v5. No commit, archive or push.

## v5 executor exit: 2026-10-03 08:19 America/Chicago

The scheduled check at 2026-10-03T13:19:32Z found that executor PID24056 no longer exists; Get-CimInstance and Get-Process both return no process. status.json still reports state running, phase ett-reference, last written at 12:45:09Z. console.log ends at "running / ett-reference"; no failed record was written. The exit time, exit code and cause are unknown. In the 07:44-08:19 local interval, Application/System level 1-3 events show only Schannel 36871 and DCOM 10010 (GameBar); no relation to the exit was established.

The ett-reference stage child survives: pwsh PID56192 (parent 24056, created 12:45:09Z) -> uv/python chain -> python PID28660 (created 13:15:26Z). Official ETT has 24 completed entries; official-ETTm1-h96-s2021 is active. ett-reference.stderr.log is empty. Without the executor, no later stage (ett-rust, d4, cement, recompute, final quality, final acceptance) will start after this stage ends.

Evidence: runs/p2-development/p2-v5-20261003/executor-20261003-073859-9869953/ (events.jsonl, status.json, console.log, ett-reference.stdout.log, ett-reference.stderr.log). Nothing was stopped, restarted or rerun. The scheduled job 112ef141 was deleted. Thresholds, seeds and tooling are unchanged. A user decision is required on how to continue.

User decision 2026-10-03 08:3x America/Chicago: let the surviving ett-reference child finish official ETT48, then relaunch execute-v5.ps1 unchanged. The runner verifies existing entries and skips them (p2_acceptance.py ett_reference). runs/p2-development/p2-v5-20261003/relaunch-v5.ps1 launches only after all surviving P2 processes exit, 48 official entries exist and the stage log contains "PASS official CUDA ETT: 48/48"; otherwise it waits or exits 2 without launching. A session-only 15-minute check (job e3e5a08e) runs the script and, after launch, creates an hourly monitor.

## v5 relaunch: 2026-10-03 10:13 America/Chicago

The surviving ett-reference child completed official ETT 48/48 and exited; its log ends with "PASS official CUDA ETT: 48/48". relaunch-v5.ps1 then started execute-v5.ps1 unchanged (SHA256 b30f7ad2a2e42e8276bf8790a2e514b7235a79ffc1672878ec88e96194c42ed6) through Windows CIM. Predecessor: PID24056 (exit cause unknown). New executor: PID17988, created 2026-10-03T15:13:51.9053400Z. Records: runs/p2-development/p2-v5-20261003/executor-20261003-101352-9001675. Launch receipt: runs/p2-development/p2-v5-20261003/relaunch.json.

Preflight, initial quality, data, numerics and resume were skipped on their existing result files. ett-reference verified all 48 existing official entries without retraining and passed at 15:14:22Z. ett-rust is running. Wait job e3e5a08e was deleted; an hourly session-only monitor replaces it. No commit, archive or push.

## v5 host power-off: 2026-10-03 10:57 America/Chicago

The scheduled check at 2026-10-04T01:50:27Z found executor PID17988 absent and no P2 process alive. status.json still reports running / ett-rust (last written 15:14:22Z); no failed record exists. ett-rust.stderr.log is empty. Cause: System event User32 1074 at 2026-10-03T10:57:17-05:00 records a planned "power off" initiated by StartMenuExperienceHost.exe for user DESKTOP-516BF74\lyh; EventLog 6006 (clean shutdown) followed at 10:57:34. The host booted again at 20:13:01. No CUDA failure was established.

Rust ETT has 27/48 completed entries. The active attempt rust-ETTm1-itransformer-cuda-h192-s2021-20261003-105623-332087 had completed epoch 1 (1,071 optimizer steps) and has no final summary or command receipt; it is not counted. Official ETT remains 48/48. Evidence: runs/p2-development/p2-v5-20261003/executor-20261003-101352-9001675/ and the incomplete run directory. Nothing was restarted or rerun. Monitor job 16d819d2 was deleted. Thresholds, seeds and tooling are unchanged. A user decision is required on how to continue.

## v5 resume after power-off: 2026-10-05

The user asked to check the current Trellis task and continue implementation. Live checks matched the power-off record: no P2 process, executor.lock free, official entries 48, Rust ETT entries 27, cement entries 0, runtime source fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055, CUDA binary hash equal to suite.json. The incomplete ETTm1/H192/seed2021 directory has no entry receipt and stays uncounted.

execute-v5.ps1 was relaunched unchanged (SHA256 b30f7ad2a2e42e8276bf8790a2e514b7235a79ffc1672878ec88e96194c42ed6) through Windows CIM. Executor PID47012, created 2026-10-05T08:39:49.0917910Z, parent WmiPrvSE.exe PID9020. Records: runs/p2-development/p2-v5-20261003/executor-20261005-033949-8301611. Receipt: runs/p2-development/p2-v5-20261003/relaunch-20261005.json. The earlier relaunch.json was not overwritten.

Preflight, initial quality, data, numerics and resume were skipped on existing result files. ett-reference verified official ETT 48/48 without retraining and passed at 08:40:31Z. ett-rust then verified the 27 existing Rust entries and started rust-ETTm1-itransformer-cuda-h192-s2021-train. Remaining after that entry: 20 Rust ETT runs, then D4, cement 9, recompute, final quality and final acceptance. No commit, archive or push.
