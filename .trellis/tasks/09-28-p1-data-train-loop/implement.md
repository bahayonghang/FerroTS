# P1 顺序实施与验收清单

状态：in_progress，用户于 2026-09-28 明确回复“批准，继续”。以下命令按步骤执行，不提前调用未实现的接口。

每步完成后执行该步全部验证命令，记录退出码与证据，再勾选。失败时停在当前步骤修复；不得跳过、缩小矩阵或用局部测试替代正式门槛。内联执行，不派生子代理。

## 0. 审批与上下文门槛

- [x] P0 finish-work：归档 99d13f1，journal f840dfa；实施提交 134737d。
- [x] feat/p1-data-train-loop 从完成收尾的 feat/p0-foundation 创建。
- [x] 阅读 P1 PRD、父任务 PRD/研究、总计划相关章节、docs/datasets/、docs/environment.md、training-foundation.md 与现有 API。
- [x] 用户在本次最终规划摘要之后明确批准：2026-09-28“批准，继续”。
- [x] 批准后核对规划文件无实质变更，执行 task.py validate → task.py start → 读取并执行 trellis-before-dev。

批准后的激活命令（先执行第 1 节通用前置）：

~~~powershell
python ./.trellis/scripts/task.py validate .trellis/tasks/09-28-p1-data-train-loop
if ($LASTEXITCODE -ne 0) { throw 'P1 context validation failed' }
python ./.trellis/scripts/task.py start .trellis/tasks/09-28-p1-data-train-loop
if ($LASTEXITCODE -ne 0) { throw 'P1 activation failed' }
Get-Content .agents/skills/trellis-before-dev/SKILL.md -Raw
~~~

启动时读取 design.md、prd.md、research/planning-evidence.md 和上下文清单；总计划额外读取 §5–7、§10–11、§14–15。无 CONTEXT.md，不创建占位文件。

## 1. 所有命令的环境前置

每个新 PowerShell 进程执行任何命令前首先设置以下变量，不依赖父进程的偶然设置。

~~~powershell
$env:RUSTUP_AUTO_INSTALL = '0'
$env:RUSTUP_TOOLCHAIN = '1.98.0'
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:CARGO_HTTP_PROXY = 'http://127.0.0.1:7897'
$ErrorActionPreference = 'Stop'
$P1Cargo = (rustup which cargo --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 Cargo unavailable' }
$env:RUSTC = (rustup which rustc --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 rustc unavailable' }
$env:RUSTDOC = (rustup which rustdoc --toolchain 1.98.0).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Rust 1.98.0 rustdoc unavailable' }
function Invoke-P1Cargo {
    & $P1Cargo @args
    if ($LASTEXITCODE -ne 0) { throw 'Cargo failed; stop this step' }
}
function Invoke-P1Reference {
    uv run --frozen --project tools/reference python @args
    if ($LASTEXITCODE -ne 0) { throw 'Reference check failed; stop this step' }
}
~~~

直接 Cargo 同时固定绝对路径 RUSTC/RUSTDOC。缺少系统工具链时先询问安装；不卸载 Rust 1.78.0。Python 只复用 tools/reference/.venv，不使用全局 pip 或另一环境。

### CPU 环境

单独 PowerShell 执行通用前置，再移除进程内 CUDA/CUDNN 变量与 Toolkit PATH；不改机器环境。保留兼容的用户 flags。CPU-only 构建门槛使用独立二进制。

~~~powershell
Get-ChildItem Env: | Where-Object Name -Match '^(CUDA|CUDNN)' | ForEach-Object {
    [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
}
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -notmatch 'NVIDIA GPU Computing Toolkit' }) -join ';'
$env:CARGO_TARGET_DIR = Join-Path (Get-Location) 'target/p1-cpu'
$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-msvc'
~~~

### CUDA 环境

另一个 PowerShell 执行通用前置，再导入既有 vcvars64.bat。沿用 docs/environment.md；已有 CL/RUSTFLAGS/NVCC_APPEND_FLAGS 冲突时先核对并保留兼容项，不静默抹去用户 flags。以下示例用于无冲突进程。

~~~powershell
$P1Vs = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
$P1MsvcEnv = & $env:ComSpec /d /c ('call "{0}" >nul && set' -f $P1Vs)
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment failed' }
foreach ($entry in $P1MsvcEnv) {
    if ($entry -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH"
$env:CL = '/Zc:preprocessor /utf-8'
$env:NVCC_APPEND_FLAGS = '-Xcompiler=/MT'
$env:RUSTFLAGS = '-C target-feature=+crt-static'
$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-msvc'
$env:CARGO_TARGET_DIR = Join-Path (Get-Location) 'target/windows-static'
$env:VSLANG = '1033'
~~~

target/windows-static 只用已验证的同一 CRT flags。若 flags 必须变化，改用新的 ignored target 目录并记录；不删除旧缓存。所有子进程继承 RUSTUP_AUTO_INSTALL=0。

## 2. 执行步骤

### Step 01 — 类型化配置与模块契约

- [x] core 实现严格配置、DataSpec/ModelSpec/TrainSpec/OutputSpec、合并顺序与 schema；新增依赖写入 workspace/Cargo.lock，保持 Candle/cudarc 固定。
- [x] 明确 future-known、目标子集、h/t 时间频率与 FP32 的支持边界。

执行记录：2026-09-28，config 测试 4 passed / 0 failed / 0 ignored；check --workspace 退出 0。固定 Rust 1.98.0，CPU 环境。
- 验证（CPU）：

~~~powershell
Invoke-P1Cargo test -p ferrots-core --test config
Invoke-P1Cargo check --workspace
~~~

覆盖嵌套未知键、中文/符号列名、CLI 未设置值不覆盖文件、非法模型参数、错误切分/长度和不支持能力。

### Step 02 — 独立夹具与验收工具基础

- [x] 新增 tools/reference/gen_p1_fixtures.py：从固定上游生成公共 ETT 索引/时间/scaler 参考，以及合成 DLinear 双 individual 的分解、输出、梯度与 Adam 更新夹具。
- [x] 新增 tools/validation/p1_acceptance.py 的 preflight 与证据清单；第 3 节列出的子命令逐步实现。
- 验证：

~~~powershell
python tools/datasets/verify_manifest.py docs/datasets/manifest.toml
if ($LASTEXITCODE -ne 0) { throw 'Public fingerprint mismatch' }
Invoke-P1Reference tools/reference/gen_p1_fixtures.py
Invoke-P1Reference tools/reference/gen_p1_fixtures.py --check
bash tools/check_private_data.sh
if ($LASTEXITCODE -ne 0) { throw 'Private data guard failed' }
~~~

--check 重新计算并比对，不覆盖已有夹具；元信息含上游 SHA、公共/合成来源、版本与容差。

### Step 03 — CSV 与 DatasetReport

- [x] data 实现时间解析、按名取列、raw 统计、排序/去重；缺文件不下载；错误不含私有数值。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-data --test csv_report
- 覆盖 UTF-8、Exchange 日期、水泥格式的合成微秒时间戳、空值/非数值、乱序、7/14/7 重复计数语义、列重排、非法目标和缺文件。

### Step 04 — 网格、运行段与切分

- [x] 实现每列最少样本均值聚合、状态比较、5 min 断段、30 min 最短段、10 min 因果填充、完整运行段中点切分。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-data --test preprocess
- 用合成列/值测试 cement 四项调整，以及 5/6 min 缺失、29/30 min 段、10/11 min 陈旧时间、段首尾 unknown 和停止分钟。不把私有行或私有列名复制到测试 CSV。

### Step 05 — scaler、timeF 与解析缓存

- [x] 实现 train-only f64 scaler、按名保存、h/t 特征、带 key/version/checksum 的处理表缓存；命中不调用 CSV parser。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-data --test scaler --test time_features --test cache
- 覆盖常量列、闰年、实际 ETT timeF <1e-6、内容/配置变化失效、损坏缓存与命中 parse_count=0。

### Step 06 — WindowIndex 与 HostBatch

- [x] 实现目标归属、g/stride、段内窗口、固定 val/test 顺序、seed/epoch shuffle、尾批与输入/监督隔离。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-data --test windows --test batches
- ETTh1/ETTm1、L96、四 H、三切分与官方 len() 夹具逐项一致；覆盖 L=1、B=1、无合法窗口、gap 跨段及前切分历史。

### Step 07 — 四基线与训练数值

- [x] nn 实现 LastValue、SeasonalNaive、Linear、DLinear、稳定参数名和 freeze_for_eval；补来源/许可。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-nn --test baselines --test dlinear_reference
- 验证（CUDA）：Invoke-P1Cargo test -p ferrots-nn --features cuda --test baselines --test dlinear_reference
- 检查显式共享权重的前向/梯度夹具、双 individual、bias 初始化、复制 padding、常量输入、L=1、B=1、空参数与冻结快照。CUDA 不可用必须失败。

### Step 08 — 掩码损失、指标与学习率

- [x] train 实现 masked loss、有效元素梯度累计；core 实现 f64 指标；在 StatefulOptimizer 加入安全 lr setter，保持 moments/step，按 epoch-end 实现 type1。
- 验证（CPU）：

~~~powershell
Invoke-P1Cargo test -p ferrots-core --test metrics
Invoke-P1Cargo test -p ferrots-train --test loss --test accumulation --test dlinear_update --test optim
~~~

- 验证（CUDA）：Invoke-P1Cargo test -p ferrots-train --features cuda --test loss --test accumulation --test dlinear_update --test optim
- 在 train 的 dlinear_update 测试对照官方单步更新，避免 nn 反向依赖 train。对比不同大小/mask 的 microbatch 与等效大 batch；覆盖全空组、NaN 无效标签、非有限有效值、lr 序列及 setter 不重置状态。

### Step 09 — 检查点与发布事务

- [x] 实现不可变 generation、best/last 选择器、五必需文件、manifest、完整校验与冻结加载；last 不支持恢复训练。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-train --test checkpoint
- 验证（CUDA）：Invoke-P1Cargo test -p ferrots-train --features cuda --test checkpoint
- 覆盖空参数文件、roundtrip、中断保留旧 best、最新引用损坏、缺文件/坏 hash/列/shape/dtype/协议不符及校验前不修改模型。使用新临时路径，不删除用户已有目录。

### Step 10 — Trainer 完整循环

- [x] 串联 batch、模型、loss、Adam、type1、冻结验证、早停、best/last；无参数模型 optimizer_steps=0。
- 验证（CPU）：Invoke-P1Cargo test -p ferrots-train --test trainer
- 验证（CUDA）：Invoke-P1Cargo test -p ferrots-train --features cuda --test trainer
- 记录 split 访问，断言训练未迭代 test；验证输出无图；全 epoch/val 无有效标签失败；平局 patience、尾累计及 best 加载正确。

### Step 11 — CLI、配置与流式输出

- [x] 实现 data inspect/train/evaluate/predict，保留 doctor，错误指明实际子命令，保留已有二进制名 ferrots。
- [x] 提供 configs/reference/ett-h1/h2/m1/m2.toml、configs/engineering/cement-twin.toml，以及验收 Exchange/合成配置。
- [x] 完成指标 JSON、CSV/CSV.gz、未来模式、按名重排、最终配置落盘与导出列过滤。
- 验证（CPU）：

~~~powershell
Invoke-P1Cargo test -p ferrots-cli --test cli_pipeline --test doctor
Invoke-P1Cargo run -p ferrots-cli '--' --help
Invoke-P1Cargo run -p ferrots-cli '--' data inspect --help
Invoke-P1Cargo build --release -p ferrots-cli
~~~

- 验证（CUDA）：Invoke-P1Cargo build --release -p ferrots-cli --features cuda
- 记录正式二进制 SHA256；源码变动需重建，并使受影响实验失效。

### Step 12 — 真实数据门槛与缓存

- [x] 用 Rust CLI 检查 ETT 四文件、Exchange、水泥；先校验真实文件指纹，再比较统计。
- [x] 水泥 raw 统计、48 段、三切分/三 H 窗口数、18 列 scaler 对齐；逐窗口检查段与目标归属。
- [x] 用新 cache 命名空间测首读与二读，记录总耗时、哈希耗时、解析数、命中；二读 parse_count=0；不删除旧缓存。
- 验证：

~~~powershell
$P1Suite = 'runs/p1-acceptance/' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$P1CpuBin = 'target/p1-cpu/x86_64-pc-windows-msvc/release/ferrots.exe'
$P1CudaBin = 'target/windows-static/x86_64-pc-windows-msvc/release/ferrots.exe'
Invoke-P1Reference tools/validation/p1_acceptance.py preflight --suite $P1Suite --cpu-bin $P1CpuBin --cuda-bin $P1CudaBin
Invoke-P1Reference tools/validation/p1_acceptance.py data --suite $P1Suite
~~~

保存实际 suite ID。后续新 shell 复用该 ID，不混合不同 suite 的证据。启动 Rust CPU 子进程时剥离 CUDA 环境；GPU 子进程使用已验证环境。

### Step 13 — ETT CPU 完整闭环

- [x] ETTh1/L96/H96/seed2021，在 CPU-only 二进制完成 max_epochs10/patience3 的 DLinear 训练、best evaluate 与 test predict。
- 验证：Invoke-P1Reference tools/validation/p1_acceptance.py ett-cpu --suite $P1Suite
- 加载 best 的预测/指标/键和指纹保持一致；全部样本参与。本步骤不计入 D4 CUDA 三种子矩阵。

### Step 14 — 官方 TSLib 48 次基准

- [x] 四数据集 × 四 H × 三种子从现有 runner 发起，单 GPU 串行；失败立即停止，成功条目保存可校验 receipt。
- 验证：Invoke-P1Reference tools/validation/p1_acceptance.py ett-reference --suite $P1Suite
- 固定 seeds=2021/2022/2023、L96、label_len48、individual=false、moving_avg25、M/7 变量、Adam/lr1e-4/batch32/epochs10/patience3/type1、FP32、GPU0、num_workers0、itr1。不使用旧 smoke 或论文表格补缺项。

### Step 15 — Rust 48 次训练与 D4

2026-09-28：原 suite 20260928-0734 完成双方 48 次，D4 为 31/32。受控三种子重放确认随机序列差异；用户批准对齐固定 PyTorch RNG，门槛不变。修复与证据见 research/d4-diagnosis.md。新 suite 为 20260928-0842-rng；已重跑数据与 CPU 闭环，严格校验后关联原官方 48 次，全部 Rust 48 次重跑完成，D4 为 32/32。运行日志 20260928-084159-1494760.log，D4 日志 20260928-085719-1808282.log；门槛未改变。

新增验证：gen_rng_fixtures.py 与 --check；core/nn reference_rng；data batches/windows；CPU/CUDA trainer/dlinear_update；两套 release build。夹具覆盖 15 初始化场景、150 采样轮次和 4 个原始随机种子。

- [x] 同机 cuda:0 运行完整 Rust 48 次矩阵，检查每次配置、代码与数据身份。
- 验证：

~~~powershell
Invoke-P1Reference tools/validation/p1_acceptance.py ett-rust --suite $P1Suite
Invoke-P1Reference tools/validation/p1_acceptance.py d4 --suite $P1Suite --report docs/results/p1-dlinear-ett.md
~~~

- 32 项 MSE/MAE 判定全部通过；公式及 ddof=0 见 design.md §10。报告双方逐种子值、mean/std、阈值/差值/pass。
- 失败则本步不完成。修复实现/数据/运行错误后重跑受影响正式条目；不得按测试结果选种子、调宽门槛或静默修改超参数。

### Step 16 — cement-twin 正式 18 次运行

- [x] LastValue/SeasonalNaive/DLinear × H15/30/60 × cpu/cuda:0，seed2021，所有样本遵守已批准协议。
- 验证：Invoke-P1Reference tools/validation/p1_acceptance.py cement --suite $P1Suite --report docs/results/p1-cement-baselines.md
- DLinear 全量训练评价；无参数基线完成 fit/检查点/评价。每组合报告全部 18 变量的原始单位 MAE/RMSE、有效数与耗时。

### Step 17 — 预测独立复算与错误矩阵

- [x] 导出 DLinear/cuda:0/H60/seed2021 的三变量全 test 长表 CSV.gz，共18574560行；独立复算三个变量的 MAE/RMSE，六项绝对差均 <1e-6。
- [x] 实现 tools/validation/recompute_predictions.py；测试 D4 缺/重复种子、非有限数、错误配置/指纹和阈值边界。
- 验证：

~~~powershell
Invoke-P1Reference tools/validation/p1_acceptance.py recompute --suite $P1Suite
Invoke-P1Reference -m unittest discover -s tools/validation/tests -p 'test_*.py'
Invoke-P1Cargo test -p ferrots-cli --test regression_matrix
~~~

回归必须覆盖尾批、全缺失标签、常量列、L=1、B=1、重复时间戳、跨段、非法目标、列重排、缺文件、CPU-only 请求 CUDA、实际 cuda:9、损坏检查点；每项具有明确断言。

### Step 18 — 全量门槛与内联 trellis-check

- [x] 按真实实现更新 backend/data-pipeline.md、training-loop.md、index；training-foundation.md 只改受影响契约；补 ett.md 的 type1 调用时点、模型卡与 NOTICE。
- [x] 全部产品/测试改动完成后执行完整检查；修复使正式证据失效时回到相应步骤。

CPU 环境：

~~~powershell
Invoke-P1Cargo fmt --all --check
Invoke-P1Cargo build --workspace
Invoke-P1Cargo clippy --workspace --all-targets '--' -D warnings
Invoke-P1Cargo test --workspace
~~~

CUDA 环境：

~~~powershell
Invoke-P1Cargo build --workspace --features cuda
Invoke-P1Cargo clippy --workspace --all-targets --features cuda '--' -D warnings
Invoke-P1Cargo test --workspace --features cuda
Invoke-P1Cargo run -p ferrots-cli --features cuda '--' doctor --device cuda:0 --training-smoke-test
& $P1Cargo run -p ferrots-cli --features cuda -- doctor --device cuda:9
if ($LASTEXITCODE -eq 0) { throw 'cuda:9 must fail' }
~~~

正式证据与任务检查：

~~~powershell
Invoke-P1Reference tools/validation/p1_acceptance.py final --suite $P1Suite --report .trellis/tasks/09-28-p1-data-train-loop/research/acceptance.md
python ./.trellis/scripts/task.py validate .trellis/tasks/09-28-p1-data-train-loop
if ($LASTEXITCODE -ne 0) { throw 'P1 task validation failed' }
bash tools/check_private_data.sh
if ($LASTEXITCODE -ne 0) { throw 'Private data guard failed' }
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' }
~~~

- [x] 读取并内联执行 .agents/skills/trellis-check/SKILL.md，逐项检查 R1–R17、AC1–AC10、D4/X1–X6，写 research/review.md。正式缺项不能以 focused tests 替代。
- [x] 通过后才勾 PRD 验收项，记录实际 CPU/CUDA 测试数、失败/忽略数、退出码和矩阵覆盖。

### Step 19 — 本地提交与后续交接

- [ ] Step18 全通过后读取 git-commit 技能；只 stage 本任务代码、公共/合成夹具、配置、统计报告、spec 与任务记录，保留无关改动。
- [ ] 暂存后再次运行 bash tools/check_private_data.sh 与 git diff --cached --check；核查无 data/、runs/、target/、.venv/ 或上游缓存。
- [ ] 使用 local git commit，不推送、不创建 PR、不配置 CI。核对提交中产品源码与被测快照一致；不一致则返回验证。
- [ ] 报告 SHA、分支、工作区状态和正式验收结果，生成下述交接 Prompt。P1 finish-work 的实际待办状态如实保留，不提前归档。

## 3. 验收工具与证据契约

p1_acceptance.py 仅用于开发验收，不参与产品训练。

| 子命令 | 输入/输出与失败条件 |
| --- | --- |
| preflight | suite、CPU/CUDA 二进制；检查环境、九公共指纹、私有指纹、上游 SHA、磁盘、配置/源码快照；输出 suite.json |
| data | ETT/Exchange/水泥 raw/prepared/scaler/window/cache 与独立期望比对 |
| ett-cpu | 一次完整 CPU ETTh1 train/evaluate/predict、checkpoint roundtrip |
| ett-reference | 调用原 run_tslib.py；保留每次参数、SHA、数据、机器、seed、未舍入指标和退出码 |
| ett-rust | 固定矩阵调用 Rust GPU 二进制；检查配置/数据/源码一致 |
| d4 | 48+48 条覆盖和32项门槛；缺项、坏指纹、阈值失败退出非零 |
| cement | 18条完整运行及18变量指标；可提交输出仅统计量 |
| recompute | 单独流式 Python 复算；六项差异、预测键与行数全部检查 |
| final | 检查全部 AC、receipts、数值门槛、覆盖与代码身份；任一缺失失败 |

每条 receipt 含完整命令、开始/结束时间、退出码、配置/数据/环境/上游/代码指纹；失败不能伪装成功。重跑只复用身份完全相同且产物校验通过的条目，旧 smoke 不自动导入。所有原始运行产物在 runs/。

实验期间未提交代码记录 code_sha=HEAD、dirty=true、源码快照 SHA256。最终提交后核对快照，再在交接给出 commit 对应关系；不声称旧 HEAD 包含被测的新实现。

## 4. 需求与验收映射

| PRD | 步骤 | 判定证据 |
| --- | --- | --- |
| R1–R3 | 01/03/11/12 | 严格配置、CLI 合并、raw 报告、Exchange/水泥统计 |
| R4–R6/R8 | 04/05/06/12 | 聚合/运行段边界、四项调整、缓存零解析 |
| R7/R9 | 02/05/12 | ETT timeF <1e-6、train-only scaler、cement18列容差 |
| R10 | 06/12 | ETT实际len()、逐窗口目标/段归属 |
| R11/R13 | 08/10/17 | 元素加权、全空失败、冻结验证、test隔离 |
| R12 | 02/07/08/13–16 | 四基线、双individual、梯度/单步与正式矩阵 |
| R14 | 09/11/13/17 | 五文件、best/last、发布故障、schema与校验 |
| R15 | 08/11/16/17 | 输出schema、18变量报告、独立三变量复算 |
| R16/R17 | 01/11/12/13 | 四CLI命令、四ETT+cement配置、CPU闭环 |
| AC1 | 02/06/12 | ETTh1/ETTm1 × 四H × 三split 的len() |
| AC2 | 02/05/12 | ETT来源时间夹具与最大误差 |
| AC3 | 14/15 | 官方48+Rust48、32项D4全部通过 |
| AC4 | 03/12 | Exchange7588行与起止时间 |
| AC5 | 04/06/12 | raw统计、运行段/窗口、逐窗口同段断言 |
| AC6 | 16 | 三模型 × 三H × 两设备，共18次 |
| AC7 | 17 | 三变量MAE/RMSE六项差异<1e-6 |
| AC8 | 05/12 | 首读/二读耗时与二读零解析 |
| AC9 | 03–11/17/18 | 完整回归与CPU/CUDA全部测试 |
| AC10/X1 | 02/18/19 | 夹具来源、私有检查、暂存区审查 |

## 5. 风险与回退

- Step04/06：切分/缺失顺序影响全部下游；修正 data owning 模块，不在 Trainer 临时删除窗口。
- Step07/08：bias、padding、梯度权重和 epoch-end LR 先通过夹具，再长训练。
- Step09：发布失败保留旧引用和诊断产物，不删除目录重试。
- Step14–17：资源不足或条目失败则保留证据并停在当前步，不缩减正式矩阵。
- 回退基线 f840dfa；只恢复本任务文件，不执行 reset --hard/clean，不删除未知文件或 runs/data。

## 6. 提交后的交接 Prompt 模板

交付时用实际结果替换尖括号字段，不提前填入通过结论：

> 继续 FerroTS，仓库 D:\Documents\Code\Rust\Exp\ferrots，Trellis 管理。分支 feat/p1-data-train-loop，HEAD=<完整本地SHA>，工作区=<实际状态>，未推送。P1的19步及内联trellis-check=<结果>。正式证据见 <suite路径>、P1 research/acceptance.md和research/review.md；ETT官方/Rust各48次、32项D4、水泥18次、独立三变量复算及CPU/CUDA全量检查结果为<摘要>。请先按实际状态执行P1 finish-work，再从当前分支创建下一阶段分支，补齐P2规划后等待批准再start。内联执行，不派生子代理；沿用docs/environment.md。任何命令前禁用rustup自动安装、固定1.98.0，直接Cargo固定RUSTC/RUSTDOC；系统工具链安装先问，保留1.78.0；Python复用tools/reference环境与代理；不删除、不推送、不配置远程CI，私有数据不提交。
