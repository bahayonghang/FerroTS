# Informer 实施计划

状态 in_progress。用户已于 2026-10-13 授权按顺序实施；开发门已通过，正式矩阵预算已于 2026-10-13 获用户批准。证据见 research/implementation-evidence.md。

1. [x] 冻结TSLib来源/许可、官方profile、fixture manifest、AC、CUDA预算和磁盘估算。
2. [x] 实现PrefixSum、采样score、top-query、ReplaceRows及蒸馏算子；CUDA前后向夹具、边界、scan峰值和shape门通过（AC1–AC3）。
3. [x] 接入配置/静态派发、embedding、encoder/decoder、历史尾部+未来零值；全参数真实损失梯度与标签隔离检查（AC1、AC6）。
4. [x] 命名BatchNorm buffer及采样身份贯穿snapshot、manifest、save/load、resume、freeze、预测，检查损坏buffer和验证隔离（AC4、AC5）。
5. [x] CUDA独立进程连续/恢复训练比对，开启dropout/BN/采样，覆盖B=1、尾批及CLI预测（AC4–AC6）。
6. [x] 跑L96/192/384/768/1536复杂度门、CUDA短训练及阶段性能；通过后冻结suite并审批正式矩阵预算（AC2、AC3、AC9）。
7. [ ] 官方ETT48+RustETT48，校验身份/退出/全窗口，计算32项D4；失败保留证据，不改门槛（AC7）。
8. [ ] 水泥9次，108格分变量报告；H60/seed2021三变量18,574,560行独立复算6项（AC8）。
9. [ ] 最终质量与独立证据审阅，更新模型卡/registry/报告，核对105次及未执行项，提交并按真实状态收尾（AC9、AC10）。

## 验证与产物

使用docs/environment.md固定Linux Rust/CUDA环境与target/linux-cuda。

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --features cuda --locked -- -D warnings
cargo test --workspace --features cuda --locked
uv run --frozen --project tools/reference python -m unittest discover -s tools/validation/tests -p 'test_*.py'
bash tools/check_private_data.sh
git diff --check
```

每个增量先跑相关CUDA行为测试，最终跑适用质量门。CUDA workspace含既有CPU数值用例，不额外安排CPU-only套件或CPU训练。P3专用fixture/runner在步骤1定义CLI/manifest后实现，不虚构已存在命令。

正式产物在runs/p3-acceptance/<suite-id>/，报告为docs/results/p3-informer-{ett,cement,performance}.md。逐增量提交，失败仅回退局部修改，不覆盖旧suite。CPU约三epoch仅限CUDA失败后必要诊断。

## 当前执行边界

步骤 7 未完成。20261014 旧冻结套件完成 official48 + Rust48，D4 30/32，仅 ETTh2/H96 的 MSE、MAE 失败；水泥0/9。它不验收当前 Philox、torch.randint 与 checkpoint 源码。

2026-10-18 当前源码的 ETTh2/H96、seed2023 前两批 CUDA 对照固定了数据、初始权重、完整采样索引和精度设置。12处 dropout mask 每批逐元素匹配官方，Philox offset 分别为 0→188、188→376；第一处主要前向分歧在 encoder 第0层 1×1 conv：官方 cuDNN TF32 与 Rust FP32 线性计算差约 8.9e-4。仅在诊断中关闭官方 cuDNN TF32，前两批 loss、梯度和 Adam 矩接近；这不修改固定官方行为或正式要求。当前最小代码修复是保存/加载/恢复 Philox offset 并更新 Informer RNG 身份，MT19937 采样游标按 post_init/epoch/batch 重放；CUDA 独立进程恢复已重验。完整数值与命令见 research/implementation-evidence.md。

当前运行源码摘要 `8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47`；ETTh2/H96、ETTh1/H720 与预算对照案例各完成32步 CUDA 暂停运行，该阶段未形成新 D4。旧 ETTh1/H720 的 D4 MSE/MAE 余量分别为 0.018374219934、0.011763713868；当前源码余量未测。正式训练损失的旧 Rust/官方差使用了不同尾批聚合口径，不能视为逐步差。当前短运行估算的训练预算约146.64 GPU小时（均衡分配假设）至198.89 GPU小时（慢卡外推），尚未计验证等开销，超过旧约80 GPU小时授权；186 GiB磁盘量仍是旧估算。AC4/AC5已用当前CUDA证据复验，AC7–AC10未完成。生产早停、D4 阈值和 seed 未改；Informer 仍为 preview。新正式 suite 尚未冻结，需先评审源码身份、局部证据和预算并取得新的正式运行批准，保留旧失败 attempt。

同一源码的 ETTh2/H96、seed2023 官方/Rust 单案例完整训练与测试已完成：官方 MSE/MAE 为 2.9419164657592773/1.3997275829315186，Rust 为 2.6772834590976093/1.3073967803726594；仍有明显差距，单 seed 不构成 D4。随后的前5批对照验证完整采样索引与60处 dropout 掩码一致，Philox offset 到940；关闭官方 cuDNN TF32 的诊断控制使第5批更新参数最大差从约5.30e-4降到2.04e-5；一轮训练控制也使与Rust的参数平均绝对差从1.87e-4降到1.17e-4，但没有消除差距。对保留的独立官方epoch1控制、完整运行官方epoch2与Rust epoch1权重，用同一官方评估器、2,785个窗口、相同稀疏采样在验证/测试集各重放8次：官方epoch2相对epoch1控制的验证MSE始终低0.0115–0.0125，测试MSE却高约0.241；Rust epoch1相对官方epoch1控制测试MSE低约0.0246。等批次与按标签加权口径均无选点排序翻转。该对照量化了独立epoch1控制和选中epoch2之间的单案例指标差，但未保留完整运行的epoch1权重，不能严格归因全程选点；也没有发现现行验证MSE选点的实现错误。不能凭此改生产早停或推断三seed D4。按先前失败即停止扩展的边界，ETTh1/H720 的完整训练守护未启动。ETTh2 实测验证、检查点、评估耗时及配对运行约396 MB产物已记入 research/implementation-evidence.md；五批轨迹另行保留，GPU1、H720和水泥的完整开销仍未测，预算尚不能签准。步骤7及AC7–AC10保持未完成，不冻结或启动新正式 suite。

2026-10-18 同运行检查点诊断增量：原完整官方运行仅留 epoch2 best，无法复原其 epoch1 权重；新 `attempt-002` 在同一次默认精度官方训练中保留 epoch1/epoch2 完整权重和 BatchNorm buffer，原验证MSE选中 epoch1（0.822675330 对 0.863464478），与旧完整运行选中 epoch2 的轨迹不同。验证与测试各2,785窗口、8组共同稀疏采样的四种 MSE/MAE 口径均支持新运行 epoch1 优于 epoch2；新官方 epoch1 与 Rust epoch1 的测试 label-weighted MSE 均值分别为 2.680610086、2.676746030。旧/新官方第100训练批已出现 loss 差，早于新观察钩子，具体首发训练差异仍未定位；本次新权重不能回填旧完整运行的缺失权重。首个失败 attempt 及修复后的完整 attempt 均保留，独立只读核查通过。无生产修复、无新 D4；详情及哈希见 research/implementation-evidence.md。训练-only预算仍为 146.64–198.89 GPU小时；单案例开销比率的条件外推为 164.13–222.47 GPU小时，尚不含正式全量评估、导出等，也没有可信的全矩阵磁盘上界。ETTh2 跨训练 seed、当前源码 ETTh1/H720 完整余量与 GPU1/H720 未测，故尚不具备新正式 suite 的最终预算审批条件。步骤7–9及 AC7–AC10 保持未完成，Informer 保持 preview。
