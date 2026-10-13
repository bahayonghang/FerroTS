# Informer 实施计划

状态 in_progress。用户已于 2026-10-13 授权按顺序实施；开发门已通过，正式矩阵预算待审批。证据见 research/implementation-evidence.md。

1. [ ] 冻结TSLib来源/许可、官方profile、fixture manifest、AC、CUDA预算和磁盘估算。
2. [x] 实现PrefixSum、采样score、top-query、ReplaceRows及蒸馏算子；CUDA前后向夹具、边界、scan峰值和shape门通过（AC1–AC3）。
3. [x] 接入配置/静态派发、embedding、encoder/decoder、历史尾部+未来零值；全参数真实损失梯度与标签隔离检查（AC1、AC6）。
4. [x] 命名BatchNorm buffer及采样身份贯穿snapshot、manifest、save/load、resume、freeze、预测，检查损坏buffer和验证隔离（AC4、AC5）。
5. [x] CUDA独立进程连续/恢复训练比对，开启dropout/BN/采样，覆盖B=1、尾批及CLI预测（AC4–AC6）。
6. [ ] 跑L96/192/384/768/1536复杂度门、CUDA短训练及阶段性能；通过后冻结suite并审批正式矩阵预算（AC2、AC3、AC9）。
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

步骤1的来源、许可、profile、fixture及验收条件已落实；预算与磁盘估算已实测整理，正式预算审批尚未完成，因此步骤1、6保持未勾选。步骤6的复杂度和CUDA短训练已通过。步骤7–9尚未执行：正式0/105，Informer保持preview，不启动WaveForM。
