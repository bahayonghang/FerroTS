# P0 执行计划

每步完成后运行该步的验证命令；失败时停在该步修复，不跳步。

## 检查清单

1. [ ] 工作区骨架：根 `Cargo.toml`、`rust-toolchain.toml`、5 个 crate 的 `Cargo.toml` 与 `lib.rs`/`main.rs`；`.gitignore` 增加 `target/`。
   - 验证：`cargo build --workspace`
2. [ ] `ferrots-core`：`Device` 解析、`DTypePolicy`、错误类型；单元测试覆盖 `cpu`、`cuda:0`、非法字符串。
   - 验证：`cargo test -p ferrots-core`
3. [ ] 固定 Candle `=0.11.0` 依赖与 `cuda` feature 链；CPU 构建不需要 CUDA 环境。
   - 验证：在未设置 `CUDA_PATH` 的 shell 中 `cargo build --workspace`
4. [ ] **平台门槛**：`cargo build --workspace --features cuda`，并运行一个在 `cuda:0` 上做 matmul + backward 的最小测试。按 design.md 第 9 节顺序尝试；需要安装工具链时先询问用户。
   - 验证：`cargo test -p ferrots-nn --features cuda -- cuda_smoke`
   - **评审点 A**：把平台结果与错误记录报告给用户，确认后再继续。
5. [ ] `train_ops`：softmax、masked_softmax、pop_var、layer_norm、attention；CPU f64 有限差分测试（含常量列、L=1、全掩码行报错）。
   - 验证：`cargo test -p ferrots-nn train_ops`
6. [ ] `RngStream` 与 dropout；测试同一 `(seed, step, layer)` 得到相同 mask，train/eval 行为正确。
   - 验证：`cargo test -p ferrots-nn dropout`
7. [ ] `infer_ops` + `InferenceOnly`；负面测试 `fused_path_cuts_grad`；源码守卫 `no_fused_in_train`（临时插入一行 `softmax_last_dim` 确认守卫失败后撤回）。
   - 验证：`cargo test -p ferrots-nn --test fused_path_cuts_grad --test no_fused_in_train`
8. [ ] `tools/reference/`：`uv` 项目 + `gen_optim_fixture.py`；生成 `tests/fixtures/optim/{adam,adamw}.safetensors`。
   - 验证：`uv run --project tools/reference python tools/reference/gen_optim_fixture.py`
9. [ ] `StatefulAdam` / `StatefulAdamW` + 保存/加载；夹具对照测试；恢复测试（10 步连续 vs 5+5 步）。
   - 验证：`cargo test -p ferrots-train`
10. [ ] `ferrots doctor`（`build.rs` 编译期信息、cudarc 运行期信息、`--training-smoke-test`）。
    - 验证：`cargo run -p ferrots-cli --features cuda -- doctor --device cuda:0 --training-smoke-test`；`... doctor --device cuda:9` 退出码非 0
11. [ ] CPU/CUDA 一致性测试（f32，atol=1e-5、rtol=1e-4）：train_ops 前向与梯度、优化器 10 步。
    - 验证：`cargo test --workspace --features cuda`
12. [ ] `docs/environment.md`：实测 OS、驱动、CUDA、rustc、Candle 版本与 SHA、平台选择与原因、失败尝试记录。
13. [ ] 全量检查：
    - `cargo fmt --all --check`
    - `cargo clippy --workspace --all-targets -- -D warnings`
    - `cargo clippy --workspace --all-targets --features cuda -- -D warnings`
    - `cargo test --workspace`
    - `cargo test --workspace --features cuda`
    - **评审点 B**：运行 `trellis-check`，对照 prd.md 验收项逐条确认。

## 风险文件与回滚点

- 根 `Cargo.toml` 的依赖固定：改动后需重新跑第 3、4 步。
- `candle-kernels` 在 sm_120 + CUDA 13.3 上的编译：第 4 步为主要风险点；失败时按 design.md 第 9 节切换路径，不修改 Candle 源码。
- 回滚点：第 4 步之前的提交（纯 CPU 可构建状态）。

## 启动前检查

- [ ] `implement.jsonl` / `check.jsonl` 已列出计划书与父任务研究记录。
- [ ] 用户已确认最终规划摘要。
