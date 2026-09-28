# P0 工具链固定与训练安全算子

父任务：`09-28-ferrots-implementation`。计划书对应：第 3、4、5、13 节，第 15 节 P0。

## Goal

建立 Rust 工作区，固定 Candle 版本，在本机 GPU 上确认 CUDA 构建与训练路径可用，并提供训练安全算子层与可恢复的优化器。完成后，后续子任务可以在 CPU 与 `cuda:0` 上做前向、反向与参数更新，且不会误用无反向的融合算子。

## 前置

- 无。可与 `09-28-datasets-protocol` 并行。

## 已知事实

- Candle `0.11.0` = `31f35b147389700ed2a178ee66a91c3cc25cc80d`，`cudarc 0.19.8`（父任务 `research/upstream-verification.md` 第 1 节）。
- 本机：Windows 11，RTX 5090 Laptop（sm_120，24 GB），驱动 617.14，CUDA 13.3，rustc 1.98.0 MSVC；WSL2 Ubuntu-24.04 可用。
- Candle 无反向的路径：`candle_nn::ops::softmax_last_dim`、`candle_nn::ops::layer_norm`、`candle_nn::LayerNorm::forward`（连续输入且有 bias）；`var_keepdim` 为无偏方差。

## Requirements

- R1 工作区：`crates/ferrots-{core,data,nn,train,cli}` 五个 crate；`rust-toolchain.toml` 固定工具链；提交 `Cargo.lock`。`ferrots-core` 不依赖 Candle。
- R2 依赖固定：所有 Candle 子 crate 使用 `=0.11.0`；提交 SHA 记录在 `docs/environment.md`。
- R3 构建特性：默认构建为 CPU-only，不需要 CUDA Toolkit；`cuda` feature 启用 GPU。
- R4 本机 CUDA 构建门槛：先在 Windows 原生（MSVC + CUDA 13.3 + sm_120）构建 `--features cuda` 并在 `cuda:0` 上完成一次前向→反向→更新。该路径失败时，记录失败命令与完整错误输出，切换到 WSL2 Ubuntu-24.04 重试。最终选择的平台与原因写入 `docs/environment.md`。
- R5 `ferrots doctor --device <dev> [--training-smoke-test]`：输出 Rust 版本与目标三元组、驱动与 CUDA 版本、GPU 名称/算力/显存、Candle 构建 features；`--training-smoke-test` 执行真实的前向→反向→参数更新。显式指定的 `cuda:N` 不可用时以非零码退出，不改用 CPU。
- R6 训练安全算子（`ferrots-nn` 的 `train_ops`）：
  - 可微 softmax（沿任意维），含全掩码行检测。
  - LayerNorm：与 `layer_norm_slow` 等价的基础运算实现，显式缩放/偏移参数。
  - 总体方差（除以 L）与 epsilon 参数。
  - Dropout：显式 train/eval 标志，随机流可由种子与步号重建。
  - 缩放点积注意力：QK 矩阵乘法 + 稳定 softmax + AV。
- R7 推理算子（`infer_ops`）：只有在调用方声明不需要梯度时才使用 Candle 融合 softmax/LayerNorm。
- R8 无反向路径检测：
  - 负面测试：用 Candle 融合 softmax/LayerNorm 搭建小网络，断言上游参数缺少梯度且测试能识别。
  - 源码检查：`ferrots-nn` 训练路径中禁止调用 `softmax_last_dim`、`ops::layer_norm`、`candle_nn::LayerNorm`；以脚本或测试实现，并纳入本地检查命令。
- R9 `StatefulAdam` / `StatefulAdamW`：稳定参数名、逐参数 step、FP32 一阶/二阶矩（与参数 dtype 无关）、参数组；状态以 `optimizer.safetensors` + JSON 元信息保存与加载。
- R10 精度：只支持 FP32；其他 dtype 在构建模型或优化器时返回错误。
- R11 参考夹具工具：`tools/reference/` 使用 `uv` 管理的独立 Python 环境（固定 Python 与 torch 版本），生成 Adam/AdamW 单步更新夹具。生产构建不依赖 Python。

## Acceptance Criteria

- [x] 未设置 CUDA 环境变量时，`cargo build --workspace` 成功。
- [x] 在 R4 选定的平台上 `cargo build --workspace --features cuda` 成功；`docs/environment.md` 记录 OS、驱动、CUDA、rustc、Candle 版本与 SHA、选择原因。
- [x] `ferrots doctor --device cuda:0 --training-smoke-test` 退出码为 0，并输出 R5 列出的全部字段；`ferrots doctor --device cuda:9` 退出码非 0。
- [x] CPU（f64）有限差分梯度测试通过：softmax、LayerNorm（输入、缩放、偏移）、总体方差（含常量列、L=1）、注意力（Q、K、V）。
- [x] 同一输入下 CPU 与 CUDA（f32）的前向与梯度差异在 atol=1e-5、rtol=1e-4 以内。
- [x] R8 负面测试通过；源码检查在故意引入 `softmax_last_dim` 调用时失败。
- [x] Adam 与 AdamW 单步更新结果与 torch 夹具差异在 atol=1e-6 以内（FP32，CPU）。
- [x] 恢复测试：连续 10 步与“5 步 → 保存 → 加载 → 5 步”的参数在 CPU 上逐位相同，在 CUDA 上差异在 atol=1e-6 以内。
- [x] `cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace` 全部通过。

## Out of Scope

- 自定义 CUDA kernel、FlashAttention、BF16/FP16（P5）。
- 数据管线与模型（P1 起）。
- GitHub Actions 等远程 CI（已配置 origin；本任务不设置远程 CI）。

## Notes

- 本子任务进入规划时补写 `design.md` 与 `implement.md`。
- 本子任务完成后，建议执行 `00-bootstrap-guidelines`，以真实 Rust 代码为依据填写 `.trellis/spec/`。
