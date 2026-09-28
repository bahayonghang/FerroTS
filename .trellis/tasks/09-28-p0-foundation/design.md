# P0 技术设计

## 1. 工作区与边界

```text
Cargo.toml               # [workspace]，resolver = "3"，统一 [workspace.dependencies]
rust-toolchain.toml      # channel = "1.98.0"，components = ["rustfmt", "clippy"]
crates/
  ferrots-core/          # Device、DTypePolicy、错误类型；不依赖 Candle
  ferrots-data/          # P0 只建空 crate（lib.rs），P1 实现
  ferrots-nn/            # train_ops、infer_ops、源码守卫测试
  ferrots-train/         # StatefulAdam、StatefulAdamW、参数集合保存/加载；重导出 RngStream
  ferrots-cli/           # 二进制 `ferrots`；P0 只有 `doctor` 子命令
tools/reference/         # uv 管理的 Python 夹具工具，仅开发期使用
tests/fixtures/optim/    # 合成数据生成的 Adam/AdamW 夹具（可提交，体积 < 1 MB）
docs/environment.md      # 实测环境矩阵与平台选择记录
```

依赖方向：`cli → train → nn → core`；`data → core`。`ferrots-core` 不引入 Candle 类型，Device 在 `nn`/`train` 中转换为 `candle_core::Device`。

## 2. 依赖固定与 features

- `candle-core = "=0.11.0"`、`candle-nn = "=0.11.0"`（tag `0.11.0` = `31f35b147389700ed2a178ee66a91c3cc25cc80d`），写入 `[workspace.dependencies]`，`default-features = false`。
- 其他：`serde`、`serde_json`、`safetensors`（与 Candle 0.11.0 所用版本一致）、`clap`（derive）、`thiserror`、`rand_chacha`。
- feature 链：`ferrots-cli/cuda → ferrots-train/cuda → ferrots-nn/cuda → candle-core/cuda + candle-nn/cuda`。默认不启用。
- `doctor` 查询驱动与设备属性时，使用与 Candle 相同版本与 features 的 `cudarc = "=0.19.8"`，只在 `cuda` feature 下编译；复用 Candle 已创建的 CUDA context，不另建。

## 3. Device 与精度策略

- `Device` 解析：`cpu`、`cuda:N`。显式 `cuda:N` 创建失败时返回 `DeviceError::CudaUnavailable { index, source }`，调用方不改用 CPU。
- `DTypePolicy`：P0 只有 `F32` 变体（测试可用 `F64` 于 CPU 有限差分）。模型或优化器收到其他 dtype 时返回 `DTypeError::Unsupported`。

## 4. 训练安全算子（`ferrots-nn::train_ops`）

全部由 Candle 基础可微运算组成，不调用任何 `*_no_bwd` 路径。

| 函数                                | 定义                                                          | 说明                                                        |
| ----------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `softmax(x, dim)`                   | `e = exp(x - max(x, dim).detach()); e / sum(e, dim)`          | 减最大值只为数值稳定，梯度不变                              |
| `masked_softmax(x, mask, dim)`      | 无效位置置为 `-inf` 后同上                                    | 任一行全部无效时返回 `OpError::FullyMaskedRow`              |
| `pop_var(x, dim, keepdim)`          | `mean((x - mean(x))², dim)`                                   | 除以 L；L=1 返回 0                                          |
| `layer_norm(x, gamma, beta, eps)`   | `(x - mean) / sqrt(pop_var + eps) * gamma + beta`（最后一维） | 与 Candle `layer_norm_slow` 数值等价；gamma/beta 为显式参数 |
| `dropout(x, p, mode, stream)`       | train：`x * mask / (1-p)`；eval：`x`                          | mask 见第 5 节                                              |
| `attention(q, k, v, mask, dropout)` | `softmax(q kᵀ / sqrt(d)) v`                                   | 形状 `[B, H, Lq, D]` × `[B, H, Lk, D]`                      |

`infer_ops` 提供 `softmax_fused`、`layer_norm_fused`（调用 Candle 融合算子）。两个函数要求 `&InferenceOnly`，调用方通过 `InferenceOnly::evaluation()` 显式声明推理。Rust 无法根据调用方模块限制公开构造函数，因此入口还检查全部输入与 affine 参数的 `track_op()`，拒绝变量及保留计算图的派生张量。调用方必须先冻结输入；token 与运行期校验共同防止误用。

## 5. 随机流

- 可重建随机流：`RngStream { seed, run_step, layer_id }` → `ChaCha8Rng::from_seed(key)`。32 字节 key 为三个 u64 的小端编码加 `b"FerroTS0"`，避免未指定 hash 算法导致跨版本变化。RngStream 定义于 `ferrots-nn` 并由 `ferrots-train` 重导出，避免 train → nn → train 循环依赖。
- Dropout mask 在主机上用该流生成，再传到目标设备。CPU 与 CUDA 因此得到相同 mask，一致性测试可直接比较；代价是每步一次主机→设备传输。P4 剖析后再决定是否改为设备端生成。
- 随机流状态只由 `(seed, run_step)` 决定，恢复训练时不需要保存生成器内部状态。

## 6. 优化器（`ferrots-train::optim`）

```text
StatefulAdamW {
  params: Vec<NamedParam { name: String, var: Var }>,   // 名称稳定、顺序固定
  groups: Vec<ParamGroup { names, lr, beta1, beta2, eps, weight_decay }>,
  state:  BTreeMap<String, ParamState { step: u64, exp_avg: Tensor(F32), exp_avg_sq: Tensor(F32) }>,
}
```

- 更新公式与 PyTorch 一致：
  - Adam：`g = g + wd·p`（L2），`m = β1·m + (1-β1)·g`，`v = β2·v + (1-β2)·g²`，`p = p - lr · (m / (1-β1^t)) / (sqrt(v / (1-β2^t)) + eps)`。
  - AdamW：先 `p = p · (1 - lr·wd)`，其余同上且不把 wd 加到 g。
- 矩状态始终为 F32，与参数 dtype 无关。
- 保存：`optimizer.safetensors`（键 `<name>.exp_avg`、`<name>.exp_avg_sq`）+ `optimizer-state.json`（每参数 step、各组超参数、参数名顺序、dtype、格式版本）。
- 加载校验：参数名集合、shape、dtype、格式版本一致；任一不符返回错误，不部分加载。
- 模型参数保存/加载：`NamedParams::save/load`（safetensors），同样按名称与 shape 校验。

## 7. 无反向路径检测

- 负面测试（`ferrots-nn/tests/fused_path_cuts_grad.rs`）：用 `candle_nn::ops::softmax_last_dim` 与 `candle_nn::LayerNorm`（连续输入 + bias）搭建两层网络，`backward` 后断言第一层权重在 `GradStore` 中缺失；再用 `train_ops` 搭建同一网络，断言全部参数有梯度。
- 源码守卫（`ferrots-nn/tests/no_fused_in_train.rs`）：扫描 `crates/ferrots-nn/src/` 下除 `infer_ops.rs` 之外的文件，出现 `softmax_last_dim`、`ops::layer_norm`、`candle_nn::LayerNorm`、`rms_norm` 时测试失败。

## 8. `doctor`

- 编译期：`build.rs` 写入 rustc 版本、目标三元组、启用的 features。
- 运行期（`cuda` feature）：复用 Candle 的 context，查询驱动 CUDA API 版本、CUDA 运行时版本、设备名、算力、总显存。CLI 对同一个 cudarc 依赖额外启用 `runtime` feature 以调用版本接口。可用时用 `nvidia-smi` 补充显示驱动发布版本；缺失时标记 unavailable，CUDA API 版本仍为必填诊断项。
- `--training-smoke-test`：在目标设备上构建 `Linear → layer_norm → softmax → Linear` 小网络，前向 → 掩码 MSE → backward → 断言全部参数有非零梯度 → AdamW 一步 → 断言参数变化；输出每步耗时。
- 退出码：全部通过为 0；任一步失败为非 0，并打印失败步骤与错误。

## 9. 本机 CUDA 构建门槛

按顺序尝试，每一步记录命令、完整错误输出与耗时：

1. Windows 原生：MSVC + CUDA 13.3 + sm_120（`CUDA_COMPUTE_CAP` 由 cudaforge 自动检测）。
2. 失败原因与 CUDA 13 相关时：Windows 原生 + 并行安装的 CUDA 12.9 Toolkit（支持 sm_120）。
3. 失败原因与 Windows/MSVC 相关时：WSL2 Ubuntu-24.04 + WSL 版 CUDA Toolkit（不在 WSL 内安装显示驱动）。

第 2、3 步需要安装系统级工具链，执行前征得用户同意。最终选择写入 `docs/environment.md`。

## 10. 参考夹具

- `tools/reference/pyproject.toml`：固定 Python 3.12 与 torch 版本（CPU wheel 足够生成优化器夹具）。
- `gen_optim_fixture.py`：用固定种子生成合成参数与梯度序列，执行 torch Adam/AdamW 若干步，保存初始参数、每步梯度、每步参数与状态到 safetensors。
- 夹具只使用合成数据（父任务 X1）。

## 11. 取舍

- 选用基础运算组合实现 LayerNorm/softmax：训练正确性优先，速度低于融合算子；融合训练路径在 P5 单独验收。
- Dropout mask 主机生成：换取跨设备可复现，代价是传输开销。
- 自建优化器：Candle `Optimizer` trait 无状态导出接口，且矩状态随参数 dtype，不满足恢复与 FP32 状态要求。

## 12. 回滚

P0 为新建代码，无迁移。回滚方式为回退本任务提交；平台门槛失败时保留 CPU 构建，GPU 验收延后并在父任务标注。
