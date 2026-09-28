# 上游事实核对（2026-09-28）

核对对象：`FerroTS_Rust_Implementation_Plan.md` 第 2、4 节引用的 Candle、TSLib、iTransformer 行为。

## 1. Candle

- 版本：crates.io `candle-core` 最新稳定版 `0.11.0`，发布于 2026-06-26。计划书第 3.1 节“未确认已发布”的表述已过期。
- 核对提交：tag `0.11.0` = `31f35b147389700ed2a178ee66a91c3cc25cc80d`。
- 工作区 `cudarc = 0.19.8`，features 含 `cuda-version-from-build-system`、`dynamic-linking`（`Cargo.toml:50-61`）。

| 计划书论断                                | 结论               | 证据（Candle 0.11.0）                                                                                                      |
| ----------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `softmax_last_dim` 无反向                 | 成立               | `candle-nn/src/ops.rs:437-438` 使用 `apply_op1_no_bwd`；`ops.rs:287` 的 `CustomOp1 for SoftmaxLastDim` 未实现 `bwd`        |
| 优化版 `layer_norm` 无反向                | 成立               | `candle-nn/src/ops.rs:944` 使用 `apply_op3_no_bwd`；`ops.rs:692` 的 `CustomOp3 for LayerNorm` 未实现 `bwd`                 |
| `LayerNorm::forward` 对连续输入走快路径   | 成立               | `candle-nn/src/layer_norm.rs:117-121`：`x.is_contiguous() && self.remove_mean` 且有 bias 时调用 `ops::layer_norm`          |
| 存在可微慢路径                            | 成立               | `candle-nn/src/ops.rs:912` `layer_norm_slow`                                                                               |
| `var_keepdim` 除以 n-1                    | 成立               | `candle-core/src/tensor.rs:1068-1072`                                                                                      |
| `cumsum` 构造 n×n 三角矩阵后矩阵乘法      | 成立               | `candle-core/src/tensor.rs:2806-2819`，`Tensor::triu2` + `matmul`/`broadcast_matmul`                                       |
| MaxPool2D 反向在 kernel≠stride 时受限     | 成立，且为直接报错 | `candle-core/src/backprop.rs:363-364` `bail!("backward not supported for maxpool2d if ksize != stride")`                   |
| Candle 无 `max_pool1d`                    | 成立               | `tensor.rs` 仅有 `max_pool2d`（`:1378`）与 `max_pool2d_with_stride`（`:1385`）                                             |
| 自定义算子默认反向返回“不支持”            | 成立               | `candle-core/src/custom_op.rs:38-39` `BackwardNotSupported`                                                                |
| FlashAttn 无反向                          | 成立               | `candle-flash-attn/src/lib.rs:223`、`:824` 两个 `CustomOp3` 实现均无 `fn bwd`                                              |
| AdamW 矩状态随参数 dtype 创建             | 成立               | `candle-nn/src/optim.rs:128-129` `Var::zeros(shape, dtype, device)`                                                        |
| CUDA 内核构建按本机算力编译，含 MSVC 分支 | 成立               | `candle-kernels/build.rs:48-60`：`cudaforge::detect_compute_cap()`；`target.contains("msvc")` 时追加 `-D_USE_MATH_DEFINES` |

## 2. iTransformer（thuml/iTransformer main）

- `model/iTransformer.py` `forecast()`：`means = x_enc.mean(1, keepdim=True).detach()`；`torch.var(x_enc, dim=1, keepdim=True, unbiased=False)`；`enc_embedding(x_enc, x_mark_enc)` 把时间特征作为额外变量 token；输出 `projector(enc_out).permute(0, 2, 1)[:, :, :N]`。
- 结论：官方 ETT 复现路径默认使用时间特征 token（`x_mark_enc`）。计划书第 8 节把时间 token 写成可选项，第 15 节 P1 未列出时间特征生成。P1 必须提供 `timeF` 时间特征，否则无法对齐官方 ETT 结果。
- `data_provider/data_factory.py`：测试集 `batch_size = 1`、`shuffle = False`、`drop_last = True`。测试集按 1 个样本一批，因此 `drop_last` 不丢样本；训练/验证集 `drop_last = True`。

## 3. TSLib（thuml/Time-Series-Library main）

- `Dataset_ETT_hour`：`border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]`，`border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]`；scaler 只在训练段拟合。`Dataset_ETT_minute` 同样结构，长度乘 4。
- `Dataset_Custom`：`num_train = int(len * 0.7)`，`num_test = int(len * 0.2)`，其余为验证。本地文件缺失时调用 `load_dataset(HUGGINGFACE_REPO, ...)` 远程回退。
- 结论：计划书第 2.1 节只描述了 `Dataset_Custom` 的 70/10/20 切分。ETT 复现使用固定的 12/4/4 个月边界，DataSpec 必须支持显式行边界切分。

## 4. 本机环境（2026-09-28 实测）

| 项目         | 值                                                                               |
| ------------ | -------------------------------------------------------------------------------- |
| OS           | Windows 11 Pro 10.0.26200                                                        |
| GPU          | NVIDIA GeForce RTX 5090 Laptop GPU，24463 MiB，compute capability 12.0（sm_120） |
| 驱动         | 617.14                                                                           |
| CUDA Toolkit | 13.3（`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`）     |
| Rust         | rustc 1.98.0，`stable-x86_64-pc-windows-msvc`                                    |
| WSL2         | Ubuntu-24.04（已安装，状态 Stopped）                                             |
| Python       | 3.14.7；`uv` 可用；全局未安装 polars                                             |

计划书第 13 节建议首个 GPU 验收环境为 Linux x86_64。本机为 Windows 原生 + sm_120 + CUDA 13.3，该组合在 Candle 上的构建与训练结果尚未实测。P0 必须把该组合作为第一个构建门槛，WSL2 Ubuntu-24.04 作为回退路径。
