# P0 评审点 A：Windows 原生 CUDA 平台验证

日期：2026-09-28。分支：`feat/p0-foundation`。

后续状态：用户已于 2026-09-28 确认评审点 A。第 5–13 步已完成，见 `review-b.md`。LNK4098 的定位与静态 CRT 修复见 `docs/environment.md` 及第 10 步日志。以下保留第 4 步结束时的历史结果与暂停记录。

## 结论与暂停位置

第 1–4 步的验证命令已通过。Windows 原生 CUDA 构建成功，`cuda:0` 的矩阵乘法、反向传播与一次参数更新测试通过。第 4 步技术验证完成；评审点 A 等待用户确认。第 5–13 步尚未开始。

建议继续使用已安装的 Windows 原生工具链。此次未安装系统级工具链，未修改 Candle、cudarc 或 cudaforge 源码，未切换 WSL。后续若需安装工具链，仍须先获得用户确认。

测试输出仍有 `LNK4098`：默认库 `LIBCMT` 与其他库冲突。触发冲突的具体对象尚未定位。该警告需在第 13 步的 CUDA Clippy `-D warnings` 门槛前解决；当前结果不能作为全量质量检查通过的证据。

## 已完成的任务状态操作

- datasets-protocol 已归档到 `.trellis/tasks/archive/2026-09/09-28-datasets-protocol/`；归档提交 `8c2ed9c`。归档前修正了任务中错误的 `branch=main` 元数据。
- datasets-protocol journal 已记录；提交 `9940843`，关联交付提交 `dd725e4`。归档前私有数据检查与 9/9 manifest 校验通过。
- 从 `feat/datasets-protocol` 的 `9940843` 新建 `feat/p0-foundation`，执行 `task.py start 09-28-p0-foundation`。任务状态为 `in_progress`。
- P0 改动尚未提交；未推送。按用户要求内联实施，未派生子代理。

## 实测环境

| 项目 | 实测值 |
| --- | --- |
| OS | Windows 11 Pro 10.0.26200，x64 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU，24463 MiB，compute capability 12.0 |
| NVIDIA 驱动 | 617.14 |
| CUDA Toolkit | 13.3，nvcc V13.3.33 |
| MSVC | Visual Studio 2026 Enterprise 18.10.2，MSVC 14.51.36231 |
| Rust | rustc 1.98.0，commit 88d9e12ae178fab0fb5cc050a94da85685d449ea |
| 目标三元组 | x86_64-pc-windows-msvc |
| Candle | candle-core / candle-nn `=0.11.0` |
| Candle SHA | 31f35b147389700ed2a178ee66a91c3cc25cc80d；已读取已下载 crate 的 `.cargo_vcs_info.json` 核对 |
| cudarc | `=0.19.8`；Cargo.lock 已固定 |
| GPU 编译目标 | sm_120；Candle 构建时自动检测，独立 nvcc 诊断使用显式 sm_120 |

## 逐步验证

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 1 | `cargo build --workspace` | 5 个 crate 骨架构建通过，退出码 0 |
| 2 | `cargo test -p ferrots-core` | 4 项测试通过，退出码 0；覆盖设备解析、非法输入、错误来源、FP32 策略 |
| 3 | `cargo build --workspace`，移除进程中的 CUDA 环境变量与 Toolkit PATH 项 | 退出码 0；固定 cudarc 后耗时 3.21 秒，见 `step-03-cpu-build-pinned.log` |
| 4 | `cargo build --workspace --features cuda`，加载 MSVC 并启用标准预处理器 | 退出码 0，耗时 65.89 秒，见 `step-04-windows-cuda-build-conforming.log` |
| 4 | `cargo test -p ferrots-nn --features cuda -- cuda_smoke`，补齐 CUDA 库路径 | 1 项 GPU 测试通过，退出码 0；测试 10.33 秒、命令 13.10 秒，见 `step-04-windows-cuda-smoke-runtime.log` |

`ferrots-core` 仅依赖 `thiserror`，无 Candle 依赖。默认 feature 树未启用 CUDA。`cuda` feature 经 `cli → train → nn → Candle` 传递；CLI 的 cudarc 为可选依赖，版本与 Candle 所用实例一致。

GPU 测试直接创建 `Device::new_cuda(0)`，无 CPU 回退。测试用合成矩阵核对前向输出与解析梯度，再执行一次直接 SGD 更新并核对全部参数。该测试不代替第 9 步的 Adam/AdamW 验收，也不代替第 10 步的完整 doctor 训练测试。

## 失败记录与处理

| 尝试 | 结果与处理 | 完整日志 |
| --- | --- | --- |
| 初次 CUDA 构建 | 退出码 101；`nvcc fatal : Cannot find compiler 'cl.exe' in PATH`。加载已安装的 `vcvars64.bat`。 | `step-04-windows-cuda-build.log` |
| 加载 MSVC 后重试 | 退出码 101；`candle-kernels` build script 退出 1。 | `step-04-windows-cuda-build-msvc.log` |
| 直接调用固定工具链 Cargo | 退出码 101，仍在相同 build script 失败；不能据此归因于 mbx。 | `step-04-windows-cuda-build-direct.log` |
| 串行构建 GPU 测试 | 退出码 101；仍未进入测试。检查产物发现缺少 `reduce.ptx`。 | `step-04-windows-cuda-smoke.log` |
| 单独编译上游 `reduce.cu` | nvcc 退出码 2；CUDA 13.3 CCCL `preprocessor.h(20)` 的 `fatal error C1189` 要求 `/Zc:preprocessor`。另有代码页 936 的 C4819 警告。 | `step-04-reduce-nvcc.log`，逐行转码的可读副本 `step-04-reduce-nvcc.utf8.log`；原始日志保留 |
| 启用标准预处理器与 UTF-8 | 设置进程级 `CL=/Zc:preprocessor /utf-8` 后，同一个 `reduce.cu` 编译退出码 0。随后工作区 CUDA 构建通过。 | `step-04-reduce-nvcc-conforming.log`、`step-04-windows-cuda-build-conforming.log` |
| 首次链接 GPU 测试 | 退出码 101；`LNK1181` 无法打开 `cuda.lib`。当时进程未设置 `CUDA_PATH`；指定已安装的 v13.3 根目录后，cudarc 生成了相应库搜索路径。 | `step-04-windows-cuda-smoke-conforming.log` |
| 设置 CUDA_PATH 后运行 | 链接成功；运行退出 `0xc0000135 / STATUS_DLL_NOT_FOUND`。核实 DLL 位于 v13.3 的 `bin/x64`，该目录未在 PATH 中。 | `step-04-windows-cuda-smoke-path.log` |
| 补齐 DLL PATH 后运行 | 退出码 0；GPU 前向、反向和更新通过，仍有 LNK4098 警告。 | `step-04-windows-cuda-smoke-runtime.log` |

第 2 步曾因直连 crates.io 的 TLS 握手失败而中止。使用进程级 `HTTPS_PROXY`、`HTTP_PROXY`、`CARGO_HTTP_PROXY` 指向 `http://127.0.0.1:7897` 后通过，未关闭 TLS 校验。

## 已验证的本机命令环境

在新开的 Windows `cmd.exe` 中，按行执行以下命令。路径来自本机核实结果；不修改用户或系统环境变量。

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "CL=/Zc:preprocessor /utf-8"
set "VSLANG=1033"
set "HTTPS_PROXY=http://127.0.0.1:7897"
set "HTTP_PROXY=http://127.0.0.1:7897"
set "CARGO_HTTP_PROXY=http://127.0.0.1:7897"
cd /d D:\Documents\Code\Rust\Exp\ferrots
rustup run 1.98.0 cargo build --workspace --features cuda
rustup run 1.98.0 cargo test -p ferrots-nn --features cuda -- cuda_smoke
```

最终成功尝试直接调用了 `rustup which cargo --toolchain 1.98.0` 返回的 Cargo；上面的 `rustup run` 使用同一固定工具链。`CL` 的设置对新开的终端进程有效；已有自定义 CL 参数的终端需保留其原有参数。

## 待用户确认

- 是否确认评审点 A，继续在 Windows 原生平台实施第 5–13 步？
- LNK4098 列为待处理事项，最终必须通过 CUDA Clippy `-D warnings` 与全部验收项后才提交 P0。
- P0 提交与 P1 下一轮实施 Prompt 在第 13 步和 trellis-check 通过后提供。
