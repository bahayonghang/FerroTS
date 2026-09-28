# P0 environment and local validation

Measured on 2026-09-28 on branch `feat/p0-foundation`. Production builds do not require Python.

## Selected platform

Use native Windows. The existing toolchain passed CUDA matrix multiplication, backward propagation, an AdamW update, and CPU/CUDA consistency tests. WSL2 was not required. CUDA 12.9 and WSL CUDA Toolkit were not installed.

| Component | Observed value |
| --- | --- |
| OS | Windows 11 Pro 10.0.26200, x64 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, compute capability 12.0 (sm_120) |
| GPU memory | 25,622,413,312 bytes from the CUDA context; nvidia-smi reports 24,463 MiB |
| Display driver | 617.14 |
| CUDA Toolkit / nvcc | 13.3 / 13.3.33 |
| CUDA driver API / runtime API | Both report 13.4 (13040) through cudarc at runtime |
| Visual Studio | 2026 Enterprise 18.10.2 |
| MSVC | 14.51.36231 |
| rustc | 1.98.0 (88d9e12ae, 2026-08-18) |
| Target | x86_64-pc-windows-msvc |
| Candle core / nn | =0.11.0; SHA 31f35b147389700ed2a178ee66a91c3cc25cc80d |
| cudarc | =0.19.8; one resolved instance shared with Candle |
| Reference Python / torch / numpy | 3.12.12 / 2.14.0+cu130 / 2.1.2 |
| Fixture safetensors | Python 0.7.0; Rust 0.8.0 (Candle-compatible) |

The Candle SHA was checked against the downloaded crate's .cargo_vcs_info.json and the parent task's upstream evidence. Runtime API version 13.4 and nvcc version 13.3 are separate observations. The cause of their different reported versions is not established. Do not infer the runtime API version from nvcc output.

## CPU commands

The default workspace feature set is CPU-only. CUDA_PATH and CUDA Toolkit PATH entries were removed for the CPU build. Use the repository's rust-toolchain.toml and Cargo.lock.

```powershell
$env:RUSTUP_AUTO_INSTALL = '0'
$env:RUSTUP_TOOLCHAIN = '1.98.0'
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Native Windows CUDA commands

Run in cmd.exe at the repository root. These environment variables affect only the current process and its children. Adapt the Visual Studio path to the installed location. Preserve any unrelated local flags when adapting these commands.

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "CL=/Zc:preprocessor /utf-8"
set "NVCC_APPEND_FLAGS=-Xcompiler=/MT"
set "RUSTFLAGS=-C target-feature=+crt-static"
set "CARGO_BUILD_TARGET=x86_64-pc-windows-msvc"
set "CARGO_TARGET_DIR=%CD%\target\windows-static"
set "RUSTUP_AUTO_INSTALL=0"
set "RUSTUP_TOOLCHAIN=1.98.0"
set "VSLANG=1033"
set "HTTPS_PROXY=http://127.0.0.1:7897"
set "HTTP_PROXY=http://127.0.0.1:7897"
set "CARGO_HTTP_PROXY=http://127.0.0.1:7897"
rustup run 1.98.0 cargo build --workspace --features cuda
rustup run 1.98.0 cargo run -p ferrots-cli --features cuda -- doctor --device cuda:0 --training-smoke-test
rustup run 1.98.0 cargo run -p ferrots-cli --features cuda -- doctor --device cuda:9
rustup run 1.98.0 cargo test --workspace --features cuda
rustup run 1.98.0 cargo clippy --workspace --all-targets --features cuda -- -D warnings
```

The cuda:9 command must fail. No CUDA request falls back to CPU. The CUDA context used for diagnostics belongs to Candle. The CLI enables cudarc's runtime feature for version queries and uses nvidia-smi only for the display driver release. Missing nvidia-smi is reported explicitly.

For validation logs, Cargo was invoked from the path returned by `rustup which cargo --toolchain 1.98.0`. RUSTC and RUSTDOC were also set to the corresponding absolute toolchain binaries. This bypasses the local mbx Cargo wrapper and prevents dependency directories from selecting another rust-toolchain.toml. Direct Cargo without these settings is not sufficient.

Use a new target directory after changing native CRT flags. cudaforge's object cache does not include NVCC_APPEND_FLAGS. Do not reuse objects built with a different CRT. No existing target directory was deleted during validation.

## Failures and corrections

Full logs are under .trellis/tasks/09-28-p0-foundation/research/ while P0 remains active. After archive, use the same task under tasks/archive/2026-09/.

| Failure | Evidence and correction |
| --- | --- |
| nvcc could not find cl.exe | Initialize vcvars64.bat. |
| CCCL fatal C1189 in reduce.cu | Set CL=/Zc:preprocessor; /utf-8 also prevents code page C4819. |
| Link could not find cuda.lib | Set CUDA_PATH explicitly. |
| STATUS_DLL_NOT_FOUND | Add CUDA bin/x64 to PATH. |
| LNK4098 LIBCMT conflict | dumpbin showed /MT objects in libmoe.a. An /MD rebuild alone still warned: the installed CUDA cuda.lib also requests LIBCMT. Align Rust and native objects to static CRT, with an explicit Cargo target. The final static build has no LNK4098. No /NODEFAULTLIB or warning suppression was used. |
| Dependency selected Rust 1.78.0 | Direct Cargo launched the rustup rustc proxy inside macro_rules_attribute-0.2.3, whose rust-toolchain.toml requests 1.78.0. rustup automatically installed that toolchain without prior approval. The event was reported to the user. It was retained because deletion requires approval. Subsequent commands disable automatic installation and fix RUSTUP_TOOLCHAIN, RUSTC and RUSTDOC to 1.98.0. |
| Localized linker output | This VS installation has the 2052 language directory. Rust reports Chinese import-library creation text as linker_messages even with VSLANG=1033. These messages contain no LNK warning/error after the CRT correction. They remain visible; no allow/ignore flags were added. |
| Initial CLI compile error E0597 | Added the explicit &str type to the smoke-test parameter lookup closure; CLI tests passed after the correction. |

Platform-gate-a.md retains the original step-4 history. Step-10 logs record the later CRT investigation, toolchain selection failure, and successful doctor runs.

## Numerical validation

- CPU f64 finite differences: softmax on both axes, masked softmax, LayerNorm input/gamma/beta, population variance including constant input and L=1, attention Q/K/V.
- CPU/CUDA f32 forward and gradients: atol=1e-5, rtol=1e-4, including masked attention and dropout. Dropout masks match exactly.
- Adam and AdamW: two parameter groups, ten torch-reference steps, missing gradients on steps 3 and 7; parameters and FP32 moments use atol=1e-6.
- Ten continuous steps versus five steps, save/load, five steps: CPU parameters and moments match bitwise; CUDA uses atol=1e-6. Per-parameter counters and parameter groups are restored.
- GPU doctor: six parameter tensors have finite nonzero gradients and all six change after AdamW. Observed loss 0.04878764. The recorded synchronized forward/backward/update times are 176.462/4.456/1.295 ms. These are a smoke test, not a benchmark.
- Step 11: 30 tests passed with CUDA enabled. No test silently skips an unavailable GPU in a CUDA build.
- Final gate: CPU 24 tests and CUDA 30 tests passed, with no failures or ignored tests. Formatting, both builds, and CPU/CUDA Clippy with -D warnings passed. The final command logs and acceptance mapping are in the P0 task research/review-b.md.

## Reference fixtures

Reuse tools/reference/.venv through uv; do not create a second reference environment.

```powershell
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:PYTHONIOENCODING = 'utf-8'
uv run --project tools/reference python tools/reference/gen_optim_fixture.py
bash tools/check_private_data.sh
```

The fixture generator uses synthetic CPU FP32 tensors, seed 137, and torch foreach=False/fused=False. Its metadata includes parameter groups and actual package versions. The two checked-in safetensors files total 20,608 bytes. No private data or runs are included.
