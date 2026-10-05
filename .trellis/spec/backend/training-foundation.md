# P0 training and checkpoint contracts

## 1. Scope / trigger

Apply this spec when changing core device/precision types, nn operations, optimizer state, or doctor. The dependency direction is cli → train → nn → core; data → core. Core must not import Candle. P1 data/model/Trainer contracts are in data-pipeline.md and training-loop.md. Data remains independent of Candle.

## 2. Signatures

- core: `Device::from_str("cpu" | "cuda:N")`, `DTypePolicy::F32`.
- nn: `resolve_device(DeviceSpec) -> Result<candle_core::Device, DeviceError>`.
- train_ops: `softmax(x, dim)`, `masked_softmax(x, mask, dim)`, `pop_var(x, dim, keepdim)`, `layer_norm(x, gamma, beta, eps)`, `attention(q, k, v, mask, dropout)`.
- nn: `dropout(x, p, Mode, RngStream)`; train reexports RngStream.
- infer_ops: `softmax_fused(x, &InferenceOnly)`, `layer_norm_fused(x, gamma, beta, eps, &InferenceOnly)`.
- train: `NamedParams::new(Vec<(String, Var)>)`, `save(path)`, `load(path)`; `StatefulAdam/StatefulAdamW::new(params, groups)`, `step(&GradStore)`, `save(directory)`, `load(directory)`.
- CLI: `ferrots doctor --device <cpu|cuda:N> [--training-smoke-test]`.

## 3. Contracts

### Devices and numeric operations

Default features are CPU-only. CUDA follows cli → train → nn → Candle and never falls back to CPU. Pin candle-core/candle-nn to =0.11.0 and cudarc to =0.19.8. Cargo.lock is tracked. Runtime CUDA queries reuse Candle's context.

Production parameters and optimizer moments are F32. Low-level nn operations permit CPU F64 for numerical tests. A U8 mask marks nonzero entries as valid and broadcasts to the input. A fully masked row is an error. LayerNorm reduces the final dimension and uses population variance, finite positive epsilon, and explicit gamma/beta. Attention shapes are Q=[B,H,Lq,D], K=[B,H,Lk,D], V=[B,H,Lk,Dv].

Dropout train mode uses a host-generated mask and inverted scaling. Eval returns the original graph-preserving tensor. RngStream contains seed, run_step, layer_id. Its ChaCha8 32-byte key is three little-endian u64 values followed by b"FerroTS0". All three inputs must be restored or derived consistently. Frozen evaluation is separate from the dropout mode.

InferenceOnly::evaluation() is an explicit caller declaration. Both fused wrappers also reject all tensors with track_op()==true, including gamma/beta and graph-derived values. A public constructor alone cannot enforce a caller-module boundary in Rust.

### Parameters and optimizer files

NamedParams sorts names in a BTreeMap. Names and variable identities are unique, names are nonempty, and all parameters use one device. Cloning NamedParams shares variables. A detached tensor can still share mutable storage: use an explicit tensor copy or host values for immutable test snapshots.

Each ParamGroup names its parameters and owns lr, beta1, beta2, eps and weight_decay. Groups cover every parameter exactly once. Missing gradients skip weight decay, moments, and that parameter's step counter. Adam uses coupled L2 decay; AdamW applies decoupled parameter decay. Moments are detached F32 tensors.

`optimizer.safetensors` has exactly two keys per parameter: <name>.exp_avg and <name>.exp_avg_sq. `optimizer-state.json` has format_version=1, kind=adam/adamw, dtype=f32, ordered parameters[{name,shape,step}], and groups. Unknown JSON fields are errors. Model safetensors keys are the exact parameter names.

Load validates all names, shapes, dtypes, moment keys, version and groups before replacing live state. Optimizer loading restores hyperparameters. A separate model load is required to restore weights. Save writes into the caller's staging directory. The checkpoint coordinator owns atomic publication of the combined model/state bundle.

### CLI and environment

Doctor prints rustc, target, build features, Candle version/SHA, requested device, CUDA API/runtime versions, GPU name/capability/bytes, and display driver when nvidia-smi is available. CPU fields use n/a. The smoke test checks finite nonzero gradients for six tensors and updates all six through AdamW; timings synchronize the target device.

Use the measured Windows environment in docs/environment.md: initialized MSVC, CUDA_PATH, bin/x64 on PATH, CL=/Zc:preprocessor /utf-8, and consistent static CRT. Do not suppress LNK4098. Select a new ignored target directory when changing native CRT flags. Set RUSTUP_AUTO_INSTALL=0 and pin the toolchain before commands. Installing a system toolchain requires user approval.

For CUDA development and test builds that retain deterministic host-side dropout or gradient validation, package-scoped Cargo opt-level = 3 is allowed when measurements identify host code as the bottleneck. Keep debug assertions enabled. Preserve RNG calls, tensor arithmetic, synchronization boundaries, release settings and formal training profiles. Record the source and binary identity change, and do not reuse prior runtime receipts as current-source acceptance without a new identity audit.

## 4. Validation and error matrix

| Input or state | Required result |
| --- | --- |
| Invalid device spelling or negative index | DeviceError::InvalidSpec; CLI nonzero |
| Explicit unavailable CUDA or CPU-only binary | DeviceError::CudaUnavailable; no fallback |
| Non-F32 optimizer parameter | DTypeError::Unsupported |
| Duplicate name/variable, mixed devices, incomplete/duplicate group | TrainError::InvalidState |
| Non-finite/invalid hyperparameter, nonpositive epsilon | Error before optimizer creation/load |
| Reduction axis missing/empty, incompatible attention shape | OpError or Candle shape error |
| All-masked softmax row | OpError::FullyMaskedRow |
| Fused operation receives a tracked tensor | OpError::InvalidArgument |
| Missing gradient | No mutation for that parameter |
| Invalid optimizer/model checkpoint | Error before modifying live state |
| Failed doctor stage | Nonzero status with stage and source error |

## 5. Good / base / bad cases

- Good: save weights and optimizer after step 5, create a fresh model, load both, then run the same five remaining gradient steps.
- Base: doctor on CPU works without CUDA Toolkit or Python.
- Bad: a CUDA request fails and a caller silently retries CPU; such a fallback violates the device contract.
- Bad: using a cloned NamedParams or a detached view as an immutable pre-update snapshot.

## 6. Required tests

Run fmt, CPU/CUDA Clippy with -D warnings, and CPU/CUDA workspace tests. CUDA tests must fail when CUDA is unavailable; do not silently skip them. Keep the following assertions:

- CPU F64 finite differences for all train_ops inputs and affine parameters, including constant input, L=1 and all-masked errors.
- CPU/CUDA F32 forward and gradient comparisons at atol=1e-5, rtol=1e-4; identical dropout masks.
- Both optimizers versus synthetic torch fixtures for ten steps; parameters/moments at atol=1e-6, including missing gradients and two groups.
- CPU resume bitwise equality and CUDA resume atol=1e-6, including moments/counters and restored groups.
- Invalid checkpoint cases reject unknown names, changed order/shape/dtype/version, missing/extra moment keys, and do not partially mutate parameters/state.
- Fused negative tests prove gradients are absent on the unsafe upstream path; safe operations retain gradients. The source guard rejects banned calls in nn/src except infer_ops.rs.
- CLI CPU smoke and unavailable-device failures. Run real cuda:0 smoke and cuda:9 failure checks on the acceptance machine.
- Before commit, run bash tools/check_private_data.sh. Fixtures contain synthetic/public values only.

## 7. Incorrect / correct example

Incorrect training code:

```rust
let weights = candle_nn::ops::softmax_last_dim(&scores)?;
```

Correct training code:

```rust
let weights = ferrots_nn::train_ops::softmax(&scores, scores.rank() - 1)?;
```

The upstream fused call drops the backward path. Use infer_ops only after explicitly freezing all inputs. Keep the source guard and gradient tests when changing this boundary.
