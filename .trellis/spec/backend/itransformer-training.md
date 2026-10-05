# iTransformer and resumable training contracts

## 1. Scope / Trigger

Read this contract before changing iTransformer, model dispatch, training state, v2 checkpoints or resume commands. Core owns configuration and tensor-free descriptors. NN owns models and parameter mappings. Train owns state progression and persistence. CLI owns arguments and provenance. Data contracts remain in data-pipeline.md.

The implementation source is thuml/iTransformer at c2426e68ca13f74aaec08045c5c724d8ad328124. The current formal acceptance state belongs to the active P2 task and its immutable run receipts. The presence of this specification does not establish formal reproduction.

## 2. Signatures

- `ForecastModel::from_config(config, device)` constructs the static baseline or iTransformer variant. `freeze_for_eval()` creates independent detached weights.
- `ForwardContext { mode: Mode, forward_id: u64 }` supplies the training microbatch identity.
- `ITransformer::official_weights()` and `load_official_weights()` convert pointwise Conv1d shapes while preserving official parameter names.
- `TrainingSession::advance() -> Result<Progress>` advances one accumulation group or one validation boundary. Fresh training and resumed training use the same function.
- `TrainingSession::snapshot()` requires empty gradient accumulation. `TrainingSession::resume(config, data, device, snapshot)` validates the complete state before returning a runnable session.
- `checkpoint::load_session(path, config, data, device, provenance)` verifies the persisted bundle and its current execution bindings.
- CLI: `ferrots models list --task forecast --device cpu|cuda:0`; `ferrots train --config <path> [--stop-after-steps N]`; `ferrots resume --checkpoint <path> [--stop-after-steps N]`.

## 3. Contracts

### Model and configuration

The strict iTransformer variant accepts d_model, n_heads, d_ff, e_layers, dropout, activation and use_norm. Dimensions are positive; d_model is divisible by n_heads; dropout is finite and lies in [0,1). Activation is gelu or relu. Unknown fields and decoder parameters fail. Future-known numeric variables and unavailable history fail. Targets form an ordered subset of input columns.

History is FP32 [B,L,C]. History-time is [B,L,K], with K=0, 4 or 5. Time tokens follow variable tokens. Projection removes time tokens before target selection. Checkpoint bindings include L, H, C, K, column order, frequency, model/mapping versions and data protocol.

Window means are detached. Population variance and standard deviation retain gradients. Normalization and LayerNorm use epsilon 1e-5. The encoder uses post-norm attention and feed-forward residuals, followed by final LayerNorm. GELU uses erf. Training must retain the backward paths governed by training-foundation.md.

The two-layer model has 38 parameter tensors. Names match the official state_dict. Conv1d weights [out,in,1] map to Rust [out,in]; all other parameter shapes are unchanged. Loading validates the complete name, shape, dtype, device and finite-value set before assignment.

### Randomness and optimizer boundaries

Initialization consumes the pinned PyTorch-compatible MT19937 stream in official construction order. Sorting parameter names occurs after initialization. Sampling uses the immutable post-initialization state and the established five-draws-per-epoch policy.

Dropout uses the existing ChaCha8 contract. Site 0 is embedding dropout; layer i owns sites 1+4i through 4+4i. Each training microbatch consumes one forward_id, including empty-supervision batches. An optimizer update can cover several microbatches. Evaluation does not advance the training stream.

TrainingState binds phase, epoch, next_batch_offset, permutation hash, global_optimizer_step, next_forward_id, weighted loss aggregates, split access, epoch history, early stopping and best generation. Scheduler state binds the applied epoch and learning rate to optimizer groups. RNG state stores all 624 MT19937 words and the index. A single seed cannot replace the persisted state.

The phases are train, validate and complete. A pause counts real optimizer updates and occurs only after the accumulation group is empty. A pause at the final training group restores into validate. Completed sessions do not invent additional updates. Type1 timing retains the P1 contract. Use serde_json float_roundtrip when reading persisted FP64 state; a one-bit parsing change can break exact continuation.

### V2 persistence and provenance

A v2 generation contains manifest.json plus model.safetensors, optimizer.safetensors, optimizer-state.json, buffers.safetensors, trainer-state.json, scheduler-state.json, rng-state.json, scaler.json, data-schema.json, resolved-config.toml and environment.json. The manifest hashes the other eleven files. The current model has a valid empty buffers.safetensors file.

Publication writes an immutable staging generation, verifies payloads, publishes the generation and appends a best/last reference. Faults must preserve the last committed reference and live state. A damaged latest reference fails. Best selection requires strict validation improvement.

Resume verifies config/data/scaler, source digest, exact environment and device. CLI provenance includes Rust, target, features, Candle, host/thread settings, GPU identity and CUDA versions. Use docs/environment.md and tools/validation/p1_env.ps1 for the measured Windows environment; keep RUSTUP_AUTO_INSTALL=0 and Rust 1.98.0.

A continuation creates a separate run directory, records the parent manifest and inherits a verified historical best generation. The parent run remains unchanged. V1 bundles remain readable for inference and fail resume with an explicit v2 requirement.

## 4. Validation & Error Matrix

| Condition | Required result |
| --- | --- |
| Invalid heads, dimensions, dropout, activation or unknown fields | Reject before creating training output or model tensors |
| Wrong L/H/C/K, columns, frequency, model or mapping version | Reject the binding |
| Missing, extra, malformed or non-finite parameter | Reject before assigning weights |
| Missing file, changed checksum or nonempty unexpected buffers | Reject the bundle |
| Invalid cursor, phase, permutation, RNG, scheduler or optimizer counter | Reject resume |
| Changed data, scaler, source, environment or device | Reject resume; no device fallback |
| V1 inference bundle supplied to resume | Explain that resume requires v2 |
| Fault during generation publication | Retain the previous committed reference and live state |
| Official NumPy 2 reference raises on np.Inf | Restore the equivalent np.inf alias only in the reference subprocess and record the wrapper identity |

## 5. Good / Base / Bad Cases

- Good: train N updates with dropout enabled, save at an empty accumulation boundary, terminate the process, resume N updates in a new process and compare all parameters, moments, counters, RNG, cursors and predictions.
- Base: evaluate or predict from a historical v1 bundle without adding training state.
- Bad: restore only model weights and optimizer step while discarding moments, the sampling state or the next microbatch identity.
- Bad: reuse a formal receipt after changing source, binary, configuration or required tooling fingerprints. Preserve failed attempts and rerun affected experiments.

## 6. Tests Required

Keep config/model/mapping negative cases; p2_layers and itransformer tests; p2_numerics; checkpoint_v2; trainer continuation cases; CLI resume and resume_process; models list; and Python reference/acceptance runner tests.

Forward and all true MSE gradients use atol=1e-5, rtol=1e-4. Key-projection bias can have mathematical zero MSE gradients. An explicit intermediate projection probe proves connectivity. Do not use floating-point residuals as connectivity evidence.

The user-approved AC3 gate supplies the same official true MSE gradients to both Adam implementations. Compare every parameter, both moments and step at atol=1e-6, rtol=0. Retain independent-backward update differences as diagnostics. Do not modify production gradients or remove key-bias parameters.

Fresh-process restoration covers N=3/7, accumulation=1/2, empty supervision, tail groups, scheduler transitions and strict patience. CUDA tensor comparisons use atol=1e-6, rtol=0. Existing CPU bitwise evidence remains valid for its recorded source. Preserve all corruption and publication-fault cases.

Follow the 2026-09-29 CUDA-first user instruction in AGENTS.md. Do not schedule routine CPU-only builds or CPU training. After a CUDA failure, retain the original evidence and use about three CPU epochs for diagnosis only when needed. A short diagnostic does not complete the original unexecuted CPU formal control.

Formal CUDA acceptance requires 48 official ETT runs, 48 Rust ETT runs and 9 cement runs. Seeds are 2021/2022/2023. All 32 D4 comparisons use population standard deviation and the unchanged two-sided bound. Cement evaluates all 18 variables; independent recomputation covers all 18,574,560 selected prediction rows and six MAE/RMSE checks with differences below 1e-6.

Every run retains synchronized epoch timing and steady throughput. Device memory uses 50 ms device-wide sampling with baseline, sample count and gaps. Report allocator peaks separately when available. Missing process memory is unavailable, not zero.

Original-unit metrics store variables in a dictionary keyed by the exact configured column name. Validate the complete 18-name set and each variable's full test count before aggregation. Resolve values by name and use configuration order for reports. Dictionary iteration order is not a variable identity. P2 reports use three seeds and population standard deviation; accepted P1 single-seed baselines use n=1 with standard deviation not applicable.

A user-approved validation-tool correction can preserve unaffected training through an explicit versioned evidence transition. Audit the complete parent matrix with its original validator before installing the correction. Bind the parent suite, validator snapshot, entry inventory, receipt paths and hashes, prerequisite gates and audit. Require unchanged runtime, binary, configuration, reference and data identities. Retain original receipts and copy imported gate and entry bytes exactly. Reject unlisted, changed or nested evidence. Run current quality, affected gates and final acceptance in the successor suite.

## 7. Wrong vs Correct

Wrong: deriving dropout identity from optimizer updates repeats a mask across accumulated microbatches.

```rust
ForwardContext {
    mode: Mode::Train,
    forward_id: session.state.global_optimizer_step,
}
```

Correct: use the next training-forward identity. TrainingSession owns its increment and persistence.

```rust
ForwardContext {
    mode: Mode::Train,
    forward_id: session.state.next_forward_id,
}
```
