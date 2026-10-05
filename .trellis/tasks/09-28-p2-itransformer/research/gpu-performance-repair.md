# CUDA performance repair, 2026-09-30

## Authorization and state

The user requested stopping training and fixing low GPU utilization. This authorizes a scoped performance correction to the formerly frozen implementation. Batch size 32, accumulation 1, seeds, model dimensions, FP32 arithmetic, RNG identities, data protocol and numerical gates remain unchanged. Do not switch to release, install tools, create another Trellis task, use subagents, commit or archive.

The main session stopped executor PID67188 and its complete process tree, including PID25580, on September 30 at 08:42 America/Chicago. No related processes remained. The heartbeat is PAUSED. Two completed cement epochs and original checkpoints remain. The run has no final receipt and contributes zero completed cement entries. The user requested the interruption; no CUDA failure was established.

Stop evidence: runs/p2-development/tooling-transition-20260929-2129/user-requested-performance-stop-20260930-084222-4339185.
Baseline source and original executable: runs/p2-development/gpu-performance-repair-20260930/baseline/manifest.json.

## Change boundary

- Gap: CUDA training takes approximately 0.98 seconds per update while twelve device-utilization observations average 7.33 percent.
- Owners: NN owns deterministic dropout; Train owns gradient checks and optimizer boundaries; Cargo owns compilation profiles.
- First measure: add an opt-in CUDA example with synthetic cement-shaped inputs and the real model, loss, accumulator and optimizer. Measure synchronized forward/loss, backward/accumulation and update stages. Save final tensors for numerical comparison.
- Candidate corrections: optimize the measured host bottleneck without changing the ChaCha8 stream. Reduce gradient-check transfer or synchronization costs only if measurements justify the change.
- Expected files: the NN random implementation or its development-profile dependency settings; Train accumulation if measured; the diagnostic example and regression tests; task evidence and affected contracts.
- Exclusions: no model redesign, mixed precision, FlashAttention, larger batch, tolerance changes, private-data export or formal completion claims from short diagnostics.

## Ordered checks

1. Preserve baseline source/executable and stop/epoch evidence. Complete.
2. Measure the unchanged CUDA path with synthetic B32/L96/C18/K5/H15 inputs.
3. Apply the smallest supported correction and repeat the same diagnostic. Compare all parameters, both moments and predictions. Check dropout and non-finite gradients where affected.
4. Run CUDA workspace formatting, Clippy and tests using the pinned environment, then task/private-data checks and inline Trellis review.
5. Record the measured result and acceptance identity impact. Preserve all old receipts. Do not restart long formal training during the repair.

## Evidence boundary

The diagnostic is not a formal run. Runtime or binary changes prevent direct reuse of old Rust receipts as current-source acceptance. The prior validation-tool transition does not authorize a runtime-identity bypass. P2 stays in_progress. No CPU-only suite is scheduled.

## Baseline and correction

The unchanged CUDA path completed two warmup and ten measured updates. Mean times were 951.517 ms for forward/loss, 69.017 ms for backward/accumulation and 60.064 ms for update, totaling 1080.598 ms. A forward with dropout disabled took 4.928 ms. One largest dropout call took 269.233 ms. Evidence: runs/p2-development/gpu-performance-repair-20260930/profile-before/profile.json.

The first correction only changes Cargo development-profile optimization for ferrots-nn, ferrots-train, rand_chacha and ppv-lite86. These packages own the measured random-mask and gradient-validation work. The dev profile and its debug assertions remain; no release build or fast-math flag is introduced. Production Rust algorithms, masks, RNG calls, scalar types and synchronization boundaries remain unchanged. The follow-up diagnostic must verify all saved parameters, moments and predictions before accepting the correction. Gradient-transfer refactoring is deferred unless the corrected measurement still requires it.

## Final repair evidence: 2026-10-01

The repair quality suite passed with Rust CUDA 102/102 and Python 52/52. No failed or ignored test was recorded. The current development binary passed the cuda:0 doctor and training smoke checks. The current binary SHA256 is a8cc02c50065b714cecd96a55edc2a018a6854461a484d47382add1173585379.

The synthetic before/after comparison measured 115 tensors: 38 parameters, 38 first moments, 38 second moments and one prediction. All tensors were bitwise identical. All 12 loss values matched exactly. Mean synchronized training step time changed from 1080.59810 ms to 85.94863 ms. The measured speedup is 12.572604x.

The bounded real-data diagnostic used the unchanged cement H15/seed2021 configuration, batch32, accumulation1, FP32 and 128 optimizer steps. The training process was PID52824. It started at 2026-10-01T00:14:51.0794780Z and exited at 2026-10-01T00:15:27.3477064Z with exit code 0. The training summary reports 128 optimizer steps and 14.3774837 seconds of training compute time. The diagnostic output is under runs/p2-development/gpu-performance-repair-20260930/real-data-128-20260930-191449-2643853.

The GPU observer recorded 34 one-second samples while PID52824 was present in the compute-process list. The reported device utilization ranged from 0 to 20 percent and averaged 5.5 percent. The sample interval does not resolve short synchronized GPU bursts. The result confirms CUDA execution and the large throughput improvement. It does not establish a sustained utilization target.

The user-requested training stop remains final. No formal ETT or cement run was restarted. Historical Rust receipts remain tied to the previous runtime identity. P2 remains in_progress.
