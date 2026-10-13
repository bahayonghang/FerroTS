# Informer implementation evidence

## Current boundary

The user approved sequential implementation on 2026-10-13. P3 is in_progress on dev; P2's outstanding CPU/device clauses remain open and its prior receipts are unchanged. The original implementation delegation was interrupted by account usage limits; the main session integrated and verified the written work, with an independent check agent reviewing the resulting code.

Formal P3 matrix at development handoff: 0/105 entries. Live execution progress is subsequently recorded under the frozen suite entries and executor status. Development diagnostics and synthetic references do not count toward this matrix. The user approved the measured budget on 2026-10-13. Remaining work includes complete ETT/reference training, D4, cement9, independent private recomputation and final acceptance.

## Implemented mechanisms

- Strict model configuration, static dispatch and preview capability entry; 16 ETT profiles, three private engineering profiles and a synthetic smoke profile.
- CUDA PrefixSum/VJP, overlapping MaxPool1d, differentiable row replacement, sparse self/cross attention, calendar/value/position embeddings, distillation and zero-future decoder.
- Named buffers through snapshot, v2 save/load/resume/freeze/inference; isolated validation/evaluation/prediction identities and no supervision access.
- Isolated pinned official reference wrapper, synthetic CUDA fixture generator and bounded smoke/budget command. P2 frozen validators were not edited.

## Completed checks

- Rust workspace with CUDA feature: 120 passed, zero failed/ignored; log runs/p3-development/20261013-initial/workspace-cuda-tests.log. This required workspace command includes existing CPU numerical tests; no separate CPU training or CPU-only suite was scheduled.
- CUDA Clippy all targets with -D warnings: passed; clippy-attempt-2.log in the same directory.
- Python tool regressions: 61 passed, zero failures; python-tests-final.log (including tail-batch observer regression).
- CUDA reference generator and bitwise check: two cases, 72 parameter gradients each, six operator cases. Artifact identity and hashes are in tests/fixtures/p3/manifest.json.
- Fresh CLI Informer restoration: ten cases, including eight dropout/sparse/BatchNorm continuation cases with N3/7 and accumulation1/2, missing supervision and tail groups; two strict-patience cases disable distillation because zero learning rate does not freeze BatchNorm statistics. Reports are runs/p3-development/resume-process-cuda-*.json with model=informer.
- Operator peak: runs/p3-development/20261013-initial/prefix-operator-peak.log reports n1,000,000 FP32, baseline8MB, peak16MB, delta8MB=2 input sizes. The exact production scan callback is measured with output and gradient alive. Outer loss/GradStore are excluded by the original design's supplied-cotangent boundary.
- Sparse shape/allocator measurements at five L values: attention-complexity.log; all four attention sites use u×Lk scores, including causal mask.

## Retained diagnostic evidence and fixes

The broader multiply/sum/full-autograd memory diagnostic measured delta40,000,004 bytes (10.000001×). Its failed measurement remains in complexity-attempt-1.log and complexity-diagnosis.log. Independent review confirmed the design's operator forward-plus-VJP boundary; no claim is made that the full scalar graph or training memory was optimized to 2×. Native pool watermark semantics are documented by [NVIDIA CUDA 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html).

The decoder forecast is a noncontiguous slice. CUDA target selection previously failed; noncontiguous-targets-reproduction.log preserves the reproduction. The shared target-selection boundary now materializes contiguity and its regression passes.

An unselected exp branch in ELU produced NaN backward when BatchNorm affine bias was 100. elu-reproduction-2.log preserves that failure; model-and-elu-pass.log records the corrected, masked-exponent implementation and retained official gradient tolerances.

The pinned TSLib contains only an ETTh1 Informer script. Other ETT profiles are explicitly derived; its data_factory keeps tail batches. P3 configs use drop_last=false, avoiding the separate iTransformer upstream's drop_last=true policy. source-profiles.json records source/license/profile/loader fingerprints.

## Measured diagnostics and approved budget

The bounded diagnostic command is tools/validation/p3_smoke.py. It completes synthetic training, pauses three real-data Rust runs after 32 updates, and runs one official ETTh1 epoch. Its output is diagnostic only; the 105-command manifest is prepared for review, not executed. All bounded diagnostics completed normally; budget.json and planned-matrix.json are under runs/p3-development/20261013-smoke-v1/.


| Diagnostic | CUDA updates/epochs | Steady windows/s | Device-wide sampled peak MiB |
| --- | --- | --- | --- |
| Synthetic | 9 updates / 3 epochs, evaluate and future predict | 154.18 | 513 |
| Rust ETTh1 H96 | 32 updates / no complete epoch | 151.77 | 3553 |
| Rust ETTh1 H720 | 32 updates / no complete epoch | 42.77 | 10945 |
| Rust private cement H60 | 32 updates / no complete epoch | 178.70 | 3265 |
| Official ETTh1 H96 | 1 epoch, 8449 windows | 2539.10 | See command memory receipt; torch allocated peak 733044736 bytes |

The Rust real-data pauses are intentional diagnostic boundaries, not completed formal training. Peaks in this table are 50ms device-wide samples, include other processes and are not the native allocator AC2 measurement. The official wrapper's initial relative model-directory import failure is retained; only that reference command was rerun after fixing filesystem paths. Independent review then found P2's full-batch observer overcounted TSLib's retained tail. A P3-only adapter now counts the real loader dataset (8449 training and 8417 steady windows); the frozen P2 tools were not edited. budget-before-tail-fix.json and both official attempts remain retained. The latest official command is official-command-20261013-113642-915446. Final observer/tool hashes are in budget-tools-final.json; Rust source/config/binary identities remain bound to their original diagnostic receipts.

At at most ten epochs, estimated training-only costs are Rust ETT 36.63 GPU hours, Rust cement 41.76 GPU hours and official ETT approximately 1.11 GPU hours using only the measured H96 rate. Interior Rust horizons are interpolated between H96/H720; private H15/H30 use H60 rate and its slightly smaller window count. These are estimates, exclude validation/checkpoints/evaluation/export/parsing, and do not predict early stopping or two-GPU elapsed time. Roughly 80 GPU hours before those overheads is the budget approval input, not a runtime guarantee.

Disk estimate: a measured full-size Rust generation is 156484134 bytes. Current trainer retains up to two generations per epoch (best plus last), so 57 runs × 10 epochs × 2 generations is about 166.14 GiB before other artifacts. Reserve approximately 20 GiB for official checkpoints/arrays, shared caches, metrics, compressed three-variable recomputation export and temporary writes: approximately 186 GiB total planning allowance. Available space measured after diagnostics is approximately 200 GiB. This leaves limited headroom; do not delete old suites, change retention or export all private rows without authorization. This estimate is not an observed final artifact size.

Formatting and CUDA doctor training smoke also passed (doctor-cuda.json). Formal orchestration, immutable suite freeze, per-run acceptance receipts, 32-item D4, cement9/108-cell reports, full private recomputation and final quality/review remain required. The proposed 105-command manifest is reviewable input, not a complete formal acceptance runner.

Independent final narrow review passed for the corrected observer, source paths, 105-command coverage, diagnostic/formal boundary, budget arithmetic and disk estimate. Development implementation commit: 5f0d74d0c853bb0ebf4f1cc765863eb6fa25e825. The development checks bind the runtime source digest e84a8431f81ba3deea32c3baa688eb5dc8485598db6a757489d0dd1f5fd42bde; formal execution must freeze a new suite/binary identity after approval.


## Formal execution preparation

The user explicitly approved the reported budget after reviewing the development commits. tools/validation/p3_acceptance.py now owns freeze/run/verify; p3_reports.py computes the unchanged 32 D4 checks, 108 private cells with published P1/P2 references, and 105 performance records. New tool regressions bring the Python suite to 72 passed. No runtime Rust change was made; existing CUDA120/Clippy/fixture evidence retains its exact runtime source digest.

Each suite freezes a private binary copy, current runtime source, all model profiles, reference/validation tools, Python/Rust tests, all fixtures, upstream SHA, package environment and data fingerprints. File and command hashes bind accepted entries; failed command attempts and rejected candidate entries remain retained. A nonblocking Linux flock prevents another executor from sharing the suite. Two sequential workers each own one CUDA device. A failed cell stops further dispatch after the other GPU completes its current cell. ETT96 must meet the original D4 threshold before cement9. The executor checks full private coverage and independent recomputation before final CUDA quality; a passing execution still requires the main independent evidence review and remains preview until that completes.

Independent review checked real diagnostic config/output schemas and found the duplicate-executor, failure-dispatch and missing-validation-input freeze gaps. All three were fixed before suite creation. P2 frozen tools remain unmodified; only generic execution/receipt/epoch-check functions are imported. No formal training was started during implementation of these tools.


## Frozen suite and launch

Suite runs/p3-acceptance/20261013-linux-cuda is frozen from commit 1aae46be9bc6ff25bf2e607e32aea90b2be8df4f. Suite SHA256 e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec; runtime source remains e84a8431f81ba3deea32c3baa688eb5dc8485598db6a757489d0dd1f5fd42bde. Its suite-owned binary and 130 validation inputs are fingerprinted. Initial suite quality passed Rust120/Python72, CUDA Clippy, format, doctor, actual cuda:9 rejection, task validation, privacy and diff checks.

Active executor PID2117817, process start ticks176533124, launched at 2026-10-13T11:56:43.674597+00:00 with the virtualenv Python path. launch-attempt-2.json and executor-attempt-2.log are the current launch/log. An earlier launcher resolved the virtualenv Python symlink to the base interpreter; its owned process group was terminated during initial quality, before formal training. Original launch/log, executor-before-launcher-fix.json and launcher-correction.json remain retained. No frozen tool or runtime changed during this launcher correction.

At launch there were 0 accepted formal entries. The executor performs data audits, ETT96, unchanged D4, cement9, independent recomputation and final quality in order. This launch record does not claim that those pending stages completed. Do not start another executor or modify frozen inputs while it runs. Independent final evidence review and task/report reconciliation remain required even if final.json later passes.
