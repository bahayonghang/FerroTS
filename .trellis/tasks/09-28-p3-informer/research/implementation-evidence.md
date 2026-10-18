# Informer implementation evidence

## Current boundary

The user approved sequential implementation on 2026-10-13. P3 is in_progress on dev; P2's outstanding CPU/device clauses remain open and its prior receipts are unchanged. The original implementation delegation was interrupted by account usage limits; the main session integrated and verified the written work, with an independent check agent reviewing the resulting code.

Formal P3 matrix at development handoff: 0/105 entries. Live execution progress is subsequently recorded under the frozen suite entries and executor status. Development diagnostics and synthetic references do not count toward this matrix. The user approved the measured budget on 2026-10-13. Remaining work includes complete ETT/reference training, D4, cement9, independent private recomputation and final acceptance.

## Current-source single-case gate, 2026-10-18

The follow-up retained-checkpoint replay uses the same official model/evaluator for the full-run official epoch-2 best checkpoint, the independent default-precision official epoch-1 checkpoint, and Rust epoch-1 best weights with their BatchNorm buffers. The diagnostic script and 8-draw validation/test reports are `runs/p3-development/20261018-single-case/validation-replay/{score.py,report-val.json,report-test.json}` (SHA256 respectively `ee2254543719d2f821569f3948ae75d8ea7fd3dd4dc500fa876c9fda6cbf18dc`, `547951df11a7cb9681560c80ed7022230afd01b31b081cde3ae0ee9942e79289`, `6261604913d046c17f12922fb2c36b1ee2085eba5b14730160fd3ed5e0bc04ca`). Each split has 2,785 ordered windows, 88 batches including a one-window tail, and 1,871,520 labels. For each seed 2023–2030, the three candidates start from identical CPU/CUDA RNG states and have identical hashes over all 352 `torch.randint` sample calls. This compares candidate weights and checkpoint selection under common sparse draws; it does not reenact the training-time validation order/state, and these eight draws are repeated evaluations of one trained seed, not D4 seeds.

Under the *same full-window, label-weighted MSE* on validation, the 8-draw means are official epoch 1 `0.808436042`, official epoch 2 `0.796335118`, Rust epoch 1 `0.798213864`. Official epoch 2 beats official epoch 1 on all 8 draws by `0.011502–0.012452`; the batch-equal MSE also ranks epoch 2 better on every draw (`0.817678003` versus `0.805084362`). Validation MAE ranks official epoch 1 better (`0.629022` versus `0.647313`), consistent with the production selector using validation MSE rather than MAE. The candidate ranking is therefore not a tail-batch aggregation or sparse-sampling accident in this controlled replay, and it supplies no basis to switch the production early-stop rule.

With those same checkpoint weights, evaluator, windows and paired draws, the full-window *test* MSE means are official epoch 1 `2.701347369`, official epoch 2 `2.942311850`, Rust epoch 1 `2.676746030`; MAE means are `1.313394`, `1.399791`, `1.307308`. Official epoch 2 minus epoch 1 test MSE is `+0.240964481` (draw range `+0.239846–+0.242000`), while Rust epoch 1 minus official epoch 1 is `-0.024601339`. Thus the observed official-epoch-2/Rust-epoch-1 test MSE difference `0.265565820` is arithmetically dominated by the gap to the *independent* official epoch-1 control (about 90.7% of that particular difference). The full official run did not retain its own epoch-1 weights: its logged epoch-1 validation loss was `0.8192074`, versus `0.8245808` for the one-epoch control, so they must not be treated as an identical checkpoint. This is a counterfactual checkpoint comparison, not a strict causal decomposition of that run's checkpoint selection or proof of generalization/D4 behavior across seeds. Batch-equal test MSE also preserves all rankings; never subtract its values from full-window metrics. The earlier claim that checkpoint selection was only suggested by unmatched per-epoch test-loss sampling is strengthened by this paired replay, within that checkpoint-identity limit. There is no demonstrated selection bug to repair. Current-source ETTh1/H720 full guard, a new suite, D4 and production early-stop changes remain deferred.

The user authorized the bounded next step, not a new formal suite. The existing P3 session binding is unchanged. Runtime source, CLI binary and all five-batch Rust probes use `8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47`; the fixed ETTh2 CSV hash is `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. The independent `runs/p3-development/20261018-single-case/ETTh2-h96/` pair used seed 2023, the pinned official profile with cuDNN TF32 enabled, Rust's unchanged production configuration and GPU0. Commands, identities, source/config/data hashes and memory observations are in `report.json` and `identity.json`; `diagnosis.json` summarizes the comparisons. Both processes exited 0 and covered the full test shape `[2785,96,7]`. The official best validation epoch was 2 of 5; Rust's was 1 of 4. The diagnostic runner initially exited after the completed Rust train because it looked for a paused-only `status` field; its completed receipt and checkpoints were retained, the runner check was corrected, and evaluation resumed without retraining.

Current same-seed full-test metrics are official MSE/MAE `2.9419164657592773/1.3997275829315186` and Rust `2.6772834590976093/1.3073967803726594`; Rust minus official is `-0.264633006661668/-0.09233080255885917`. The old-source pair was `-0.31516601193091054/-0.08371309038421537`. MSE gap narrowed while MAE gap widened; this is neither a clean improvement nor a D4 result. D4 requires all three seeds from one frozen identity. A mixed old-two/new-one-seed calculation is diagnostic risk only and cannot count toward acceptance. The official same-seed metrics also vary between the old and current diagnostic runs; validation sampling and train-loss reduction differ, so their losses are not directly subtracted. Because the affected case remains substantially separated, the planned ETTh1/H720 full-training guard was not started; its current-source D4 margin remains unknown.

The same fixed initial weights, order and precision settings were then compared for five CUDA train updates. The earlier two-step tensors were copied to `runs/p3-development/20261018-single-case/step2-archive/` before expanding the probe. All four complete ProbSparse index tensors per batch match. All 60 dropout site masks match element-wise, including the required storage-order remaps; Philox offsets after each batch are `188,376,564,752,940`. The default official/Rust post-update maximum differences by batch are `1.99693e-4,3.62365e-4,3.71132e-4,4.70912e-4,5.29725e-4`. A separate control disables only official cuDNN TF32: the corresponding differences are `8.82521e-6,1.35861e-5,1.67266e-5,1.90036e-5,2.03867e-5`. Complete traces and comparisons are in `runs/p3-development/20261017-informer-steps/` and `...-cudnn-ieee/` (`comparison-5.json`, `rust-steps-5.log`, `official-steps-5.log`). Thus no new sampling or dropout divergence appeared through batch 5, and the cuDNN precision path remains the dominant measured early numerical difference. These five updates do not prove that precision alone causes the final test gap; no production precision, early-stop, seed or D4 rule was changed.

A further one-epoch-only official control used the same seed/profile, changing only process-local `torch.backends.cudnn.allow_tf32` before model construction; the pinned checkout and production runner were not edited. Its two clean process receipts, bootstrap hashes, full-window metrics and epoch-1 checkpoints are under `runs/p3-development/20261018-single-case/precision-control/`. Across 72 parameters versus the current Rust epoch-1 best checkpoint, default official had element-weighted mean absolute weight gap `0.000187357577`, max `0.020103414`; cuDNN TF32 off reduced these to `0.000116855283` and `0.010607132`. This confirms a material epoch-level precision contribution but does not make the trajectories identical. The default official one-epoch full-window MSE/MAE were `2.70133376121521/1.3134418725967407`, versus Rust epoch-1 `2.6772834590976093/1.3073967803726594`; the TF32-off official control was `2.6959941387176514/1.3135713338851929`. In the separate full official run, validation selected epoch 2 and its own per-epoch *batch-equal, stochastically sampled* test losses were `2.677197` at epoch 1 and `2.920647` at epoch 2. This, together with older same-seed official reruns selecting different epochs, supports the inference that checkpoint selection amplifies the full-run metric gap. The per-epoch test loss is not the same aggregation or sampling identity as the final full-window `metrics.npy`; the one-epoch control is also an independent process. Neither comparison by itself proves a causal D4 fix. Production early stopping remains unchanged.

The paired ETTh2 run also measured previously excluded costs. Official 5 epochs used 16.643 s train, 2.071 s validation, 0.085 s checkpoint and 24.547 s total command time; Rust 4 epochs used 254.685 s train, 3.232 s validation, 23.463 s checkpoint and 284.374 s total command time; Rust evaluation took 3.301 s. This single pair wrote 80,885,304 official bytes, 313,223,148 Rust bytes and 1,500,022 cache bytes (396,159,681 total diagnostic bytes). GPU0 sampled peaks were 1,365 MiB official and 3,585 MiB Rust. These are measured case costs, not validated multipliers for H720, GPU1 or cement. The prior 146.64–198.89 GPU-hour matrix estimate remains training-only; full overhead and disk budget cannot yet be signed off. Available `/mnt/data` space after the probes was 454,031,306,752 bytes. No new formal suite was frozen or launched, the old failed attempts remain, AC7–AC10 remain open, Informer remains preview, and P2/WaveForM/P4 boundaries are unchanged.

## Current CUDA diagnosis and recovery, 2026-10-18

Task binding: `python3 .trellis/scripts/task.py current --source` initially returned no current task in this Codex session. `task.py start 09-28-p3-informer` bound the existing in-progress P3 in `session:codex_01a14aeb-64ce-7860-8118-95ec80360ab7`; no model task was created. The pre-existing dirty diff and old failed suite remain. Current HEAD is `d1d73b9`; final `source_snapshot()` gives `8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47` over 52 runtime files. The final step probe, AC4 report and 32-step paused CLI runs use this source identity. The 20261014 suite froze runtime `ef70dd3488002a847d5354440ce79fc9ba992f6c64be923ec7adb1999bce097e`; no acceptance result transfers across these identities.

The bounded step probe fixed ETTh2/H96, seed 2023, dropout 0.1, the pinned ETTh2 data hash `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`, the official seed-2023 initial parameter/buffer fixture, batch order, FP32, `torch.backends.cuda.matmul.allow_tf32=False`, official `cudnn.allow_tf32=True`, and Rust `candle_gemm_reduced_precision_f32=false`. It ran two CUDA train batches, not an epoch. Official source/probe: `runs/p3-development/20261015-etth2-h96-s2023-cross/step_diagnostic.py`; Rust: `p3_d4_probe step-diagnostic`; full retained tensors and comparison JSON are under `runs/p3-development/20261017-informer-steps/`. Both sides used 32 windows and 21,504 supervised values per batch. Orders were `[8034,6390,3066,4116]` then `[864,1598,489,6334]`. All elements of each of the four ProbSparse index tensors matched at both batches; this is a whole-tensor check, beyond the earlier head-only tests.

The official dropout wrapper replayed each site's CUDA state on an all-ones tensor, restored that state, then ran the real dropout. It captured all 12 site masks per batch without changing the real forward stream. The Rust diagnostic regenerated masks from `cuda_philox::mask`; each site had zero differing elements after the existing `[B,C,L]` to `[B,L,C]` remap where applicable. The official and Rust Philox offsets matched at every site: batch 0 `0→188`, batch 1 `188→376`. The unchanged official CUDA-state SHA256 before/after each forward and complete masks are retained in the step trace. The batch-0 loss was official `1.196363568306` versus Rust `1.196367859840`; batch 1 official `1.351979374886` versus Rust `1.351985335350`.

The first material forward gap appears at `encoder.attn_layers.0.conv1`. Immediately preceding attention output differs by at most `2.15e-6`; the Conv1d output differs by `8.89599e-4`. With the same official weights and same Rust layer input, PyTorch `F.linear` differs from Rust by at most `1.67e-6`, while PyTorch/cuDNN `F.conv1d` differs by `8.88944e-4`. This isolates the default cuDNN 1×1 convolution precision path from the Rust FP32 matrix product. The default-precision batch-0 gradient maximum was `5.72e-5` (`encoder.conv_layers.0.downConv.weight`), Adam first-moment maximum `5.72e-6`, second-moment failure count zero at `1e-6`, and post-update parameter maximum `1.99693e-4` (`encoder.conv_layers.0.downConv.weight`). Batch 1's maximum post-update gap was `3.62515e-4`; these are observed differences, not a bitwise official-weight acceptance gate.

One diagnostic control changed only official `cudnn.allow_tf32=False`, saved separately under `runs/p3-development/20261017-informer-steps-cudnn-ieee/`; official source, data, weights, samples, dropout states, optimizer and order stayed fixed. Its batch-0 loss was `1.196367859840`, identical to Rust at the printed 12 decimals; batch-1 loss differed by `1.19e-7`. Across both batches, the largest parameter-gradient gap was `8.94e-8`, Adam first-moment maximum `1.12e-8`, and all complete ProbSparse indices still matched. Remaining post-update maximums `8.83e-6` and `1.21e-5` occur in `encoder.conv_layers.0.downConv.bias`, so equal forward/loss does not imply bitwise optimizer identity. This control identifies a numerical source of divergence; it does not alter the pinned official implementation, production precision, D4 threshold, seed, or early stopping. No speculative convolution emulation layer was added.

Before repair, `cuda_snapshot_checkpoint_resume_and_inference_keep_buffers` failed after resume on `dec_embedding.temporal_embedding.embed.weight` by about `1.37836e-6` (recorded Cargo log `/home/lyh/.local/share/rtk/tee/1792258479_cargo_test.log`). The minimal repair persists the Informer CUDA Philox offset in RNG v2, restores it before a resumed train forward, and rejects missing/misaligned/old Informer RNG records. Baseline/iTransformer retain RNG v1. The disposable MT19937 index cursor is rebuilt from persisted post-init state, epoch and batch; it is not separately serialized. A CUDA test compares every index element at every site across three epochs, including a one-window tail; checkpoint tests cover bad RNG payloads and frozen evaluation isolation.

Current-source validation: `cargo test -p ferrots-train --features cuda --locked --test p3_buffers` passed 6; `--test informer_order` passed 4; `cargo test -p ferrots-cli --features cuda --locked --test resume_process` passed 2. The Informer independent-process test exercised 8 dropout/ProbSparse/BatchNorm 2N versus N→checkpoint→N cases and 2 patience cases; all tensor maxima were 0.0, integer counters and complete RNG JSON matched exactly. Its final-source report is `runs/p3-development/resume-process-cuda-1792309642582625356.json` with source digest `8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47`. Final `cargo test --workspace --features cuda --locked` passed 132 tests in 51 suites; `cargo clippy --workspace --all-targets --features cuda --locked -- -D warnings`, `cargo fmt --all --check`, Python reference/validation 73 tests, privacy script and `git diff --check` passed. An earlier workspace run overlapped Clippy and failed transiently in the iTransformer CLI recovery test; its log is `/home/lyh/.local/share/rtk/tee/1792309642_cargo_test.log`. That exact recovery test passed alone, then the full workspace rerun passed. The first Clippy attempt's five new lint findings were fixed, then the complete gate passed. Task validation passed with a warning that this large evidence file exceeds the 32 KiB context-injection cap; this latest section is at the front.

Final-source bounded CLI runs under `runs/p3-development/20261018-budget-recheck/` paused after 32 CUDA optimizer updates each, without a validation epoch or formal scoring. Each had 992 steady train windows. ETTh2/H96 and ETTh1/H720 on GPU0 measured 130.888 and 34.038 windows/s respectively; ETTh1/H96 measured 130.419 on GPU0 and 63.024 on GPU1. Cement H60 measured 154.406 on GPU0 and 71.237 on GPU1. All six receipts and the assumptions are in `budget.json` at that path; the GPU1 H720 rate is extrapolated, not directly measured. Earlier two-step paused runs under `20261018-bounded-recheck/` used the pre-format source digest `1cb66538aa425f452eeabc77e77f1713ddf04626ddc1079a2377d3ac524a8bb2` and are retained as preliminary evidence. The 20261014 *old-source* ETTh1/H720 D4 margins were MSE `0.018374219934`, MAE `0.011763713868`; current-source D4 margins remain unmeasured. Old suite results remain official48 + Rust48, D4 30/32, ETTh2/H96 MSE/MAE failed, cement 0/9. Its metrics and receipts remain, but earlier authorized cleanup removed the old binary and many checkpoint payloads; do not claim full artifact verification.

Historical epoch train-loss deltas on 2026-10-16 (`+0.000571263064`, `+0.000756933925`) mixed Rust all-label weighted mean with the official `np.average` of per-batch means. ETTh2/H96 has 264 full batches and one one-window tail, so those subtractions are not a same-reduction trajectory measurement. The epoch-1 parameter differences are real. Validation comparisons likewise require the same sampling identities and reduction. That earlier repair phase did not start another full epoch or formal matrix. The old approval was roughly 80 GPU hours of training plus overhead and 186 GiB disk. The bounded final-source rates imply about 146.64 GPU hours of training under a half-on-each-GPU assignment, including 1.11 hours from an older official H96 GPU0 rate; an all-slow-GPU extrapolation gives 198.89 GPU hours. These are maximum-ten-epoch, training-only planning estimates, excluding validation, checkpoint, evaluation, export, parse, early stopping and overhead. GPU1 H720 is inferred from the measured H96 ratio; the assignment is an assumption, not a runner promise. At that earlier check `/mnt/data` had 465,140,174,848 bytes free (about 433.2 GiB); the 186 GiB disk allowance is historical, not a fresh write-volume measurement. The new budget materially exceeds the old approval, so a new suite must be separately reviewed and approved before it is frozen or launched. AC4 and AC5 remain checked with current CUDA evidence; AC7–AC10 remain open. Informer stays preview, P2 CPU/device clauses remain open, and WaveForM/P4 were not started.

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

At launch there were 0 accepted formal entries. The executor performs data audits, ETT96, unchanged D4, cement9, independent recomputation and final quality in order. This launch record does not claim that those pending stages completed. Independent final evidence review and task/report reconciliation remain required even if final.json later passes.

## Live status 2026-10-14T01:50:21Z

Re-read on branch `dev` at `78872055d2ebf8bd79265f2380ed2e133b3ab6d2` (`7887205`). The branch is 5 commits ahead of `origin/dev`. Working tree tracked files are clean. Untracked `docs/results/p3-informer-ett.md` and `docs/results/p3-informer-performance.md` are still present; both describe this same suite as D4 29/32 and 96/105 performance rows. They were not deleted. No frozen suite file was rewritten in this check.

| Item | Observed |
| --- | --- |
| Suite SHA256 | `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec` (`executor.json`) |
| Runtime source | `e84a8431f81ba3deea32c3baa688eb5dc8485598db6a757489d0dd1f5fd42bde` |
| TSLib | `4e938a1767106324dd753b2a44832bf870a0252e` |
| Entries | 96: official ETTh 24, official ETTm 24, rust ETTh 24, rust ETTm 24 |
| D4 | `complete=true`, `passed=false`, `ddof=0`, official 48, rust 48, 32 rows, 29 passed |
| Cement | `runs=0`, `complete=false`, `passed=false`, `rows=[]` |
| Performance file | `complete=false`, `passed=false`, 96 run records, not 105 |
| Executor | `status=failed`, historical PID 2117817, ended `2026-10-13T23:43:30.742923+00:00`, error `ValueError: D4 failed; original bounds and evidence retained; cement not started` |
| Process | PID 2117817 is not running. No ferrots/executor process. `executor.lock` is a 0-byte file with no `lsof` holder |
| GPU | `nvidia-smi` sees two RTX 4090s, both idle (18 MiB and 106 MiB used) |
| Disk | `df` on `/mnt/data` reports 150G available |

Failed D4 cells, unchanged formula:

| Dataset | H | Metric | Official mean | Rust mean | Difference | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 720 | mse | 1.1988215049107869 | 1.1460233534584012 | 0.052798151452385644 | 0.023976430098215738 |
| ETTh1 | 720 | mae | 0.8701238036155701 | 0.8488760034801098 | 0.021247800135460237 | 0.017402476072311403 |
| ETTh2 | 96 | mae | 1.3526554107666016 | 1.4185109745838573 | 0.06585556381725577 | 0.06523959602471169 |

ETTh1/H720 Rust means are lower than official and still fail the two-sided bound. ETTh2/H96 MAE is above the bound by about 6.2e-4. `runs/p3-development/20261013-initial/live-d4-diagnosis.md` remains an earlier 60-entry hypothesis, not this 96-entry result.

Dispatch: this session's `spawn_subagent` enum is only `codex:codex-rescue`. It cannot set `subagent_type` to `trellis-implement` or `trellis-check`, and those skill files are absent. Implementation continues inline. The CUDA feedback loop is recorded under `runs/p3-development/20261014-d4-cuda/` and does not write into the frozen suite. Informer stays `preview`. P3 stays `in_progress`. P2 CPU/device clauses stay open. No new formal suite starts before a separate budget approval.

## CUDA loop 2026-10-14T02:52Z

Device `cuda:0`. Candle f32 GEMM reduced precision stayed false. Process-default Torch is matmul TF32 off and cuDNN TF32 on. `NVIDIA_TF32_OVERRIDE=0` forces the cuDNN conv gap to 0 while leaving the Torch flag print unchanged. Suite SHA256 was re-read after the runs and is still `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. `df` available bytes were 153101918208 (142.59 GiB). The frozen suite directory is 51G. Detail is in `runs/p3-development/20261014-d4-cuda/causes.md`.

| Check | Result |
| --- | --- |
| IEEE B32 and tail B17 forward, loss, gradients, BN buffers | exceed 0 at 1e-5/1e-4 |
| Five eval identities | MSE/MAE spread about 1e-4 to 3e-4 |
| Canonical eval of the six failed-cell checkpoints | matches frozen evaluation.json to 12 printed decimals |
| Best epoch | not a sole cause; ETTh1 seed 2022 both selected epoch 2 and Rust test MSE stays 1.129365642292 |
| cuDNN TF32 | ETTh2 seed 2023 epoch-1 val stays near 0.82; ETTh1 seed 2022 epoch-2 final MSE stays near 1.206 |
| Torch initial weights inside the Rust training stream | ETTh2 seed 2023 test MAE 1.473323772315 -> 1.300776352932 (official 1.309546709061). ETTh1 seed 2022 epoch-2 test MSE 1.129365642292 -> 1.132254618507 (official 1.206086277962) |

No Informer math change was made. The weight substitution fixes the ETTh2 seed 2023 test MAE contribution and does not fix ETTh1 seed 2022. The remaining ETTh1 gap is the unsplit training RNG bundle (loader order, dropout masks, ProbSparse indices). A new formal suite of the current code would repeat D4 29/32. Cement stays 0/9. Informer stays `preview`. Task stays `in_progress`.

## RNG split 2026-10-14T03:20Z

ETTh1/H720 seed 2022 on CUDA. CPU dropout advances the CPU RNG. CUDA dropout does not; `torch.randint` without a device does, and it stays on CPU. With no forward, both official loader epochs match Rust `training_permutation` (train sha256 `63a5adfa3e9174942f9bc3c58ec82e69261aa877845f041d857cb8e7e5fdd3d7`, then `926721802ace75c48b5f2154d6df20b7a94d98205683f2de2d4450b16ae34b23`). In the real official 2-epoch run the first train order still matches, and the second is `45a15072381dafa7c6a8f4801eb52e5c63e30945abb3456ed8009278e9da210f`. Test order stays sequential on both sides. Official dropout 0 repeats those same six loader hashes. Its final mse is `1.2166088819503784`. Rust dropout 0 epoch-2 canonical mse is `1.152109353833` against the dropout-0.1 Rust value `1.129365642292`. The dropout-0 gap is still about 0.0645, so masks do not account for the cell. The Rust run that installs the official train order and keeps Rust sample indices finished with exit 0. Epoch-1 canonical mse stays `1.118315526347`. Epoch-2 canonical mse moves from `1.129365642292` to `1.199680963001` (mae `0.880491869449`). Frozen official seed 2022 is mse `1.206086277962`, mae `0.880467236042`. This is one seed, not a three-seed D4 pass. Informer math is still unchanged. Detail: `runs/p3-development/20261014-d4-cuda/rng-split.md`.

## Order RNG 2026-10-14T03:50Z

`torch.randint` on CPU consumes one MT19937 word per element when `high < 2^28`, and two words otherwise. PyTorch 2.14.0 `08187d9e` confirmed this against `ReferenceRng::consume_torch_randint`. Splitting or joining calls with the same `high` does not add a per-call word. ETTh1/H720 ProbSparse sites are `(96, 1440)`, `(49, 588)`, `(768, 16128)`, `(49, 9216)`. Replaying official init, then DataLoader lengths 7825/2161/2161 at batch 32 with those calls between loader iterations, reproduces train epoch-0 sha256 `63a5adfa3e9174942f9bc3c58ec82e69261aa877845f041d857cb8e7e5fdd3d7` and epoch-1 sha256 `45a15072381dafa7c6a8f4801eb52e5c63e30945abb3456ed8009278e9da210f`.

`InformerSamplingSchedule::permutation` is what `TrainingSession::rebuild_order` calls for Informer. It keeps the five loader draws and inserts those randint words after each split's forwards. Models without `torch.randint` still use `training_permutation`. ProbSparse index values stay on the isolated stream. Tests: `torch_randint_word_count_and_empty_sites_keep_five_draws`, `etth1_h720_seed2022_epoch1_order_matches_official_loader`, `session_epoch2_order_uses_the_randint_schedule`. All three passed. The shipped two-epoch CUDA recheck on physical GPU 0 finished with exit 0 at about 379 seconds (`train1-etth1-s2022-orderfix.log`). Epoch-1 canonical mse/mae stayed `1.118315526347` / `0.836556950170`. Epoch-2 canonical mse/mae are `1.199680963001` / `0.880491869449`, count `10891440`. These match the earlier order-injection run exactly. Frozen official seed 2022 is mse `1.206086277962`, mae `0.880467236042`, so this seed's residuals are about `0.006405` and `0.000025`. Validation loss at epoch 2 is `1.657448756776`. Seeds 2021 and 2023 were not measured at this timestamp. The next section records their full patience runs. ETTh2/H96 is not explained by this order fix. No new formal suite. Informer stays `preview`. P2 CPU/device clauses stay open.

## ETTh1 seeds 2026-10-14T04:40Z

Physical GPUs are both RTX 4090. Seed 2021 used GPU 0 and seed 2023 used GPU 1, started together at 2026-10-14T03:59:03Z. The binary was the already-built debug probe; no source change after the seed-2022 two-epoch run. Both logs exit 0. Selection is the strict minimum `validation_loss` (`EarlyStopping::observe` updates only when `value < best`). The canonical numbers below are that epoch's in-loop `TEST_CANONICAL`, count `10891440`. They are diagnostics, not D4 cells.

| Seed | Epochs run | Selected epoch | Validation | Canonical MSE | Canonical MAE | Frozen official MSE | Frozen official MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 1–10, hit max_epochs | 9 | 1.658271457046 | 1.179827201136 | 0.851581106997 | 1.2001084089279175 | 0.8670483231544495 |
| 2023 | 1–6, then `TRAIN1_COMPLETE_WITHOUT_EPOCH` | 3 | 1.657692409991 | 1.181337900121 | 0.862242555373 | 1.1902698278427124 | 0.862855851650238 |
| 2022 | 1–2 only | not selected | 1.657448756776 at epoch 2 | 1.199680963001 at epoch 2 | 0.880491869449 at epoch 2 | 1.206086277961731 | 0.8804672360420227 |

Seed 2021 residuals versus its frozen official run are mse `-0.020281207792` and mae `-0.015467216157`. Its old Rust suite had selected epoch 2; this schedule's minimum validation is epoch 9. Seed 2023 residuals are mse `-0.008931927722` and mae `-0.000613296277`. Its selected epoch is 3, the same index as the frozen official best epoch. Seed 2023's per-epoch wall time was about 343 seconds while seed 2021's was about 186 seconds; both processes were alive on separate cards. Logs: `train1-etth1-s2021-orderfix.log`, `train1-etth1-s2023-orderfix.log`.

Seed 2022's two-epoch log is not the patience-3 selection. Its full run is in the next section. The code commit for this schedule is `ce6754c012864689345a3a0e1d98ce7a3ea56242`.

## ETTh1 three-seed mean 2026-10-14T05:00Z

`train1-etth1-s2022-orderfix-full.log` exited 0. Epochs 1 and 2 repeat the two-epoch log on every printed loss and canonical metric. Epoch 3 validation `1.654460136097` is lower than epoch 2 `1.657448756776`, so the strict minimum is epoch 3. Its canonical mse/mae are `1.144464620375` / `0.845012104345`, count `10891440`. The run then had three non-improving epochs and stopped after epoch 6. Frozen official seed 2022 stdout keeps epoch 2: validation goes from `1.6437318` to `1.6516221` at epoch 3, so the official best stays at epoch 2. The same direction split appears on seed 2021: official minimum is epoch 4 (`1.6736959`), while this Rust schedule's minimum is epoch 9. Seed 2023 matches the official index, epoch 3.

Selected diagnostics, not D4:

| Seed | Selected epoch | MSE | MAE |
| --- | --- | --- | --- |
| 2021 | 9 | 1.179827201136 | 0.851581106997 |
| 2022 | 3 | 1.144464620375 | 0.845012104345 |
| 2023 | 3 | 1.181337900121 | 0.862242555373 |

Population mean (`ddof=0`) is mse `1.168543240544` (std `0.017037322066`) and mae `0.852945255572` (std `0.007100130828`). Frozen official means stay `1.1988215049107869` and `0.8701238036155701`. Absolute differences are mse `0.030278264367` and mae `0.017178548044`. Original thresholds are mse `0.023976430098` and mae `0.017402476072`. The MSE cell is outside. Rust is lower. The MAE difference is inside that printed threshold by about `0.000224`, which is inside the earlier eval-identity spread of about 1e-4 to 3e-4, so this diagnostic does not establish an MAE pass. Detail: `runs/p3-development/20261014-d4-cuda/etth1-h720-three-seed.json`.

Reading the Rust canonical metric at the frozen official best epochs (2021 epoch 4, 2022 epoch 2, 2023 epoch 3) gives means mse `1.191210447528` and mae `0.868940048325`, differences `0.007611057382` and `0.001183755291`. Both sit inside the original thresholds. That slice is not the selected checkpoint and is not a D4 result. It localizes the remaining MSE miss to validation ranking after the order schedule, not to the epoch-2 test metric. No new formal suite is requested. Cement stays blocked. Informer stays `preview`. P2 CPU/device clauses stay open. ETTh2/H96 is still a separate init gap.

## Torch constructor on the order schedule 2026-10-14T05:20Z

`train1-etth1-s2022-orderfix-torchinit.log` exited 0. Start `2026-10-14T05:02:55Z`. Physical GPU 0. `FERROTS_PROBE_DEVICE=0`. `weights_replaced=72` from `fixtures/init-torch-etth1-s2022-dropout0.1.safetensors`. `candle_gemm_reduced_precision_f32=false`. data sha256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`, the same ETTh1/H720 window as the order-only run. Count `10891440`. This is one seed, not D4.

| Epoch | Validation | Canonical MSE | Canonical MAE | improved |
| --- | --- | --- | --- | --- |
| 1 | 1.746455730802 | 1.132025991034 | 0.844369470972 | true |
| 2 | 1.646344067709 | 1.198260877893 | 0.876906379074 | true |
| 3 | 1.653986671528 | 1.153196113900 | 0.843973867057 | false |
| 4 | 1.676078764321 | 1.163450792951 | 0.839186055958 | false |
| 5 | 1.681891089757 | 1.157973472075 | 0.832249310464 | false |

Epoch 1 matches the earlier torch-inject run that used the old five-draw order (`train1-etth1-s2022.log` inject block: mse `1.132025991034`, mae `0.844369470972`). The order schedule does not change epoch 1. Epoch 3 validation rises by `0.007642603819`. Frozen official stdout rises by `0.0078903` (`1.6437318` to `1.6516221`). The strict minimum is epoch 2, the same index as official. Patience then stops after epoch 5. Selected canonical mse/mae are `1.198260877893` / `0.876906379074`. Frozen official seed 2022 is mse `1.206086277961731`, mae `0.8804672360420227`. Residuals are mse `-0.007825400069` and mae `-0.003560856968`. Order-only selection was epoch 3, mse `1.144464620375`, residual about `-0.06162`. The inject removes that selection flip on this seed.

`compare.log` init diff at seed 2021, dropout 0.1, same shape: every non-tokenConv parameter has max abs 0 or about 1e-9, and exceed counts at 1e-5 and 1e-4 are 0. The two tokenConv weights are the O(1) mismatches (`enc` and `dec`, each n=`10752`). The training effect of the 72-tensor inject is the tokenConv Kaiming stream. Seeds 2021 and 2023 were not rerun with this inject. The three-seed MSE cell stays failed. No suite request.

## TokenConv NormalFill16 2026-10-14T05:35Z

PyTorch 2.14 `normal_` on a contiguous float tensor with `numel >= 16` writes `uniform(0, 1)` and transforms blocks of 16 (`u1 = 1 - data[j]`, `u2 = data[j+8]`, eight pairs). A length not divisible by 16 redraws the last 16 uniforms. Production tokenConv is `10752`, which is divisible by 16, so the `next_u32` count matches the old discarded-uniform plus pair path. The old pair formula used `uniform(MIN_POSITIVE, 1)` and `TAU` and is what made tokenConv differ while later uniform layers already matched.

Uncommitted `box_muller16` in `crates/ferrots-nn/src/informer.rs` implements that block formula with f32 libm. `cargo test -p ferrots-nn --test informer_init --locked` passed: seed 2021 production shape, first 16 tokenConv values within `1e-5` of `fixtures/init-torch-dropout0.1.safetensors`, and the next uniform tensors `enc_embedding.temporal_embedding.embed.weight` and `decoder.projection.weight` within `1e-6`. `cargo test -p ferrots-train --test informer_order --locked` passed both tests, including the pinned epoch-0 head `[3230, 4482, 6428, 980]` and the epoch-1 sha256. Toolchain `1.98.0`, `CARGO_TARGET_DIR=target/linux-cuda`. This is attempt 1 of the sampler. It is not yet a CUDA training result.

The native CUDA recheck started at `2026-10-14T05:35:12Z` on physical GPU 0 and exited 0 at about 953 seconds. Log `train1-etth1-s2022-orderfix-nativeinit.log`. Config `etth1-h720-s2022-orderfix-nativeinit.toml`. `weights_replaced=0`. Same data sha256. `candle_gemm_reduced_precision_f32=false`. Output root `train1-orderfix-s2022-nativeinit` does not overwrite the order-only run. The binary mtime is after `informer.rs`. `df -B1 /mnt/data` available bytes at launch were `151913627648` (141.49 GiB). The frozen suite was not modified.

| Epoch | Validation | Canonical MSE | Canonical MAE | improved |
| --- | --- | --- | --- | --- |
| 1 | 1.746859738588 | 1.132512017513 | 0.844602272243 | true |
| 2 | 1.646446855570 | 1.198082455586 | 0.876806982230 | true |
| 3 | 1.654439621133 | 1.153443743153 | 0.844165187283 | false |
| 4 | 1.675943335431 | 1.162413319567 | 0.838907152668 | false |
| 5 | 1.682327984609 | 1.155121853315 | 0.831140567589 | false |

Epoch 3 validation rises by `0.007992765563`. The strict minimum is epoch 2. Patience stops after epoch 5, then `TRAIN1_COMPLETE_WITHOUT_EPOCH`. Selected mse/mae are `1.198082455586` / `0.876806982230`. Against the torch-inject run, epoch-2 mse differs by `-0.000178422307` and mae by `-0.000099396844`. Epoch-1 mse differs by `+0.000486026479`. Frozen official seed 2022 remains mse `1.206086277961731`, mae `0.8804672360420227`, so this seed's residuals are about `-0.008004` and `-0.003660`. Order-only selection was epoch 3, mse `1.144464620375`. The production sampler, without weight injection, restores the epoch-2 selection on this seed. The CUDA binary used `n % 16 != 0`. The committed predicate is `!n.is_multiple_of(16)`, which is the same condition. `informer_init` was re-run after that lint edit and passed. `cargo clippy -p ferrots-nn --features cuda --tests -- -D warnings` passed. Local commit `95ea51d3dcb4daaa3b082a1b8b1ae8a33193e44f`. Seeds 2021 and 2023 had not been rerun when seed 2022 was recorded. This is not a three-seed D4 pass. No suite request. Cement stays blocked. Informer stays `preview`.

## ETTh1 seed 2021 native sampler 2026-10-14T06:14Z

`train1-etth1-s2021-orderfix-nativeinit.log` exited 0. Start `2026-10-14T05:55:51Z`. Physical GPU 0. `weights_replaced=0`. Commit `95ea51d`. `candle_gemm_reduced_precision_f32=false`. data sha256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. Count `10891440`. Wall time about 1125 seconds.

| Epoch | Validation | Canonical MSE | Canonical MAE | improved |
| --- | --- | --- | --- | --- |
| 1 | 1.731799778800 | 1.173806920645 | 0.845759692467 | true |
| 2 | 1.690791139647 | 1.157684050363 | 0.849697703237 | true |
| 3 | 1.685478646247 | 1.223673405142 | 0.884736290825 | true |
| 4 | 1.688963440699 | 1.202628845053 | 0.871350085784 | false |
| 5 | 1.691182549381 | 1.199179028613 | 0.863475912719 | false |
| 6 | 1.690109988326 | 1.191435900438 | 0.862850457359 | false |

The strict minimum is epoch 3. Epoch 4 validation is higher by `0.003484794452`, then two more non-improving epochs stop the run. Frozen official seed 2021 keeps epoch 4 (`stdout` validation `1.6736959`). Official epoch 3 validation `1.6762656` is slightly above epoch 2 `1.6761022`. This Rust run's epoch 3 validation `1.685478646247` is below its epoch 2 `1.690791139647`, so the selected index is 3 rather than 4. The order-only schedule had selected epoch 9, mse `1.179827201136`. Selected canonical mse/mae are now `1.223673405142` / `0.884736290825`. Frozen official seed 2021 is mse `1.2001084089279175`, mae `0.8670483231544495`. Per-seed residuals are mse `+0.023564996214` and mae `+0.017687967671`. These residuals are not a D4 row. Epoch 4's canonical mse `1.202628845053` is not the selected checkpoint. ## ETTh1 seed 2023 and native three-seed mean 2026-10-14T06:34Z

`train1-etth1-s2023-orderfix-nativeinit.log` exited 0. Physical GPU 0. `weights_replaced=0`. Commit `95ea51d`. `candle_gemm_reduced_precision_f32=false`. Same data sha256. Count `10891440`. Wall time about 1125 seconds.

| Epoch | Validation | Canonical MSE | Canonical MAE | improved |
| --- | --- | --- | --- | --- |
| 1 | 1.727693623387 | 1.177507188556 | 0.876750367167 | true |
| 2 | 1.660754398925 | 1.179853608281 | 0.866384216966 | true |
| 3 | 1.651618402560 | 1.188601783234 | 0.859765004458 | true |
| 4 | 1.652007280919 | 1.186855015411 | 0.856412813002 | false |
| 5 | 1.659937336684 | 1.169092468023 | 0.846138480497 | false |
| 6 | 1.653324185955 | 1.187774216467 | 0.853103723908 | false |

The strict minimum is epoch 3, the same index as the frozen official best epoch and the order-only run. Selected mse/mae are `1.188601783234` / `0.859765004458`. Frozen official seed 2023 is mse `1.1902698278427124`, mae `0.862855851650238`. Per-seed residuals are mse `-0.001668044609` and mae `-0.003090847192`.

Native sampler, all three seeds, strict minimum validation, `ddof=0`. This is still not D4. The number is the probe's in-loop `TEST_CANONICAL`, not a suite reload of the best checkpoint.

| Seed | Selected epoch | Official best epoch | MSE | MAE |
| --- | --- | --- | --- | --- |
| 2021 | 3 | 4 | 1.223673405142 | 0.884736290825 |
| 2022 | 2 | 2 | 1.198082455586 | 0.876806982230 |
| 2023 | 3 | 3 | 1.188601783234 | 0.859765004458 |

Population mean is mse `1.203452547987` (std `0.014812901699`) and mae `0.873769425838` (std `0.010418296321`). Absolute differences versus the frozen official means are mse `0.004631043077` and mae `0.003645622222`. Original thresholds are mse `0.023976430098` and mae `0.017402476072`. Both differences sit inside those thresholds, by about `0.019345` and `0.013757`. Rust's mse mean is higher than official. Detail: `runs/p3-development/20261014-d4-cuda/etth1-h720-native-three-seed.json`.

The earlier official in-loop test versus the frozen checkpoint differed by about `0.003`. That gap is smaller than these margins, and it is not a substitute for the suite evaluator. No new formal suite is requested. Cement stays blocked. Informer stays `preview`. P2 CPU/device clauses stay open.

## ETTh2/H96 seed 2023 native sampler 2026-10-14T06:40Z

`train1-etth2-s2023-nativeinit.log` exited 0 in about 228 seconds. Physical GPU 0. `weights_replaced=0`. Commit `95ea51d`. `candle_gemm_reduced_precision_f32=false`. data sha256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. Count `1871520`, the same count as the frozen D4 row.

| Epoch | Validation | Canonical MSE | Canonical MAE | improved |
| --- | --- | --- | --- | --- |
| 1 | 0.791864322644 | 2.638703092454 | 1.300126278436 | true |
| 2 | 0.812648624936 | 3.074831226249 | 1.424055824749 | false |
| 3 | 0.820821904830 | 3.009310080415 | 1.397198384720 | false |
| 4 | 0.838891066324 | 2.872747035798 | 1.357608570472 | false |

The strict minimum is epoch 1. Frozen official ETTh2/H96 also keeps epoch 1 on every seed. Selected mse/mae are `2.638703092454` / `1.300126278436`. The third frozen official seed, which the earlier inject identified as seed 2023, is mse `2.6813955307006836`, mae `1.309546709060669`. Per-seed residuals are mse `-0.042692438247` and mae `-0.009420430625`. The frozen Rust third seed was mae `1.4733237723147488` and mse `3.4096316809082223`. `train1-etth2-s2021-nativeinit.log` and `train1-etth2-s2022-nativeinit.log` also exited 0. Every seed's strict minimum is epoch 1. Seed 2021 validation rises from `0.856946831601` to `1.117530927506` at epoch 2. Seed 2022 rises from `0.869312626331` to `1.171216248581`. Frozen Rust seed 2021 `evaluation.json` mae `1.367123410278071` is the first `d4.json` Rust value, so the arrays are seed order 2021, 2022, 2023.

| Seed | Selected MSE | Selected MAE | Official MSE | Official MAE |
| --- | --- | --- | --- | --- |
| 2021 | 2.765552972227 | 1.313206944004 | 2.9500489234924316 | 1.3599863052368164 |
| 2022 | 3.028325491657 | 1.373608981272 | 3.114254951477051 | 1.3884332180023193 |
| 2023 | 2.638703092454 | 1.300126278436 | 2.6813955307006836 | 1.309546709060669 |

Population mean is mse `2.810860518779` (std `0.162256964557`) and mae `1.328980734571` (std `0.032005585509`). Absolute differences versus the frozen official means are mse `0.104372616444` and mae `0.023674676196`. Original thresholds are mse `0.356841417153` and mae `0.065239596025`. Both sit inside. Rust's MAE mean is lower than official. Detail: `runs/p3-development/20261014-d4-cuda/etth2-h96-native-three-seed.json`. This is in-loop `TEST_CANONICAL`, not D4. The frozen suite's MAE cell remains failed. Count `1871520` on every seed.

## Budget request, not started 2026-10-14T07:05Z

No new suite directory was created. `df -B1 /mnt/data` available bytes were `150859206656` (140.51 GiB). `du -sh` of the frozen suite is `51G`. The frozen executor ran from `2026-10-13T11:56:43Z` to `2026-10-13T23:43:30Z`, about 11 hours 47 minutes wall, and stopped at 96/105 before cement. A second suite of about the same size fits beside the frozen one. Cement's 9 runs and the H60 recompute were not executed, so their duration is unmeasured. The earlier 80 GPU hour and 186 GiB figures remain the old approval estimate. Approval is required before freeze. Informer stays `preview`. Task stays `in_progress`.

## Quality gates 2026-10-14T06:54Z

The fixed Linux CUDA environment ran `cargo fmt --all --check`, workspace CUDA Clippy `-D warnings --locked`, `cargo test --workspace --features cuda --locked`, `uv run --frozen --project tools/reference` unittest discovery, `bash tools/check_private_data.sh`, `git diff --check`, `ferrots doctor --device cuda:0 --training-smoke-test`, and `ferrots doctor --device cuda:9`. `python3 .trellis/scripts/task.py validate .trellis/tasks/09-28-p3-informer` ran after that log and was appended to it.

| Step | Result |
| --- | --- |
| fmt | exit 0 |
| clippy | exit 0, incremental 1.13s |
| workspace tests | exit 0 |
| Python unittest | `Ran 72 tests in 1.052s` / `OK`, exit 0 |
| private-data guard | `ok: no private data tracked`, exit 0 |
| git diff --check | exit 0 |
| cuda:0 smoke | exit 0, `training_smoke_test: passed` |
| cuda:9 | exit 1, required rejection |
| task.py validate | exit 0, both jsonl files 11 entries |

The 51 `test result:` lines are all `ok` and sum to 125 passed, 0 failed, 0 ignored. The sum includes `token_conv_matches_torch_kaiming_normal_and_later_uniform_stays_aligned` and both `informer_order` tests. The pre-suite figure Rust 120 in `implement.md` is that earlier gate. Python printed `START failure` / `END failure exit=1` inside the passing 72-test run; those lines are fixture text. cuda:0 requested `cuda:0` on an RTX 4090, loss `0.04878764`, `code_sha` `95ea51d3dcb4daaa3b082a1b8b1ae8a33193e44f`. cuda:9 printed `DriverError(CUDA_ERROR_INVALID_DEVICE, "invalid device ordinal")` and did not fall back. Validate warned that this evidence file was 35853 bytes, above `context_injection.max_file_bytes` 32768, so injection truncates it. The command still printed `All validations passed`.

After the log, `df -B1 /mnt/data` available bytes were `150033633280` (139.73 GiB). Frozen suite children: `rust` 42G, `official` 8.6G, `cache` 263M, `bin` 233M, `commands` 115M, `entries` 45M, total 51G. GPUs were idle at 18 MiB and 106 MiB. HEAD is `95ea51d`. Branch `dev` is ahead of `origin/dev` by 7. No push. The only dirty paths are this evidence file, `task.json`, and the two untracked result reports. `runs/p3-acceptance/` still contains only `20261013-linux-cuda`.

Scaling the rounded 42G Rust tree by 9/48 is about 8G. That is not a measurement of cement. Cement wall time is unmeasured. The observed 51G and 11 hours 47 minutes cover the 96 ETT entries only. Available space holds another suite of that observed size. The old 80 GPU hour / 186 GiB approval does not cover this rerun. Freeze waits for approval. Informer stays `preview`. Task stays `in_progress`.

## Decision 2026-10-14T07:11Z: REFUSED

The new formal suite on commit `95ea51d` is refused. No approval record exists. It was not frozen and it was not run. `runs/p3-acceptance/` still contains only `20261013-linux-cuda`. `executor.json` `suite_sha256` is still `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. That suite was not modified and was not deleted. Informer stays `preview`. Task stays `in_progress`. Cement stays blocked. P2 CPU/device clauses stay open. WaveForM and P4 stay unstarted.

Continue condition: an explicit approval of the 105-entry matrix on commit `95ea51d`, with a new suite identity, the original D4 formula, and cement only after 32/32. The measured cost basis is the frozen suite at 51G and about 11 hours 47 minutes for 96/105. Cement time and size remain unmeasured. Available space at the last `df` was `150033633280` bytes.

## Decision 2026-10-14T07:13Z: REFUSED

A new decision, separate from the 07:11Z refusal. The 105-entry formal matrix on commit `95ea51d` is refused. The refused matrix is: a new suite identity distinct from `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`, official ETT 48 plus Rust ETT 48, the original D4 formula through 32/32 before cement 9, then the 108-cell report, the H60/seed 2021 recompute, and 105 performance rows, without modifying or deleting the frozen suite. No approval is recorded. The suite was not frozen and was not run. At 2026-10-14T07:13:42Z, `runs/p3-acceptance/` still contains only `20261013-linux-cuda`, and `executor.json` `suite_sha256` is still `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. HEAD is `95ea51d`. Informer stays `preview`. Task stays `in_progress`.

## Decision 2026-10-14T07:15Z: REJECTED

The proposed 105-entry formal suite on commit `95ea51d` is rejected. Approval is absent, so it was not frozen and was not run. At 2026-10-14T07:15:31Z, `runs/p3-acceptance/` still contains only `20261013-linux-cuda`, and `executor.json` `suite_sha256` is still `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. HEAD is `95ea51d`. Informer stays `preview`. Task stays `in_progress`. The continue condition is an explicit approval of that same matrix.

## Decision 2026-10-14T07:28Z: APPROVED and frozen

The user approved the same 105-entry matrix. `runs/p3-acceptance/20261014-linux-cuda` is frozen. Its `suite.json` SHA256 is `c378969bf5195067961bf4e359bcc63f78bae70eb278f74bb1d447e67c5c888f`, distinct from `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. The retained suite was not modified and was not deleted.

`source.dirty` is false. `source.source_sha256` is `ef70dd3488002a847d5354440ce79fc9ba992f6c64be923ec7adb1999bce097e`, the same digest as the `95ea51d` CUDA binary. `source.code_sha` is `cb6589b4635473b2a452a0a1b666d7cec5d99c6f`, the local disk-gate commit whose parent is `95ea51d`. That commit changes only the freeze floor from 186 GiB to 100 GiB and the approval record. Planning files and the two untracked reports were outside the worktree during freeze, so they are not part of `source.dirty`. Upstream remains `4e938a1767106324dd753b2a44832bf870a0252e`. `formal_runs` is 105 and `ddof` is 0. Freeze recorded `free_bytes` `149789216768`. The executor started at `2026-10-14T07:32:02Z`, pid `1162768`, status `running`, in its own session. Initial quality had passed fmt and clippy and was inside workspace tests at launch. Cement has not started. Informer stays `preview`. Task stays `in_progress`.

## Formal D4 2026-10-14T19:30Z: 30/32

The executor ended at `2026-10-14T19:30:29.163639+00:00` with status `failed` and error `ValueError: D4 failed; original bounds and evidence retained; cement not started`. Pid `1162768` is gone. Entries are 96/105. Cement entries are 0. `d4.json` has `complete=true`, `passed=false`, `ddof=0`, 32 rows, 30 passed. The retained suite file SHA256 is still `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. This suite's SHA256 is still `c378969bf5195067961bf4e359bcc63f78bae70eb278f74bb1d447e67c5c888f`.

ETTh1/H720 now passes. MSE difference `0.005583167672257527`, threshold `0.023957387606302897`. MAE difference `0.005599681027770309`, threshold `0.017363394896189373`. Rust means are `1.2034525479874023` and `0.8737694258372389`.

ETTh2/H96 fails both metrics. Rust seeds, in order 2021/2022/2023, are MSE `[2.765552972227383, 3.0283254916565014, 2.6387030924544654]` and MAE `[1.3132069440040126, 1.3736089812717776, 1.300126278435975]`. Those are the same values as the earlier in-loop diagnostic. Official seeds are MSE `[3.0089969635009766, 3.0657005310058594, 2.953869104385376]` and MAE `[1.3746130466461182, 1.3789160251617432, 1.3838393688201904]`. MSE difference `0.19866168085128733`, threshold `0.09131299870572857`. MAE difference `0.05014207897209544`, threshold `0.027582456270853678`. Rust means are lower than official and still fail.

Rust selects epoch 1 on all three seeds. Recorded learning rates are `1e-4`, `1e-4`, `5e-5`, `2.5e-5`, matching the official log's update points. Official seeds 2021 and 2022 also keep epoch 1. Official seed 2023 saves epoch 3: validation goes `0.808270` to `0.813329` to `0.790699`. Its epoch-1 test loss is `2.6193644`; the reloaded checkpoint reports MSE `2.953869104385376`. Rust seed 2023 validation rises after epoch 1 (`0.791864`, `0.812649`, `0.820822`), so it keeps epoch 1 and reports MSE `2.6387030924544654`.

The old suite's official ETTh2/H96 sample was MSE mean `2.9152331352233887`, threshold `0.35684141715262124`, and MAE mean `1.3526554107666016`, threshold `0.06523959602471169`. This suite's official sample is higher and tighter. No D4 bound was changed. Cement was not started. Informer stays `preview`. Task stays `in_progress`. Another formal suite needs a separate approval.

## Approval 2026-10-15 and disk block

The user approved continuing. A new suite was not frozen. `df -B1 /mnt/data` available bytes were `84090802176` (78.32 GiB). The freeze floor is 100 GiB. Retained suites measure 51G and 49G. They were not deleted. GPUs were idle. No training process was running.

Saved checkpoints for Rust ETTh2/H96 seed 2023 were rescored with `p3_d4_probe rank` on cuda:0. Epoch 1 test MSE/MAE are `2.638703092454` / `1.300126278436`, the formal selected values. Epoch 3 test MSE/MAE are `3.009310080415` / `1.397198384720`. Fresh validation MSE still ranks epoch 1 (`0.791864322687`) ahead of epoch 3 (`0.820601562971`). Official validation does the opposite: epoch 3 is `0.790699` and epoch 1 is `0.808271`. Train losses stay within about `0.001` of the official log at each epoch. Replacing only the selected seed-2023 metrics with the epoch-3 checkpoint scores gives MSE difference `0.075126018198` against threshold `0.091312998706`, and MAE difference `0.017784710211` against threshold `0.027582456271`. That replacement is a counterfactual, not a D4 result. The same code was not launched again.

## Checkpoint retention 2026-10-15

The user asked to remove generations kept after every epoch and to keep an off-by-default save-all option. `output.checkpoint_history` defaults to `final`. Serialized config omits that default, so the resolved-config hash is unchanged. `all` retains every generation. With `final`, an improved epoch publishes one bundle and points both best and last at it; other epochs publish last, then delete generation directories not named by the latest best or last reference. Those bundles' recorded best and parent names are kept. Reference JSON files stay append-only.

Existing P3 suites were pruned to that rule. Metrics, official predictions, and the live best/last bundles stay. Historical generation payloads were removed: `20261013-linux-cuda` 185 directories, 28,949,815,771 bytes; `20261014-linux-cuda` 175 directories, 27,384,948,884 bytes. Each suite now has 96 live generations and measures 23.293 GiB and 23.294 GiB. Entry artifact hashes still name the removed files, so a later `verify_artifacts` of those entries fails. The ETTh2/H96 seed 2023 epoch-3 bundle was among the removed generations. `df` available bytes after the prune were `140322549760`. P2 `runs/p2-acceptance/20261012-linux-cuda` was not pruned; its Rust generations measured 10.438 GiB across 565 directories. No new suite was frozen.

## Disk cleanup 2026-10-15

The user asked to remove the remaining cleanup candidates, including runs and `target`. Removed: 449 unreferenced P2 generations, 8,828,645,257 bytes, leaving 116 live best/last generations; `runs/p3-development/20261014-d4-cuda`, `runs/p3-development/20261013-smoke-v1`, `runs/p2-development/batch-probe`; both P3 suite `bin/` and `cache/` directories; and `target/`. Metric JSON, live best/last bundles, official predictions and checkpoints, `data/`, and `tools/reference/.venv` stayed. The repository measured 67,320,334,814 bytes afterward. `df` available bytes were `171149225984`. `target/` must be rebuilt before the next Cargo command. Entry artifact hashes still name removed suite files. No new suite was frozen.

The user then asked to clean the acceptance suites themselves. Removed checkpoint directories, caches, and `.safetensors` / `.npy` / `.pth` / `.pdf` / `.pt` / `.gz` payloads from `20261012-linux-cuda` (9,516,021,433 bytes), `20261013-linux-cuda` (24,239,162,585 bytes) and `20261014-linux-cuda` (24,239,152,820 bytes). `d4.json`, `performance.json`, `entries/` and `recompute.json` stayed. No heavy files of those suffixes remain. The repository measured 9,325,998,683 bytes. `df` available bytes were `229202427904`. No new suite was frozen.

## Validation aggregation 2026-10-15T03:52Z: no rank flip

Scene check at 2026-10-15T03:32:18Z. Branch `dev`, HEAD `d1d73b960fe35d0cd6cc111cb05b7b4715c1a17b`, 9 commits ahead of `origin/dev`. Nothing was pushed, merged, or archived. Dirty tracked files were this evidence file and `task.json`. Untracked `docs/results/p3-informer-ett.md` and `docs/results/p3-informer-performance.md` were left in place. `data/`, `tools/reference/.venv`, and the retained JSON receipts were not moved.

`df -B1 /mnt/data` available bytes were `229202427904`. `du -sb` of the repository was `9325999146`. Both RTX 4090s were idle. `target/` was absent. The old `d4.json` has `passed=false`, `ddof=0`, 29/32. Its failures remain ETTh1/H720 MSE `0.052798151452385644` / `0.023976430098215738`, ETTh1/H720 MAE `0.021247800135460237` / `0.017402476072311403`, and ETTh2/H96 MAE `0.06585556381725577` / `0.06523959602471169`. Suite SHA256 `e5d80b42038f870cf6014392456878cc13d03b642a343b8670d5cd3448dccfec`. The 20261014 `d4.json` has `passed=false`, `ddof=0`, 30/32. The only failures are ETTh2/H96 MSE, official mean `3.0095221996307373`, Rust mean `2.81086051877945`, difference `0.19866168085128733`, threshold `0.09131299870572857`, and MAE, official mean `1.3791228135426838`, Rust mean `1.3289807345705884`, difference `0.05014207897209544`, threshold `0.027582456270853678`. Suite SHA256 `c378969bf5195067961bf4e359bcc63f78bae70eb278f74bb1d447e67c5c888f`.

`spawn_subagent` on this Grok session only accepts `codex:codex-rescue`. `trellis-implement` and `trellis-check` were not dispatched. `trellis-before-dev` is not present on disk. The Informer and training-loop spec pages were read directly, and the measurement ran in this session.

Rebuild environment: `RUSTUP_AUTO_INSTALL=0`, `RUSTUP_TOOLCHAIN=1.98.0`, `RUSTC` and `RUSTDOC` under `/home/lyh/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/`, `CUDA_PATH=/usr/local/cuda-12.9`, `CARGO_TARGET_DIR=target/linux-cuda`. `rustc 1.98.0 (88d9e12ae 2026-08-18)`, cargo `1.98.0`, nvcc `cuda_12.9.r12.9/compiler.35813241_0`. `cargo build --workspace --features cuda --locked` exited 0 at 2026-10-15T03:41:29Z. `target/linux-cuda/debug/ferrots` links `libcuda.so.1`, `libnvrtc.so.12`, `libcublas.so.12`, and `libcudart.so.12` from CUDA 12.9.

`data inspect` for `configs/reference/informer/ett-h2-h96.toml` reported `window_counts` `[8449, 2785, 2785]`. Validation has 2785 windows. With batch size 32 that is 87 full batches and one tail window. The same command for `configs/reference/informer/ett-h1-h720.toml` reported `[7825, 2161, 2161]`. Its validation remainder is 17 windows. Neither inspect trained a model.

The diagnostic config is `runs/p3-development/20261015-etth2-h96-s2023-agg/ett-h2-h96-history-all.toml`. It differs from the committed ETTh2/H96 profile by `checkpoint_history = "all"`. The committed profile still says `final`. Training was one process on `cuda:0` with `CUDA_VISIBLE_DEVICES=0`: seed 2023, the diagnostic cache, output root `runs/p3-development/20261015-etth2-h96-s2023-agg/rust`, run id `rust-ETTh2-h96-s2023-agg`. It started at 2026-10-15T03:45:38Z and exited 0 at 2026-10-15T03:49:58Z. The directory later measured `785445252` bytes and holds five generations.

The new `epochs.jsonl` matches `runs/p3-acceptance/20261014-linux-cuda/rust/rust-ETTh2-h96-s2023-20261014-085730-447449/epochs.jsonl` on epoch, learning rate, train loss, validation loss, valid labels, and optimizer steps for all four epochs. Each compared value is equal, including validation losses `0.7918643226437643`, `0.8126486249364236`, `0.8208219048303652`, and `0.8388910663243062`. `best_epoch` is 1.

`p3_d4_probe agg` rescored those checkpoints on `cuda:0`. The forward id for epoch `e` starts at `(e-1) * 88`, which is the continued validation counter in `validation_loss`. The label-weighted rescore matches each stored validation loss at 12 printed decimals. The equal-weight value is the mean of the per-batch f32 means (`batch_sum / valid_count`), the same reduction as `np.average` in the pinned `exp_long_term_forecasting.py` `vali`. Results:

| epoch | weighted | equal | tail mean | equal minus weighted |
| --- | --- | --- | --- | --- |
| 1 | 0.791864322644 | 0.801132040268 | 1.633731842041 | 0.009267717624 |
| 2 | 0.812648624936 | 0.822054057975 | 1.667026281357 | 0.009405433038 |
| 3 | 0.820821904830 | 0.828988290307 | 1.562645673752 | 0.008166385477 |
| 4 | 0.838891066324 | 0.843844870444 | 1.288888096809 | 0.004953804120 |

`ARGMIN` is epoch 1 for both reductions. `rank_flip=false`. Epoch 3 is higher than epoch 1 under both: equal-weight gap `0.027856250039`, label-weighted gap `0.028957582186`. The tail window raises every epoch by about `0.005` to `0.009` and does not change the selected epoch. Official seed 2023 stdout still saves epoch 3, with validation `0.8082705`, `0.8133293`, `0.7906989`. This reduction does not produce that order.

The earlier checkpoint rescore `0.820601562971` came from `p3_d4_probe rank`, which restarts the validation forward id at 0. It is not the continued-id value. Today's continued-id rescore reproduces the logged `0.8208219048303652`. `rank` was not run again.

Production early stopping was not changed. No aggregator regression was added. No formal suite was frozen, and no new 105-entry matrix was requested. Cement stays 0/9. D4 stays 30/32. Informer stays `preview`. The task stays `in_progress`.

ETTh1/H720 was not retrained. Its saved label-weighted margins to the next epoch are `0.003484794452` (seed 2021), `0.007992765563` (seed 2022), and `0.000388878359` (seed 2023). The 17-window tail means an aggregator change is not known to preserve those passed selections. Pinned `data_factory.py` also shuffles the validation loader. That interaction was not measured.

`cargo clippy -p ferrots-train --features cuda --locked --example p3_d4_probe -- -D warnings` exited 0 after `rustfmt --edition 2024`. After the run, `df` available bytes were `225202212864`. Repository `du -sb` was `13327193450`, of which `target/linux-cuda` was `3215742505`. P2 CPU/device clauses stay open. WaveForM and P4 were not started.

## Validation shuffle bound 2026-10-15T15:52Z: cannot flip

The fixed-order aggregation left one open reduction hypothesis: pinned `data_factory.py` shuffles validation, `drop_last` is false, and ETTh2/H96 validation has 2785 windows. With batch size 32 the tail batch contains one window. Every window in this replay has 672 valid labels, so the batch-equal loss is determined by which window lands in that tail:

`E(t) = (2785 * W + 31 * t) / 2816`.

`p3_d4_probe bounds` replays the five retained generations of `runs/p3-development/20261015-etth2-h96-s2023-agg/rust/rust-ETTh2-h96-s2023-agg` on `cuda:0` (`CUDA_VISIBLE_DEVICES=0`, `FERROTS_PROBE_DEVICE=0`). It started at 2026-10-15T15:51:58Z and exited 0 at 2026-10-15T15:52:11Z. The log is `runs/p3-development/20261015-etth2-h96-s2023-agg/bounds.log`. The example binary was rebuilt first; an earlier binary from 2026-10-15T03:52Z did not contain `bounds`. `cargo clippy -p ferrots-train --features cuda --locked --example p3_d4_probe -- -D warnings` reported no issues. `rustfmt --edition 2024` was applied to that example only. The marker device prints `CudaDevice(DeviceId(1))` under `CUDA_VISIBLE_DEVICES=0`. The label-weighted rescore still matches each stored validation loss at 12 decimals, including epoch 3 `0.820821904830`.

| epoch | weighted | mean min | mean max | equal at min tail | equal at max tail |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.791864322644 | 0.140120863914 | 2.460199594498 | 0.784689589966 | 0.810230229401 |
| 2 | 0.812648624936 | 0.120808213949 | 2.493139982224 | 0.805032484048 | 0.831148352236 |
| 3 | 0.820821904830 | 0.117312990129 | 2.321647167206 | 0.813077310954 | 0.837343773841 |
| 4 | 0.838891066324 | 0.108601957560 | 2.021704196930 | 0.830851662073 | 0.851912091555 |

Counts are uniform (`count_min=count_max=672`) and `tail_windows=1` on every epoch. The closed form applied to the actual tail matches the measured batch-equal loss within `2.810e-9`. Per-window sums disagree with the batch sum by at most `7.812e-3` on a batch of about 17,000, which moves a window mean by about `1e-5` and the equal-weight loss by about `1e-7`.

`SHUFFLE e3_at_min=0.813077310954 e1_at_max=0.810230229401 e3_at_min_minus_e1_at_max=0.002847081553 can_flip=false`. Epoch 3's lowest batch-equal loss on these predictions is still higher than epoch 1's highest. No choice of the singleton tail makes epoch 3 better than epoch 1. The predictions stay tied to the fixed-order forward ids. This pass does not assign those windows to new ProbSparse seeds. The earlier id restart changed epoch 3's label-weighted loss by about `2.2e-4`, from `0.820821904830` to `0.820601562971`. That shift is smaller than the `0.002847081553` gap.

Production early stopping was not changed. No formal suite was frozen. Cement stays 0/9. D4 stays 30/32. Informer stays `preview`. The task stays `in_progress`. The parent task note that said P3 remained planning was corrected to say P3 is `in_progress`; WaveForM and P4 stay planning. P2 CPU/device clauses stay open.

## Official validation draws 2026-10-15T16:01Z

The shuffle bound above fixes Rust's predictions. The pinned official `vali` still shuffles and returns `np.average` of per-batch means, while ProbSparse calls `torch.randint` in eval. Two diagnostic official runs used the pinned runner, seed 2023, `CUDA_VISIBLE_DEVICES=0`, and the same ETTh2/H96 arguments as the formal cell. The returned early-stop value stayed `np.average`. A process-local wrapper also printed the label-weighted mean. It did not edit the TSLib checkout. Launcher: `runs/p3-development/20261015-etth2-h96-s2023-dual/launch.py`.

The single pass started at 2026-10-15T15:56:20Z and exited 0 at 2026-10-15T15:56:41Z. Output: `runs/p3-development/20261015-etth2-h96-s2023-dual/official`. Train losses were `0.3903449`, `0.2745368`, `0.2287489`, `0.2089454`. Validation equal/weighted pairs were epoch 1 `0.8260341300` / `0.8099655535`, epoch 3 `0.8398931016` / `0.8454200760`. Both reductions selected epoch 1. Early stopping fired after epoch 4. Final test MSE/MAE were `2.705460786819458` / `1.3146029710769653`. Element-mean versus `MSELoss` differed by at most `5.109e-08`. This pass did not reproduce the 2026-10-14 validation drop.

The repeated pass started at 2026-10-15T16:00:13Z and exited 0 at 2026-10-15T16:01:00Z. Output: `runs/p3-development/20261015-etth2-h96-s2023-dual/official-repeat`. `FERROTS_VAL_REPEATS=7` scored seven extra validation passes after the official pass, then restored CPU, CUDA, NumPy, and Python RNG state. Early stopping consumed only `repeat=0`. Train losses were `0.3903563`, `0.2750086`, `0.2286424`, `0.2088432`, `0.1964023`, `0.1895271`, each within about `0.001` of the 2026-10-14 official log. Eight validation draws:

| epoch | equal mean | equal min | equal max | equal std | weighted mean | weighted std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8046204977 | 0.7967391009 | 0.8203627196 | 0.0097809801 | 0.8032950376 | 0.0000926581 |
| 2 | 0.8305838999 | 0.8236166596 | 0.8436482555 | 0.0074726409 | 0.8305299563 | 0.0001080077 |
| 3 | 0.8155505819 | 0.8110839511 | 0.8193004544 | 0.0027268024 | 0.8176201831 | 0.0001449056 |
| 4 | 0.8669567811 | 0.8577220453 | 0.8748113154 | 0.0062886224 | 0.8655887448 | 0.0001503537 |
| 5 | 0.8795746244 | 0.8715795163 | 0.8872889389 | 0.0051996005 | 0.8773921279 | 0.0001894268 |
| 6 | 0.8901618642 | 0.8832429471 | 0.8981788206 | 0.0059743947 | 0.8900004003 | 0.0002513255 |

Equal std is the population standard deviation of the eight draws. Weighted std stays below `0.0003`. The closed form matched `repeat=0`: epoch 1 predicted `0.819076790143` against printed `0.8190768043`; epoch 3 predicted `0.812870188344` against printed `0.8128702034`.

`repeat=0` was epoch 1 `0.8190768043` and epoch 3 `0.8128702034`, so early stopping kept epoch 3 and stopped after epoch 6. Final test MSE/MAE were `2.943876266479492` / `1.3797394037246704`. The 2026-10-14 official seed-2023 metrics were MSE `2.953869104385376` and MAE `1.3838393688201904`. Across the 64 pairs of epoch-1 and epoch-3 equal draws, epoch 3 was lower in 15. Epoch 3's best tail on these weights gives equal loss about `0.809684` to `0.810159`. Epoch 1's worst tail gives about `0.822228` to `0.822722`. A tail draw can therefore select epoch 3. The label-weighted draws stay near `0.803295` and `0.817620`, so that reduction keeps epoch 1.

The retained Rust checkpoints do not have this overlap. Their epoch-3 best equal loss is `0.813077310954` and their epoch-1 worst is `0.810230229401`. Rust epoch-1 label-weighted loss is `0.791864322644`, which is `0.011431` below this official weighted mean `0.8032950376`. Rust epoch-3 label-weighted loss is `0.820821904830`, which is `0.003202` above this official weighted mean `0.8176201831`. Official ProbSparse noise in the weighted mean is std `0.000093` at epoch 1. The 2026-10-14 official epoch-3 validation value `0.7906989` is below this run's epoch-3 equal floor of about `0.8097`, so that recorded value is not a shuffle of these weights.

Production early stopping was not changed. No formal suite was frozen. Cement stays 0/9. D4 stays 30/32. Informer stays `preview`. The task stays `in_progress`.

## Same weights, then initialization and batch 0 2026-10-15T17:01Z

The epoch-1 weight gap follows the parameters, not the validator. `runs/p3-development/20261015-etth2-h96-s2023-cross/cross_score.py` scored the retained Rust epoch-1 checkpoint and the official epoch-1 `checkpoint.pth` from `dual/official` on one official shuffled validation pass. The Rust probe scored the transplanted official weights. Label-weighted losses:

| weights | Rust weighted | official weighted | official equal |
| --- | ---: | ---: | ---: |
| Rust epoch 1 | 0.791864322644 | 0.7920056986 | 0.8041954670 |
| official epoch 1 | 0.810135276438 | 0.8095799099 | 0.8231712458 |

Same-weight scorer gaps are `0.000141` and `0.000555`. The Rust log is `runs/p3-development/20261015-etth2-h96-s2023-cross/rust-on-official.log`. Swapping only the BatchNorm buffers on a second official shuffle moved the weighted loss by about `0.004`: official weights with Rust BN `0.8131110060`, Rust weights with official BN `0.7875154121`. That does not close the `0.018` gap. The train scaler was already within `1.3e-13`.

Seed-2023 initialization was saved on CPU after the same pre-seed imports as `run.py`, with the model import after the seed. Torch `2.14.0+cu130`. The import left the RNG unchanged. `p3_d4_probe init-gap` on `cuda:0` compared 76 float tensors. Log: `runs/p3-development/20261015-etth2-h96-s2023-cross/init-gap.log`.

| tensor | max abs |
| --- | ---: |
| both positional embeddings | 4.88281075e-4 |
| both tokenConv weights | 1.19209290e-7 |
| next six, including attention projections | 3.72529030e-9 |

`INIT_SUMMARY` worst is `4.88281075e-4`. Adam parameters match. The positional embedding is a buffer and is not an Adam parameter. The BatchNorm counter was 0 on both sides and was not part of the float comparison.

The epoch-0 train permutation also matches. The real `DataLoader` iterator draws its base seed before `RandomSampler` runs, because `RandomSampler.__iter__` is a generator and does not draw until the first batch. Rust discards that first int64 and permutes with the second. For ETTh2/H96 seed 2023 the shared order has length 8449, sha256 `a930d25490e77f5d938398f40b26ac3fcf562e3db04dbc28ccf31dc1e8b2ce41`, and head `[8034, 6390, 3066, 4116]`. Iterating `batch_sampler` directly uses the first draw and does not match; that path is not what `exp.train` runs. The ETTh1/H720 seed-2022 loader head is `[3230, 4482, 6428, 980]`, sha256 `63a5adfa3e9174942f9bc3c58ec82e69261aa877845f041d857cb8e7e5fdd3d7`, the same value pinned by `etth1_h720_seed2022_epoch1_order_matches_official_loader`.

The first training batch of that shared order was scored on `cuda:0` without an optimizer step. History, time, and target means agree to about `1e-9`. The supervised count is 21504. Logs: `official-batch0.log`, `rust-batch0.log`, `batch0-modes.log`, and `batch0-same-samples.log`.

| mode | ProbSparse indices | official loss | Rust loss | Rust minus official |
| --- | --- | ---: | ---: | ---: |
| train, dropout 0.1 | each implementation | 1.1963635683 | 1.1982093084 | +0.0018457401 |
| train, dropout 0.1 | official indices | 1.1963635683 | 1.1912341345 | -0.0051294338 |
| eval | each implementation | 1.1834262609 | 1.1831683204 | -0.0002579405 |
| eval | official indices | 1.1834262609 | 1.1834308079 | +0.0000045470 |
| train, dropout 0 | each implementation | 1.1976703405 | 1.2040895735 | +0.0064192330 |
| train, dropout 0 | official indices | 1.1976703405 | 1.1976844243 | +0.0000140838 |

With the official indices, eval agrees to `4.5e-6` and dropout-0 training agrees to `1.4e-5`. BatchNorm in train mode is inside that second figure. The remaining train gap at dropout 0.1 is the mask stream: Rust uses ChaCha8 keyed by seed, forward id, and site, while the official CUDA dropout uses the PyTorch generator. Production dropout and early stopping were not changed.

D4 stays 30/32. Cement stays 0/9. No formal suite was frozen. Informer stays `preview`. The task stays `in_progress`. The next measurement is a batch-0 dropout mask that matches this official CUDA generator, checked by the same train loss, before another full epoch or suite. ETTh1/H720 was not retrained. P2 CPU/device clauses stay open. WaveForM and P4 were not started.

## CUDA dropout mask on batch 0 2026-10-16T01:45Z

Informer CUDA training dropout now follows PyTorch 2.14.0+cu130 fused dropout, torch commit `08187d9e0fba026dc8217405802ab5381dc88d90`, file `aten/src/ATen/native/cuda/Dropout.cu`. The shared `dropout()` in `crates/ferrots-nn/src/random.rs` is still ChaCha8. P0 and iTransformer were not switched. CPU Informer still uses that ChaCha8 path.

The CUDA stream is one Philox4x32-10 generator per Informer, seed equal to the model seed, offset starting at 0. That is the state after `torch.manual_seed` and before the first CUDA random call. Eval and p=0 do not advance the offset. Each train call advances it by the host formula: block 256, UNROLL 4, grid capped by `multiProcessorCount * (maxThreadsPerMultiProcessor / 256)`. This RTX 4090 reports 128 multiprocessors and 1536 threads per multiprocessor, so the cap is 768 blocks. A kept value is `curand_uniform < (float)(1 - p)`, then multiplied by `(float)(1 / (float)(1 - p))`.

Storage order matters. Attention residual dropout is contiguous `[B, L, C]` on both sides. Embedding dropout and both FFN dropouts are applied by PyTorch to contiguous `[B, C, L]` storage; the Rust tensors are `[B, L, C]`. The mask is generated in the PyTorch storage order and then placed into the Rust order. The first four official sites, including the non-contiguous embedding, matched that placement on the recorded 16-bit heads.

The same seed-2023 batch 0 was scored again on `cuda:0` with official ProbSparse indices, head `[8034, 6390, 3066, 4116]`, count 21504, and `candle_gemm_reduced_precision_f32=false`. No optimizer step. Probe: `target/linux-cuda/debug/examples/p3_d4_probe batch0`.

| mode | official loss | Rust loss | Rust minus official |
| --- | ---: | ---: | ---: |
| train, dropout 0.1, official indices | 1.1963635683 | 1.1963678088 | +0.0000042405 |
| eval, official indices | 1.1834262609 | 1.1834308079 | +0.0000045470 |
| train, dropout 0, official indices | 1.1976703405 | 1.1976844243 | +0.0000140838 |

The dropout-0.1 gap moved from `-0.0051294338` to `+0.0000042405`. Eval and dropout 0 repeated the previous figures. `cargo test -p ferrots-nn --features cuda --locked --lib rtx4090_masks_match_torch_2_14_cuda_dropout` passed. After the storage-order fix, `--test dropout --test p3_informer` passed 7 tests. `random.rs` was not edited.

The Philox offset lives on the Informer and is not in `RngState`. A new model starts at 0, so one uninterrupted training process follows the official stream. Loading a checkpoint does not restore the offset. Production early stopping was not changed. No full epoch was started. D4 stays 30/32. Cement stays 0/9. No formal suite was frozen. Informer stays `preview`. Step 7 stays open. The next measurement is one fresh ETTh2/H96 seed-2023 epoch compared on weights, before any suite request. ETTh1/H720 was not retrained and must stay inside its saved margins before a suite. P2 CPU/device clauses stay open. WaveForM and P4 were not started.

## One Philox epoch against official epoch-1 weights 2026-10-16T02:33Z

One new process trained ETTh2/H96, seed 2023, dropout 0.1, for one epoch. It started at 2026-10-16T02:31:14Z and exited 0 at 2026-10-16T02:32:53Z. Physical GPU 0, `CUDA_VISIBLE_DEVICES=0`. Binary `target/linux-cuda/debug/ferrots`, rebuilt after the Philox storage-order edit. Config `runs/p3-development/20261015-etth2-h96-s2023-cross/ett-h2-h96-seed2023.toml` with `--seed 2023 --max-epochs 1`. Cache hit on the 20261015 agg cache. `data_sha256` is `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. Output is `runs/p3-development/20261016-etth2-h96-s2023-philox-e1/rust-ETTh2-h96-s2023-philox-e1`. Epoch 1 used 8449 windows, 265 optimizer steps, learning rate `1e-4`. Train seconds `80.380`. Train loss `0.390916163064493`. Fixed-order label-weighted validation loss `0.7786382171142975`. The selected bundle is `checkpoints/generation-1-321299-1792117956692140779`. The Philox offset was not written into the checkpoint. Early stopping was not changed.

The weight reference is the epoch-1 `checkpoint.pth` from `runs/p3-development/20261015-etth2-h96-s2023-dual/official`. That run saved epoch 1 and logged train loss `0.3903449`, validation equal `0.8260341300`, validation weighted `0.8099655535`. The comparison script is `compare_weights.py` in the new run directory. All 77 tensors match by name. `candle_gemm_reduced_precision_f32` printed `false` on the follow-up probe. Installed torch `2.14.0+cu130` defaults are matmul `allow_tf32=False` and cuDNN `allow_tf32=True`. The dual launcher does not set either flag.

| weights | parameter element-mean abs | parameter max abs | tensors with max abs > 1e-3 | tensors with max abs > 1e-2 |
| --- | ---: | ---: | ---: | ---: |
| Philox epoch 1 | 0.000351000182 | 0.020350005478 | 30 / 72 | 7 / 72 |
| ChaCha epoch 1 | 0.000533508957 | 0.027648491785 | 41 / 72 | 10 / 72 |

Parameter count is 72 and numel is 11328007. Both maxima are `decoder.layers.0.cross_attention.query_projection.weight`. Philox has the smaller max abs on 70 of 77 tensors. The three tensors with an equal max abs are the two positional embeddings and `encoder.conv_layers.0.norm.num_batches_tracked`. Four tensors have a larger Philox max. Three of those are attention biases below `2e-6`. `decoder.layers.0.self_attention.value_projection.bias` is `5.58719039e-3` against the ChaCha max `4.12674621e-3`. Positional-embedding max abs stays `4.88281075e-4` on both runs. Philox BatchNorm `running_var` max abs is `6.29792213e-2` and `running_mean` max abs is `4.03764397e-2`. The ChaCha bundle's corresponding maxima are `1.43080235e-1` and `1.89729780e-1`.

Against the official logged train loss `0.3903449`, the Philox train loss differs by `+0.000571263064`. The retained ChaCha epoch-1 train loss `0.391403854729` differs by `+0.001058954729`. The logged validation numbers use different reductions and were not rescored on one validator in this pass.

The same Philox binary then scored batch 0 again without an optimizer step and without injected ProbSparse indices. Head `[8034, 6390, 3066, 4116]`, count 21504, train mode, dropout 0.1. Rust loss `1.1974469139`. Official loss `1.1963635683`. Difference `+0.0010833456`. The injected-index Philox difference remains `+0.0000042405`. The earlier ChaCha own-index difference was `+0.0018457401`. Log: `runs/p3-development/20261016-etth2-h96-s2023-philox-e1/batch0-own-indices.log`.

The epoch weight gap is smaller than the ChaCha gap and is not a match. The first batch still misses by about `1e-3` when each side draws its own ProbSparse indices. Index values remain on the isolated stream described by `InformerSamplingSchedule`; this pass did not compare those tensors with `official-batch0-randint.safetensors`. D4 stays 30/32. Cement stays 0/9. No formal suite was frozen. Informer stays `preview`. Step 7 stays open. The next measurement is that first-batch index-tensor comparison, before another epoch or a suite. ETTh1/H720 was not retrained. P2 CPU/device clauses stay open. WaveForM and P4 were not started.

## Training ProbSparse indices follow CPU torch.randint 2026-10-16T03:22Z

Informer training now draws each train batch's ProbSparse indices from the frozen post-init MT19937 snapshot. `ReferenceRng::torch_randint` follows PyTorch 2.14 CPU `torch.randint`: `next_u32() % high` when `high < 2^28`, and `next_u64() % high` otherwise. Batch 0 is two int64 `seed_draw` calls, then the four sites in forward order. A forward that does not receive samples still uses the isolated index stream. Shared ChaCha dropout and the Philox mask were not changed. The live index cursor is not written into `RngState` or the checkpoint. The Philox offset is still not persisted.

`cargo test -p ferrots-core --test reference_rng torch_randint` passed 2 tests: `torch_randint_values_match_pytorch_2_14_cpu` and `torch_randint_word_count_and_empty_sites_keep_five_draws`. `cargo test -p ferrots-train --features cuda --test informer_order` passed 3 tests. `etth2_h96_seed2023_batch0_indices_match_torch_randint` pins the four batch-0 heads: encoder layer 0 `[25, 81, 79, 1, 2, 58, 75, 77]`, encoder layer 1 `[25, 4, 28, 41, 30, 38, 11, 20]`, decoder self-attention `[102, 22, 137, 102, 88, 89, 117, 54]`, and decoder cross-attention `[40, 26, 32, 28, 30, 24, 35, 31]`. The ETTh1/H720 loader-order hashes were unchanged.

The probe binary was rebuilt after that edit. `target/linux-cuda/debug/examples/p3_d4_probe batch0` ran at 2026-10-16T03:22:35Z without `FERROTS_BATCH0_SAMPLES`. Config `runs/p3-development/20261015-etth2-h96-s2023-cross/ett-h2-h96-seed2023.toml`. Physical GPU 0, `CUDA_VISIBLE_DEVICES=0`, `FERROTS_PROBE_DEVICE=0`. The marker was `Cuda(CudaDevice(DeviceId(1)))` and `candle_gemm_reduced_precision_f32=false`. The probe printed `BATCH0_INDICES source=torch_randint sites=4`. Loader head `[8034, 6390, 3066, 4116]`, count 21504, train mode, dropout 0.1. No optimizer step. Exit 0.

| batch 0, dropout 0.1 | Rust loss | official loss | Rust minus official |
| --- | ---: | ---: | ---: |
| injected official indices | 1.1963678088 | 1.1963635683 | +0.0000042405 |
| torch.randint indices, this probe | 1.1963678088 | 1.1963635683 | +0.0000042405 |
| previous isolated indices | 1.1974469139 | 1.1963635683 | +0.0010833456 |

The uninjected loss matches the injected Philox loss at the printed 10 decimal places. Log: `runs/p3-development/20261016-etth2-h96-s2023-philox-e1/batch0-torch-randint.log`. This covers epoch 0 batch 0. Later train batches continue the same MT19937 cursor and were not compared tensor-by-tensor. D4 stays 30/32. Cement stays 0/9. No formal suite was frozen. Informer stays `preview`. Step 7 stays open. The next measurement is one fresh ETTh2/H96 seed-2023 epoch on this index path, compared with the official epoch-1 checkpoint, before any suite. ETTh1/H720 was not retrained. Production early stopping was not changed. P2 CPU/device clauses stay open. WaveForM and P4 were not started.

## One torch.randint epoch against official epoch-1 weights 2026-10-16T03:28Z

One new process trained ETTh2/H96, seed 2023, dropout 0.1, for one epoch on the CPU `torch.randint` index path. It started at 2026-10-16T03:27:00Z and exited 0 at 2026-10-16T03:28:23Z. Physical GPU 0, `CUDA_VISIBLE_DEVICES=0`. Binary `target/linux-cuda/debug/ferrots`, rebuilt after `session.rs`. Config `runs/p3-development/20261015-etth2-h96-s2023-cross/ett-h2-h96-seed2023.toml` with `--seed 2023 --max-epochs 1`. The committed `configs/reference/informer/ett-h2-h96.toml` was not edited. Cache hit. `data_sha256` is `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. Output is `runs/p3-development/20261016-etth2-h96-s2023-randint-e1/rust-ETTh2-h96-s2023-randint-e1`. Epoch 1 used 8449 windows, 265 optimizer steps, learning rate `1e-4`. Train seconds `67.650`. Train loss `0.39110183392516285`. Fixed-order label-weighted validation loss `0.7983538953506041`. The selected bundle is `checkpoints/generation-1-475257-1792121289646029431`. Its `rng-state.json` has keys `format_version`, `sampling_policy`, `seed`, `post_init`, `next_forward_id`, and `dropout_site_version`. The Philox offset and the index cursor are absent. Early stopping was not changed.

The weight reference is the same epoch-1 `checkpoint.pth` from `runs/p3-development/20261015-etth2-h96-s2023-dual/official`. That run logged train loss `0.3903449`. The comparison script is `compare_weights.py` in the new run directory. All 77 tensors match by name.

| weights | parameter element-mean abs | parameter max abs | tensors with max abs > 1e-3 | tensors with max abs > 1e-2 |
| --- | ---: | ---: | ---: | ---: |
| torch.randint epoch 1 | 0.000171477345 | 0.0209988244 | 21 / 72 | 2 / 72 |
| isolated-index Philox epoch 1 | 0.000351000182 | 0.0203500055 | 30 / 72 | 7 / 72 |
| ChaCha epoch 1 | 0.000533508957 | 0.0276484918 | 41 / 72 | 10 / 72 |

Parameter count is 72 and numel is 11328007. The torch.randint maximum is `decoder.layers.0.cross_attention.query_projection.weight`. Against the isolated-index Philox epoch, 69 of 77 tensors have a smaller max abs, 3 are equal, and 5 are larger. The only increase above `1e-4` is that same query projection, from `0.0203500055` to `0.0209988244`. The other four increases are at most `9.30717215e-6`. Positional-embedding max abs stays `4.88281075e-4`.

Against the official logged train loss `0.3903449`, this train loss differs by `+0.000756933925`. The isolated-index Philox train loss differed by `+0.000571263064`. The logged official validation numbers remain equal `0.8260341300` and weighted `0.8099655535`. Those reductions differ from the Rust fixed-order label-weighted loss, and this pass did not rescore them on one validator.

The epoch does not match the official epoch-1 weights. D4 stays 30/32. Cement stays 0/9. No formal suite was frozen. Informer stays `preview`. Step 7 stays open. Another epoch is not justified until a measurement names the remaining per-step divergence. Later-batch index tensors were not compared with PyTorch. ETTh1/H720 was not retrained. P2 CPU/device clauses stay open. WaveForM and P4 were not started.
