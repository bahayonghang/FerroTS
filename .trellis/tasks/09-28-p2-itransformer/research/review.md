# P2 inline review checkpoint

Date: 2026-09-29. Status: formal acceptance in progress; final sign-off pending. Review stayed in the main session.

## Verified identity and existing gates

The current runtime source digest matches the frozen suite: d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81. Suite: runs/p2-acceptance/20260929-cuda-v2. The quality validator reread the recorded input hashes and command logs: Rust 102 passed, Python 28 passed, zero failures and zero ignored tests. Preflight, data, numerical and fresh-process recovery gates remain passed. This check read evidence and did not repeat training tests.

The official CUDA ETT stage completed 48/48 with exit 0. Rust ETT training is active. D4, cement, independent recomputation and final quality remain pending. The original CPU formal control is unexecuted under the later CUDA-first instruction.

## Inspected contracts

| Requirement | Source anchors | Review result |
| --- | --- | --- |
| R1/R2 and AC1-AC3 | crates/ferrots-nn/src/itransformer.rs; crates/ferrots-nn/src/train_ops.rs; crates/ferrots-train/tests/p2_numerics.rs | Official construction order, Conv1d mapping, detached window mean, differentiable variance, post-norm encoder and token slicing agree with the implemented fixture contract. The approved shared-gradient Adam gate retains all parameters and moments. Independent-backward differences remain separate diagnostics. |
| R3/R4 | crates/ferrots-core/src/config.rs:261 and :556; crates/ferrots-nn/src/itransformer.rs:231 | Strict model fields and dimension/future-known checks precede construction. Weight loading validates the complete set before assignment. The iTransformer-specific future-known rejection runs before the shared baseline input check. |
| R5/R6 and AC4/AC5 | crates/ferrots-train/src/session.rs:246 and :429; crates/ferrots-train/src/checkpoint/training.rs:299; crates/ferrots-cli/src/pipeline.rs:272 | Cursor, phase, permutation, forward/update counters, scheduler, optimizer groups and environment bindings are checked. CLI resume loads the persisted configuration without user overrides. A continuation preserves its parent and the verified historical best. |
| R7 | crates/ferrots-nn/src/registry.rs:26; crates/ferrots-nn/src/model.rs:133; crates/ferrots-cli/src/pipeline.rs:136 | Static dispatch and the CLI share model descriptors. Build device support is separate from measured reproduction status. iTransformer remains pending while formal acceptance is incomplete. |
| R8 and AC6-AC8 | tools/validation/p2_acceptance.py; tools/reference/p2_observer.py; tools/validation/gpu_observer.py | The runner preserves fixed seeds and profiles, immutable attempts, exact matrix keys, full test coverage, source/binary/config/data bindings and normal termination. D4 uses population standard deviation. Cement prerequisites prevent advancement before D4 passes. Device-wide and allocator memory measurements are identified separately. Formal result acceptance is pending. |

## Cross-layer flow

Configuration supplies the typed model variant and data bindings. Dataset and DeviceBatch keep supervision outside ModelInput. ForecastModel performs the model dispatch. TrainingSession owns every training-forward identity, optimizer boundary and validation transition. The checkpoint coordinator persists one coherent snapshot and publishes references only after generation validation. CLI resume checks provenance and writes a separate continuation. The acceptance runner validates the produced artifacts rather than substituting expected numeric results.

No additional blocking code defect was identified in these inspected contracts. Existing numerical, corruption, publication-fault and process-recovery tests remain in the suite. CUDA workspace tests retain their existing host assertions; no separate CPU-only suite or CPU training was scheduled in this continuation.

## Recorded corrections and limits

- The pinned reference uses np.Inf. The wrapper restores the equal np.inf constant in the reference subprocess and records the compatibility action. The original failed run remains intact. The replacement suite reran prerequisite gates.
- FP64 state uses serde_json float_roundtrip. Resume evidence and the historical v1 inference regression remain identified separately in implementation-evidence.md.
- The two historical test-process access violations retain the status cause not established. Successful reruns do not establish a root cause.
- The new backend itransformer-training.md contract and its index/context links passed document checks and task.py validate. No runtime source, binary, frozen configuration or official checkout changed during this documentation work.

## Required before final sign-off

- Complete Rust ETT 48/48 and all 32 D4 comparisons.
- Complete cement 9/9 with all 18 variables and the complete independent prediction recomputation.
- Run final CUDA quality and the final acceptance command; inspect the generated metric and performance reports.
- Update the model card and acceptance report using actual final receipts. Preserve source/binary identity if a capability-status change affects runtime files.
- Recheck scope and private-data handling before any authorized local delivery. No commit, push or archive has occurred.

## Main-session continuation review: 2026-09-29 20:44 America/Chicago

The review remains inline. The existing task was selected by its explicit path because the current-task pointer is absent. task.json remains in_progress; task.py start was not repeated.

Evidence: runs/p2-development/handover-verification-20260929-204227-752639.json; runs/p2-development/process-takeover-20260929-204416-5795602.json.

Reviewed the formal runner through suite_context, verify_rust, verify_official, require_gate, cement_entries, recompute_gate, validate_quality and final_gate. The existing sequential script verifies the predecessor creation time and a complete D4 32/32 result before dependent commands. The suite validator checked current source, binary, tool/config/data fingerprints, official checkout and P1 baseline fingerprints.

R1-R6 and AC1-AC5 retain the prior implementation review and revalidated numerical/recovery/quality evidence. R7 remains pending in the runtime registry. R8, AC6-AC8 and X2 remain incomplete while Rust ETT and cement are incomplete. X1 uses ignored private artifacts and statistics-only reports; X3/X5/X6 remain bound to the frozen suite identities. X4 retains the itransformer reproduction model name.

At the checked snapshot, official ETT is 48/48 and Rust ETT is 9/48. Six comparisons are diagnostic only. No additional code defect was established during the runner and process checks. The model card now lists the frozen profiles referenced by the ETT report.

Final CUDA quality, full report reconciliation and final inline trellis-check cannot be signed off until all formal gates complete. The current update changes only P2 documentation and status. It does not alter or replace original failure receipts.

Report verification passed at 20:47 America/Chicago: all nine updated files, 32 ETT rows, 57 performance rows, pending registry status and task in_progress were checked. The temporary checker initially counted repeated pending cells; the corrected check counts result rows. No metric or acceptance threshold changed. task.py validate passed both 15-entry manifests; the private-data guard and git diff --check passed. Runtime and binary hashes still match. Evidence: runs/p2-development/handover-document-check-20260929-204737-974268.json.

At 20:47 America/Chicago on 2026-09-29, both original shells and the same training child were still alive. ETTh1/H720/seed2021 had four completed epoch records, up from one at the 20:44 snapshot. Completed Rust ETT coverage remained 9/48. The continuation remained waiting. No completion receipt exists for the active run.

## Cement consumer finding: 2026-09-29 21:05 America/Chicago

A blocking validation defect is now established in cement_entries, cement_gate and cement_report. The evaluator serializes original-unit variable metrics as a named mapping; these three consumers assume positional list entries. This finding supersedes the earlier no-additional-defect statement for the cement path. The original functions reproduce TypeError, KeyError and TypeError. A candidate patch passed eight synthetic consumer tests with zero failures, errors or ignored tests. Runtime and frozen tooling remain unchanged. Evidence: research/cement-metric-contract-diagnosis.md and runs/p2-development/cement-contract-20260929-210242-974583/review.json.

The complete final inline review remains open. An explicit frozen-tool transition decision is pending; original training continues. No modified tool or unvalidated receipt has been admitted to formal acceptance.

## Verification continuation: 2026-09-29 21:20 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 18/48 passed frozen entry validation. 12/12 complete-configuration diagnostic comparisons satisfy the unchanged D4 bound. Full D4 remains pending. The original shell PID 43756 and continuation PID 40828 retain their recorded creation times. Active run: rust-ETTh2-itransformer-cuda-h336-s2021-20260929-211846-941822. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/verification-update.json; runs/p2-development/continued-verification-20260929-211647-8010923/process-latest.json.

The cement candidate passed an additional real-artifact report integration check: nine accepted P1 entries, 18 variables per entry, 108 report rows and 324 baseline metric cells. P2 inputs were synthetic; zero formal CUDA runs were added. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/cement-integration.json. The earlier 8 synthetic methods remain passed. The production tool and old suite/receipts are unchanged. Evidence-transition selection remains pending.

Reports and the model card now show the verified snapshot. The registry remains pending. Final CUDA quality and final inline trellis-check remain open. No duplicate training, CPU training, task restart, commit, archive or push occurred.

## Approved evidence transition, 2026-09-29

The user explicitly selected versioned migration with full verification of unaffected ETT receipts and reruns of affected gates and final quality. See research/tooling-transition.md. The idle downstream waiter PID40828 was deliberately stopped; original ETT/D4 PID43756 remains active. The original waiting status is historical. The new suite and corrected production validator remain pending until parent ETT48 and D4 32/32 finish. No training was interrupted or duplicated.

## Half-matrix checkpoint: 2026-09-29 21:43 America/Chicago

All 48 official and 24 Rust ETT entries were revalidated, including identities, full coverage, best checkpoints and receipts. All 16 available diagnostic comparisons meet the original bound. Full D4, cement, recomputation and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-midpoint-20260929-214323-0225823.json. ETT/performance/model-card snapshots now reflect 24 Rust entries and 72 measured runs. No formal performance gate was generated.

Versioned migration is explicitly selected. Replacement executor PID32276 waits for the original ETT/D4 PID43756. All 52 repository plus staged Python tests passed, with zero failures/errors/ignored. The same-thread heartbeat ferrots-p2-cuda provides continued observation and recovery; it does not create a subagent or duplicate training. Frozen production files remain unchanged.

## Main-session transition review: 2026-09-29 22:05 America/Chicago

The scoped validator correction, provenance adapter, parent audit and sequential executor were reviewed in the main session. Training functions, configuration validation, fixed profiles, thresholds and performance calculations remain AST-identical. Current staged hashes match the 52-test evidence; runtime source and CUDA binary match the original frozen suite. The original production validator remains installed while ETT training continues.

The backend contract now specifies named original-unit metric dictionaries, configuration-order reporting and explicit versioned evidence import. Final reconciliation now checks all performance report cells in addition to ETT and cement values. No formal gate was marked passed by this static review. Evidence: runs/p2-development/tooling-transition-20260929-2129/main-session-transition-review-20260929-220524-020555.json.

## Retained interruption and scoped recovery: 2026-09-29 23:47 America/Chicago

The original ETT process tree exited between the 23:40:34 observation and the successor's 23:41:18 D4 check. ETTm1/H336/seed2023 retained four epoch records but has no final training summary or command receipt. The final exit code and cause are unknown. No matching Application or System level 1-3 event was found in the inspected 23:40:00-23:42:30 interval. The current CUDA doctor passed; this diagnostic does not establish the cause of the earlier exit. No CUDA failure was established and no CPU diagnostic was scheduled.

The old successor PID32276 stopped at verify-original-d4 because d4.json was absent. No dependent gate ran and no frozen tool was installed. Original logs and their copies/hashes are retained under runs/p2-development/interruption-20260929-234423-4955379. The earlier empty interruption-20260929-234348-4775294 directory is a failed evidence-copy attempt; its path composition was corrected without changing original evidence.

After confirming that all original worker processes were absent, the main session started a separate hidden PowerShell execution process, PID67224, created 2026-09-30T04:45:58.0116320Z. The new script has no wait on an old PID. It revalidates existing entries, completes missing ETT and original D4, then performs the already approved versioned transition and remaining gates. Script: runs/p2-development/tooling-transition-20260929-2129/recover-versioned-20260929-2345.ps1. SHA256: 32fbcabfd92aa51d1689b64cb7c8002353b297ec4096a46b8cd0bf792cb49e7f. Launch receipt: runs/p2-development/tooling-transition-20260929-2129/recovery-launch-20260929-234557-9868650/launch.json. Current ownership is recorded in active-executor.json in that staging directory. The process was observed alive after the launching shell returned; creation-time checks use UTC ticks.

Official CUDA ETT remains 48/48. Rust ETT has 32 verified completed entries. The latest complete-configuration diagnostic has 20 passes and zero failures; full D4 remains pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/configuration-diagnostics-20260929-230734-090871.json. The missing H336/seed2023 is a new attempt; the interrupted attempt is not counted. Runtime, binary, seeds, configuration and thresholds remain unchanged. P2 remains in_progress; no commit, archive or push occurred.

## Main-session review checkpoint: 2026-09-30 01:46 America/Chicago

The existing recovery executor remains active under PID67224 with the recorded creation time and command line. Full entry revalidation covers 48 official and 42 Rust ETT runs. All 28 available complete-configuration comparisons pass with ddof=0; zero fail. These comparisons remain diagnostic. The complete formal D4, cement, independent recomputation and final quality are still pending.

The main session reread the P2 task artifacts, backend contracts, configuration checks, model mapping/forward path, TrainingSession restoration checks, checkpoint bindings, CLI resume, numerical/recovery tests, current quality commands and staged evidence migration. No additional blocking defect was found in these inspected paths. Runtime code, binary, configurations, thresholds and staged installation files were not changed. The final inline Trellis sign-off remains open until formal gates and report reconciliation pass.

Evidence: runs/p2-development/tooling-transition-20260929-2129/inline-scope-review-20260930-014817-1098263.json. Diagnostic evidence: runs/p2-development/tooling-transition-20260929-2129/configuration-diagnostics-20260930-014609-083745.json. All prior failed and interrupted attempts remain retained. No duplicate training, independent CPU run, commit, archive or push occurred.

## ETT and migration checkpoint: 2026-09-30 02:54 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 48/48 completed with normal exits. Both original and successor D4 passed 32/32 with ddof=0, zero failed comparisons and unchanged thresholds. The complete parent audit revalidated all 96 entries, prerequisite gates, quality and original receipts. Successor suite runs/p2-acceptance/20260929-cuda-v3 contains 96 byte-identical imports with bound original receipt paths. The tested cement/provenance correction is installed. Runtime source, CUDA binary, profiles, seeds and tolerances remain unchanged. Current successor quality passed Rust 102 and Python 52, with zero failed or ignored tests. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-transition-checkpoint-20260930-025430-462084.json.

PID67224 stopped after ETT and original D4 passed. Its final error was MSVC initialization failed; retained stderr reports The input line is too long. Repeated initialization expands PATH: an isolated reproduction passed from lengths 4045 and 5802, then failed from length 7559. Two fresh-process probes passed with the required Rust 1.98.0, CUDA 13.3 and static-CRT flags. No CUDA failure was established. The frozen environment helper remains unchanged. Evidence: runs/p2-development/tooling-transition-20260929-2129/msvc-path-reproduction-20260930-024540-7128072/diagnosis.json and runs/p2-development/tooling-transition-20260929-2129/isolated-env-check-20260930-024757-6311963/result.json.

Replacement executor PID73220 was created at 2026-09-30T07:48:38.2326413Z. It keeps an uninitialized parent environment and creates a new child process for every stage. Python stages load p1_env once. Quality stages let p2_quality load the environment once. Script: runs/p2-development/tooling-transition-20260929-2129/continue-isolated-20260930-0246.ps1; SHA256 fd28baca7f56f8fc9c6235d5ac50380d5354110b3a42ca539b69817782040f8c. Ownership remains in active-executor.json. No ETT training was repeated. Cement is active; independent recomputation, final quality, final acceptance and final main-session sign-off remain pending. P2 remains in_progress. No commit, archive or push occurred.

## Cement interruption and recovery: 2026-09-30 03:11 America/Chicago

Executor PID73220 and training PID38688 exited between the 02:52:38 and 02:56:15 observations on September 30, 2026. The first H15/seed2021 attempt has zero completed epochs and no final command receipt. The exit code and cause remain unknown. The interval overlaps a host application restart; causality was not established. No CUDA failure was established. Original logs and hashes remain in runs/p2-development/tooling-transition-20260929-2129/host-interruption-20260930-030009-2678822/interruption.json.

After verifying that all prior P2 process trees had ended, the main session launched executor PID67188 at 2026-09-30T08:06:42.4105976Z through Windows CIM. The observed parent is WmiPrvSE.exe. The executor was verified alive after the launch command returned; survival through an application restart was not tested. The executor retains an uninitialized parent environment and creates one child environment per stage. Only cement, independent recomputation, final CUDA quality and final acceptance are scheduled. ETT96, the audited versioned migration, current quality and D4 were reused.

At 2026-09-30T03:11:56.2431998-05:00, cement has 0/9 complete entries. Active run: rust-cement-itransformer-cuda-h15-s2021-20260930-030647-900688; training PID25580, completed epoch records 0. Current owner: runs/p2-development/tooling-transition-20260929-2129/active-executor.json. Checkpoint: runs/p2-development/tooling-transition-20260929-2129/cement-recovery-checkpoint-20260930-031156-482696.json. The task remains in_progress. Independent recomputation, final quality, final acceptance and final main-session Trellis sign-off remain pending. No CPU training, commit, archive or push occurred.

The recovery documentation passed four local checks: task context validation (17 entries in each manifest), the private-data guard, git diff --check, and data/runs ignore checks. Each command exited 0. Evidence: runs/p2-development/tooling-transition-20260929-2129/document-check-20260930-031230-7592278/result.json. These checks do not replace the pending final CUDA quality gate.

## First cement epoch: 2026-09-30 05:44 America/Chicago

The active CUDA H15/seed2021 run completed epoch 1: 299,819 training windows, 9,370 optimizer steps, zero skipped empty batches, train MSE 0.10933995531677243 and validation MSE 0.10041589545685217. Training took 9,299.952 seconds and validation took 7.171 seconds; the combined epoch time was 9,307.124 seconds (155.1 minutes). Checkpoint timing was 10.181 seconds. Steady throughput was 32.239 windows/s. Best and last checkpoint directories are present. These observations cover one epoch of an unfinished run; they do not establish final checkpoint validity or add a completed formal entry. Cement remains 0/9; executor PID67188 and training PID25580 remain active. The full matrix, independent recomputation, final quality and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/cement-first-epoch-20260930-054416-417122.json.

## Main-session performance repair review: 2026-10-01

The review checked the user-requested stop evidence, current process absence, paused heartbeat, source diff, current binary identity, diagnostic receipts and quality output. No FerroTS training or quality process remains active.

The scoped correction is limited to package-scoped Cargo dev-profile optimization and the opt-in cuda_profile example. Batch size, accumulation, seeds, FP32, model dimensions, RNG identities, data protocol and tolerances remain unchanged. The synthetic comparison is bitwise identical across 115 saved tensors and exact across all 12 loss values.

The bounded real-data diagnostic used cuda:0 on the NVIDIA GeForce RTX 5090 Laptop GPU and exited normally after 128 optimizer steps. The diagnostic is not a formal receipt. The corrected binary changed runtime identity, so the previous 96 ETT entries and D4 receipts remain historical evidence only. The current quality suite passed Rust 102 and Python 52 with zero failures and zero ignored tests.

Inline review result: repair scope is internally consistent; formal P2 acceptance is not signed off. Required follow-up remains a fresh formal matrix under the repaired identity, then independent recomputation, final quality, report reconciliation and final review.
