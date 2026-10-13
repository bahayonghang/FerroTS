# P2 CUDA acceptance status

## Linux formal suite: 2026-10-12

Suite `runs/p2-acceptance/20261012-linux-cuda` finished. Executor PID 450356 is gone, and no training process remains. `final.json` passed.

| Check | Passed | Failed | Ignored | Boundary |
| --- | ---: | ---: | ---: | --- |
| Quality workspace tests | 102 | 0 | 0 | Sum of `test result` lines, including CPU and CUDA numerical tests on this source |
| Quality Python tests | 55 | 0 | 0 | `Ran 55 tests` and `OK` |
| CUDA numerical cases | 9 | 0 | 0 | Dedicated gate is CUDA only; `device=cuda`, acceptance version 2 |
| CUDA fresh-process recovery | 10 | 0 | 0 | All cases `cuda:0`, dropout 0.1 |
| Official CUDA ETT | 48 | 0 | 0 | cuda GPU index 0, normal exits, full test arrays |
| Rust CUDA ETT | 48 | 0 | 0 | cuda:0, batch 32 |
| D4 comparisons | 32 | 0 | 0 | ddof=0, original two-sided bound |
| Cement runs | 9 | 0 | 0 | All 18 variables; six runs cuda:0, three runs cuda:1; batch 32 |
| Cement report rows | 108 | 0 | 0 | 3 horizons × 18 variables × MAE/RMSE |
| Independent recomputation | 6 | 0 | 0 | 18,574,560 rows; largest absolute difference 1.2505552149377763e-12 |
| Performance rows | 105 | 0 | 0 | 48 official + 57 Rust, one row per completed training run |

Runtime source SHA256: `fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055`. Linux binary SHA256: `43fd7365d289e01d3c0ff14882a8b54f7facdf79795af3905d05931bba9fe596`. Host GPU: NVIDIA GeForce RTX 4090. P1 run directories were not loaded; the cement report uses the committed P1 CUDA table, seed 2021, n=1.

The earlier H30 seed 2021 directory `rust-cement-itransformer-cuda-h30-s2021-20261012-033048-460448` has no entry and is not counted. The counted H30 seed 2021 entry is the later completed run.

Not passed, and not counted as passed:

- Step 15 full CPU control. The CUDA-first instruction left it unexecuted. The CPU numerical test in the quality log is not that control.
- Parent X5 sentence that GPU acceptance runs on the RTX 5090 Laptop. Windows v5 did not finish and was not resumed here.
- R8's original wording that every cement run uses cuda:0. Three completed cement runs used cuda:1.

Evidence: `runs/p2-acceptance/20261012-linux-cuda/final.json`, `quality.json`, `recompute.json`, and `runs/p2-development/20261012-linux/executor-dual.log`.

## First cement epoch: 2026-09-30 05:44 America/Chicago

The active CUDA H15/seed2021 run completed epoch 1: 299,819 training windows, 9,370 optimizer steps, zero skipped empty batches, train MSE 0.10933995531677243 and validation MSE 0.10041589545685217. Training took 9,299.952 seconds and validation took 7.171 seconds; the combined epoch time was 9,307.124 seconds (155.1 minutes). Checkpoint timing was 10.181 seconds. Steady throughput was 32.239 windows/s. Best and last checkpoint directories are present. These observations cover one epoch of an unfinished run; they do not establish final checkpoint validity or add a completed formal entry. Cement remains 0/9; executor PID67188 and training PID25580 remain active. The full matrix, independent recomputation, final quality and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/cement-first-epoch-20260930-054416-417122.json.

## Cement interruption and recovery: 2026-09-30 03:11 America/Chicago

Executor PID73220 and training PID38688 exited between the 02:52:38 and 02:56:15 observations on September 30, 2026. The first H15/seed2021 attempt has zero completed epochs and no final command receipt. The exit code and cause remain unknown. The interval overlaps a host application restart; causality was not established. No CUDA failure was established. Original logs and hashes remain in runs/p2-development/tooling-transition-20260929-2129/host-interruption-20260930-030009-2678822/interruption.json.

After verifying that all prior P2 process trees had ended, the main session launched executor PID67188 at 2026-09-30T08:06:42.4105976Z through Windows CIM. The observed parent is WmiPrvSE.exe. The executor was verified alive after the launch command returned; survival through an application restart was not tested. The executor retains an uninitialized parent environment and creates one child environment per stage. Only cement, independent recomputation, final CUDA quality and final acceptance are scheduled. ETT96, the audited versioned migration, current quality and D4 were reused.

At 2026-09-30T03:11:56.2431998-05:00, cement has 0/9 complete entries. Active run: rust-cement-itransformer-cuda-h15-s2021-20260930-030647-900688; training PID25580, completed epoch records 0. Current owner: runs/p2-development/tooling-transition-20260929-2129/active-executor.json. Checkpoint: runs/p2-development/tooling-transition-20260929-2129/cement-recovery-checkpoint-20260930-031156-482696.json. The task remains in_progress. Independent recomputation, final quality, final acceptance and final main-session Trellis sign-off remain pending. No CPU training, commit, archive or push occurred.

## ETT and migration checkpoint: 2026-09-30 02:54 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 48/48 completed with normal exits. Both original and successor D4 passed 32/32 with ddof=0, zero failed comparisons and unchanged thresholds. The complete parent audit revalidated all 96 entries, prerequisite gates, quality and original receipts. Successor suite runs/p2-acceptance/20260929-cuda-v3 contains 96 byte-identical imports with bound original receipt paths. The tested cement/provenance correction is installed. Runtime source, CUDA binary, profiles, seeds and tolerances remain unchanged. Current successor quality passed Rust 102 and Python 52, with zero failed or ignored tests. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-transition-checkpoint-20260930-025430-462084.json.

PID67224 stopped after ETT and original D4 passed. Its final error was MSVC initialization failed; retained stderr reports The input line is too long. Repeated initialization expands PATH: an isolated reproduction passed from lengths 4045 and 5802, then failed from length 7559. Two fresh-process probes passed with the required Rust 1.98.0, CUDA 13.3 and static-CRT flags. No CUDA failure was established. The frozen environment helper remains unchanged. Evidence: runs/p2-development/tooling-transition-20260929-2129/msvc-path-reproduction-20260930-024540-7128072/diagnosis.json and runs/p2-development/tooling-transition-20260929-2129/isolated-env-check-20260930-024757-6311963/result.json.

Replacement executor PID73220 was created at 2026-09-30T07:48:38.2326413Z. It keeps an uninitialized parent environment and creates a new child process for every stage. Python stages load p1_env once. Quality stages let p2_quality load the environment once. Script: runs/p2-development/tooling-transition-20260929-2129/continue-isolated-20260930-0246.ps1; SHA256 fd28baca7f56f8fc9c6235d5ac50380d5354110b3a42ca539b69817782040f8c. Ownership remains in active-executor.json. No ETT training was repeated. Cement is active; independent recomputation, final quality, final acceptance and final main-session sign-off remain pending. P2 remains in_progress. No commit, archive or push occurred.

## Historical snapshots

Snapshot: 2026-09-29 21:20 America/Chicago. Completed entries were verified before report generation; active-process evidence is timestamped separately.

Status: in progress. Final acceptance and the final inline Trellis check remain pending. This report records verified completed work while the existing sequential execution continues.

Suite: runs/p2-acceptance/20260929-cuda-v2. Runtime source SHA256: d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81.

Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/verification-update.json; runs/p2-development/continued-verification-20260929-211647-8010923/process-latest.json.

## Verified gates

| Check | Passed | Failed | Ignored | Boundary |
| --- | ---: | ---: | ---: | --- |
| Recorded CUDA workspace tests | 102 | 0 | 0 | Pre-matrix quality logs and input hashes revalidated; final rerun pending |
| Recorded Python tests | 28 | 0 | 0 | Pre-matrix quality logs and input hashes revalidated; final rerun pending |
| CUDA numerical cases | 9 | 0 | 0 | Each case contains all 38 parameters and shared-gradient Adam controls |
| CUDA fresh-process recovery cases | 10 | 0 | 0 | All 38 parameters; maximum tensor difference 0 in every case |
| Official CUDA ETT entries | 48 | 0 | 0 | All required keys, normal exits, best weights, full arrays and receipt hashes revalidated |
| Completed Rust CUDA ETT entries | 18 | 0 | 0 | Full frozen entry validation; 30 required entries remain |

The failed counts above describe the validated gates and completed entries. The counts exclude historical failed or interrupted attempts. The initial np.Inf failure remains in the original suite. The earlier H336 attempt has no final exit code; its exit cause remains unknown.

Preflight and the complete data gate were also revalidated. The data gate retains all 16 ETT configurations, Exchange 7,588 rows, cement 48 segments with a 37/4/7 split, 303,009 scaler-fit minutes and every required horizon window count.

The approved AC3 contract remains unchanged: true MSE gradients use atol=1e-5 and rtol=1e-4; both Adam implementations receive the same official gradients and compare all parameters, moments and steps at atol=1e-6 and rtol=0. Independent-backward update differences remain diagnostics.

## Active execution

- Existing ETT and D4 shell: PID 43756, created 2026-09-30T01:24:07.9859469Z.
- Existing continuation: PID 40828, created 2026-09-30T01:30:52.2535464Z; state waiting.
- Active training: rust-ETTh2-itransformer-cuda-h336-s2021-20260929-211846-941822; PID 37916; 3 completed epoch records when read.
- The continuation waits for the exact ETT shell. A complete D4 pass is required before cement, recompute, final CUDA quality and final acceptance.
- No duplicate runner or training process was started. The current-task pointer is absent; the existing task.json remains in_progress. No task activation was repeated.

## Remaining acceptance

| Requirement | Current status | Completion evidence required |
| --- | --- | --- |
| Rust ETT | 18/48 verified | Complete remaining 30 entries with fixed seeds/configurations and immutable receipts |
| D4 | Formal gate not produced | All 32 comparisons pass with ddof=0 and the original two-sided bound |
| Cement | 0/9 completed | H15/30/60 and three seeds; all 18 variables in original units |
| P1 comparison | P1 baseline inventory hashes verified | P1 suite 20260928-0842-rng beside P2; n=1 and standard deviation not applicable for P1 |
| Independent recomputation | Not started | 103,192 origins x 60 steps x 3 variables = 18,574,560 rows; six MAE/RMSE differences below 1e-6 |
| Final quality | Not started | Full p2_quality.ps1 after the formal matrix; retained host assertions |
| Final gate and inline check | Pending | final.json, report reconciliation and complete main-session sign-off |

12 of 12 available diagnostic comparisons satisfy the original bound; 0 fail. Full D4 remains pending. Entry checks cover frozen identities, seeds, normal termination, full windows, best checkpoints and performance receipts.

## Untested and excluded boundaries

The original full CPU control remains unexecuted under the approved CUDA-first policy. No independent CPU-only suite or CPU training was run in this continuation. Earlier CPU evidence is retained with its original identity. Windows/CUDA acceptance does not establish another OS, GPU, precision or cross-hardware recovery. Process memory is unavailable; device-wide memory sampling and official allocator peaks remain distinct.

The model registry remains pending while formal reproduction is incomplete. All prediction exports stay under ignored runs/. No private raw data rows are included in reports. No commit, archive, push or pull request was performed.

## Historical continuation observations

At 20:47 America/Chicago on 2026-09-29, both original shells and the same training child were still alive. ETTh1/H720/seed2021 had four completed epoch records, up from one at the 20:44 snapshot. Completed Rust ETT coverage remained 9/48. The continuation remained waiting. No completion receipt exists for the active run.

Report checks passed for all nine updated files. ETT has 32 comparison rows, six diagnostic results and 26 pending results. Performance has 57 completed-run rows. Frozen source and binary identities remain unchanged. Evidence: runs/p2-development/handover-document-check-20260929-204737-974268.json.

## Confirmed pending tooling correction

The cement metric reader, seed aggregator and P1 report contain a named-dictionary versus positional-list mismatch. The current frozen tool cannot complete cement acceptance. A candidate patch passed 8/8 synthetic regression methods; no production tool has changed. This result is separate from the recorded Rust 102/Python 28 quality evidence and does not complete a CUDA matrix item.

See research/cement-metric-contract-diagnosis.md. The frozen-tool evidence transition decision is pending while the original ETT training continues.

At 21:08 America/Chicago on 2026-09-29, 14 Rust ETT entries were verified. ETTh1 is complete at 12/12; all eight ETTh1 diagnostic comparisons satisfy the unchanged bound. Full D4 remains pending. Evidence: runs/p2-development/ett-hourly-milestone-20260929-210833-166843.json. The original execution chain remains active; the cement candidate and evidence-transition decision remain pending. No frozen tool, runtime or receipt changed.

## Verified continuation at 2026-09-29 21:20 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 18/48 are verified. 12/12 available diagnostic comparisons meet the unchanged bound. The original shells remain active. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/verification-update.json; runs/p2-development/continued-verification-20260929-211647-8010923/process-latest.json.

Candidate real-P1 integration passed: nine entries, 18 variables per entry, 108 rendered rows and 324 baseline metric cells. The check uses synthetic P2 inputs and adds zero formal CUDA runs. Frozen tooling, suite.json and production cement report were unchanged during the integration. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/cement-integration.json.

Final quality, full formal gates, final inline sign-off and the explicit frozen-evidence transition decision remain pending.

## Additional completed-entry verification: 2026-09-29 21:22 America/Chicago

Rust ETT now has 19/48 verified completed entries. The additional entry validator passed with 0 failures and 0 ignored entries. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/additional-entries.json. The ETT/performance report tables retain their explicitly dated 2026-09-29T21:20:14.435108-05:00 snapshot of 18 Rust entries; the additional entry is recorded here and in task.json until the next report refresh. Existing PID 43756 and PID 40828 remain alive. Active run at the process snapshot: rust-ETTh2-itransformer-cuda-h336-s2022-20260929-212103-947617. No formal D4 or cement gate has completed.

## Approved evidence transition, 2026-09-29

The user explicitly selected versioned migration with full verification of unaffected ETT receipts and reruns of affected gates and final quality. See research/tooling-transition.md. The idle downstream waiter PID40828 was deliberately stopped; original ETT/D4 PID43756 remains active. The original waiting status is historical. The new suite and corrected production validator remain pending until parent ETT48 and D4 32/32 finish. No training was interrupted or duplicated.

## Half-matrix checkpoint: 2026-09-29 21:43 America/Chicago

All 48 official and 24 Rust ETT entries were revalidated, including identities, full coverage, best checkpoints and receipts. All 16 available diagnostic comparisons meet the original bound. Full D4, cement, recomputation and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-midpoint-20260929-214323-0225823.json. ETT/performance/model-card snapshots now reflect 24 Rust entries and 72 measured runs. No formal performance gate was generated.

Versioned migration is explicitly selected. Replacement executor PID32276 waits for the original ETT/D4 PID43756. All 52 repository plus staged Python tests passed, with zero failures/errors/ignored. The same-thread heartbeat ferrots-p2-cuda provides continued observation and recovery; it does not create a subagent or duplicate training. Frozen production files remain unchanged.

## Retained interruption and scoped recovery: 2026-09-29 23:47 America/Chicago

The original ETT process tree exited between the 23:40:34 observation and the successor's 23:41:18 D4 check. ETTm1/H336/seed2023 retained four epoch records but has no final training summary or command receipt. The final exit code and cause are unknown. No matching Application or System level 1-3 event was found in the inspected 23:40:00-23:42:30 interval. The current CUDA doctor passed; this diagnostic does not establish the cause of the earlier exit. No CUDA failure was established and no CPU diagnostic was scheduled.

The old successor PID32276 stopped at verify-original-d4 because d4.json was absent. No dependent gate ran and no frozen tool was installed. Original logs and their copies/hashes are retained under runs/p2-development/interruption-20260929-234423-4955379. The earlier empty interruption-20260929-234348-4775294 directory is a failed evidence-copy attempt; its path composition was corrected without changing original evidence.

After confirming that all original worker processes were absent, the main session started a separate hidden PowerShell execution process, PID67224, created 2026-09-30T04:45:58.0116320Z. The new script has no wait on an old PID. It revalidates existing entries, completes missing ETT and original D4, then performs the already approved versioned transition and remaining gates. Script: runs/p2-development/tooling-transition-20260929-2129/recover-versioned-20260929-2345.ps1. SHA256: 32fbcabfd92aa51d1689b64cb7c8002353b297ec4096a46b8cd0bf792cb49e7f. Launch receipt: runs/p2-development/tooling-transition-20260929-2129/recovery-launch-20260929-234557-9868650/launch.json. Current ownership is recorded in active-executor.json in that staging directory. The process was observed alive after the launching shell returned; creation-time checks use UTC ticks.

Official CUDA ETT remains 48/48. Rust ETT has 32 verified completed entries. The latest complete-configuration diagnostic has 20 passes and zero failures; full D4 remains pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/configuration-diagnostics-20260929-230734-090871.json. The missing H336/seed2023 is a new attempt; the interrupted attempt is not counted. Runtime, binary, seeds, configuration and thresholds remain unchanged. P2 remains in_progress; no commit, archive or push occurred.

## Current acceptance boundary after performance repair: 2026-10-01

The user-requested training stop is complete. The current heartbeat is paused and no related process remains. The performance repair quality suite passed Rust 102 and Python 52 with zero failures and zero ignored tests.

The Cargo dev-profile correction and diagnostic example changed the runtime identity. The successor suite still contains 96 byte-identical ETT entries and both D4 results pass 32/32 for their recorded identity. Those results remain retained historical evidence. They do not satisfy current-source formal acceptance for the repaired binary.

Cement remains 0/9 formal entries. The short real-data run is diagnostic only. CPU full formal control remains unexecuted.
