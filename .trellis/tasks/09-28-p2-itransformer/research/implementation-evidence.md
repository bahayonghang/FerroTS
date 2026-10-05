# P2 implementation evidence

## 2026-09-29: activation and first change boundary

The approved standalone Trellis command activated the existing P2 task. Branch: `feat/p2-itransformer`; implementation base: `1f1b9ec`. The task remains incomplete.

Step 2 owns the first code change. The repository has pinned upstream source and P1 reference tools, but no iTransformer fixture exporter or official runner. Add `tools/reference/gen_p2_fixtures.py` for synthetic tensors, layer traces, MSE gradients, connectivity probes, Adam state and initialization/sampling evidence. Add `tools/reference/run_itransformer.py` for the pinned official entry point and its execution receipt. Add focused validation tests and a fixture source/mapping note. Reuse `fetch_upstream.py` pins and the existing uv environment.

This step does not modify production Rust numerics, dependencies, upstream checkouts, private data, P1 reports, or acceptance thresholds. Later steps retain the approved order. Formal training starts only after the numerical, resume and regression gates pass.

## Steps 2 and 3

- The exporter produced nine synthetic cases with 38 parameters each, all layer traces, true MSE gradients, connectivity-probe gradients and one-step Adam states. Regeneration compared every tensor byte and manifest field. All three seeds and 30 sampler epochs passed. Python empty-buffer loading passed; Rust loading remains assigned to step 10.
- The official runner now verifies the pinned clean checkout and local data, records effective arguments and execution identity, and keeps output under runs/. No full official training has run.
- Python validation: 18 passed, 0 failed. The privacy guard and git diff check passed.
- Strict core configuration adds the iTransformer fields and an explicit drop_last flag. False is omitted from serialization. Core tests: 13 passed, 0 failed. Log: runs/p1-development/20260929-014614-8265706.log.
- Common Baseline dispatch passed the first complete CPU workspace regression. Log: runs/p1-development/20260929-014831-1949480.log. Subsequent registry and explicit-tail tests required another full run.

## Step 4 regression: concurrent fixture directory collision

The second CPU workspace run failed at regression_matrix.rs:60: the duplicate-timestamp negative test returned success. Original log: runs/p1-development/20260929-015024-2430402.log. Dependent implementation stopped for diagnosis.

The CLI fixture helper used only process ID and wall-clock nanoseconds with create_dir_all. Four concurrent tests plus one later fixture should create five directories. The failed test process left only four. A new concurrent allocation test confirmed the collision: 32 fixture calls returned only 31 unique paths, including an identical process/timestamp pair. Reproduction log: runs/p1-development/20260929-015318-0144905.log. A prior single trial passed; the failure is intermittent.

The correction adds a process-local AtomicU64 suffix and exclusive create_dir. The fix changes only CLI test fixture allocation. It retains all duplicate-policy assertions and does not change production parsing, cache rules or model numerics. The new 32-thread regression test remains in the suite. Corrected complete CPU workspace regression passed: runs/p1-development/20260929-015359-2770549.log.

## Step 5: training-safe layers

Biased last-axis Linear and exact GELU now compose training-safe Candle operations. GELU uses erf because Candle 0.11.0 gelu_erf backward uses the rounded coefficient 0.398942. CPU f64 finite differences and the official FP32 Linear/GELU/LayerNorm fixture passed. CPU nn: 18 passed, 0 failed, 0 ignored (runs/p1-development/20260929-015511-5456432.log). CUDA nn: 22 passed, 0 failed, 0 ignored (runs/p1-development/20260929-015525-9201447.log). Existing fused-path negative tests remain active.

## Step 6: model and integration

The model follows the official module construction order, inverted embedding, post-norm encoder, final norm, projection and variable-token slice. Nine official layer/output cases and three-seed initialization bits passed on CPU. Thirty sampler epochs matched the official sequences. The model is registered with reproduction status pending. Checkpoint v1 records the model-specific upstream identity.

Additional checks cover official Conv1d export/import shapes, atomic rejection of incomplete/extra/invalid weight sets, history/time shape and dtype bindings, target subsets after time-token slicing, and inference checkpoint roundtrips. Frozen snapshots have independent storage. The existing WindowIndex excludes unavailable history before Dataset batches are built; future_batch explicitly rejects missing tail history. No change to missing-data semantics was needed.

The first CUDA nn regression failed before the existing DLinear test harness printed its test count. Exit: 0xc0000005 / STATUS_ACCESS_VIOLATION. Logs: runs/p1-development/20260929-020255-3185029.log and 20260929-020325-9178797.log. A direct --list invocation also failed. The Windows debug-event probe subsequently listed the unchanged binary successfully; five native --list trials and the native DLinear test passed (runs/p2-development/dlinear-startup-debug.log, dlinear-startup-native.log; runs/p1-development/20260929-020522-0726362.log). The startup failure cause remains unknown. No environment, library or test workaround was applied. The original failures are retained.

CPU nn after the additional mapping/binding tests: 22 passed, 0 failed, 0 ignored (runs/p1-development/20260929-020656-7036619.log). CPU checkpoint integration: 3 passed, 0 failed, 0 ignored (runs/p1-development/20260929-020702-5076321.log). CUDA nn: 26 passed, 0 failed, 0 ignored (runs/p1-development/20260929-020719-3066137.log). CUDA checkpoint integration: 3 passed, 0 failed, 0 ignored (runs/p1-development/20260929-020728-9989314.log). Step 6 passed. Training dropout context, resume state and formal training remain unfinished.

## Step 7: full numerical gate failed

Both CPU and CUDA passed all 342 MSE parameter-gradient checks, 342 probe checks, 342 first moments, 342 second moments and 342 step counters. Each device failed 16 of 342 updated parameter checks; all failures are key_projection.bias. CPU maximum update error: 1.6952864825725555e-5. CUDA maximum: 2.126116305589676e-5. The tolerance remains 1e-6 with rtol=0.

The optimizer control using identical official MSE gradients passed every parameter/moment/step on both devices. The unchanged official model itself failed the same 16 updated-parameter checks between CPU and CUDA with TF32 disabled. See research/numerical-gate-diagnosis.md for the complete counts, source paths, logs, numerical mechanism and a proposed acceptance revision. The proposal is not approved and does not change the current gate. The failing test remains active. Steps 8-20 have not started; no formal training, commit, push or archive occurred.


## Steps 7-9: approved AC3 and verified continuation

On 2026-09-29 the user explicitly approved the AC3 revision in numerical-gate-diagnosis.md section 5. The preceding failure section describes the original gate and is retained as historical evidence. The revised gate retains all true MSE gradients and connectivity checks. Adam uses the same official MSE gradients for all parameters, moments and steps, with atol=1e-6 and rtol=0. Independent-backward updates remain complete diagnostics. Production gradients and hyperparameters are unchanged.

Revised numerical gate: CPU 1 passed (runs/p1-development/20260929-021908-3702758.log); CUDA build 2 passed, covering CPU and CUDA (20260929-021841-8428835.log). No failures or ignored tests.

Step 8 persists the versioned MT19937 state and separates next_forward_id from optimizer steps. Shared-mask fixtures cover nine dropout sites for two forward IDs; the official exporter reuses those exact masks. Regeneration passed (20260929-022155-8556185.log). Core RNG: 2 passed (20260929-022218-9118932.log). CPU/CUDA iTransformer: 6 passed each (20260929-022221-1038020.log, 20260929-022335-8660022.log). CPU/CUDA-build Trainer: 5 passed each (20260929-022223-4441849.log, 20260929-022342-2703739.log).

Step 9 integrates TrainingSession into the production Trainer. A single state machine handles training, pending validation, next epoch and completion. In-memory snapshots preserve all weights, moments, counters, scheduler, RNG, permutation hash and early-stopping state at empty accumulation boundaries. Tests cover accumulation 1/2, dropout 0.1, tail groups, strict ties/patience and corrupt states. CPU/CUDA-build Trainer: 8 passed each (20260929-022857-3513839.log, 20260929-023105-6637010.log). Complete CPU workspace test passed (20260929-023103-0996511.log). Filesystem resume and v2 publication remain unfinished.

A second pre-harness STATUS_ACCESS_VIOLATION affected the CPU numerical binary (20260929-021816-2113661.log and 20260929-021839-8047533.log). The debug-event probe could list tests, then the unchanged native numerical test passed. Cause remains unknown. No environment, library or test workaround was applied.

Step 10 change boundary: extend checkpoint.rs with strict v1/v2 dispatch, a twelve-file v2 transaction, complete identity/state validation and append-only references. Reuse the existing optimizer format and TrainingSession state. Add transaction/corruption tests. Preserve v1 inference and reject v1 resume. No formal training starts before the remaining resume and regression gates.


## Step 10: complete v2 checkpoint transaction

The twelve-file v2 generation records all eleven payload hashes and sizes, model/mapping/schema/RNG and L/H/C/K bindings, environment and source identity, complete optimizer/trainer/scheduler/RNG state, and an optional parent generation. Empty buffers are valid Rust-readable safetensors. V1 remains inference-only. Full resume checks run before the new session is returned; failed loads do not modify an existing session.

The first filesystem roundtrip exposed a serde_json FP64 parsing difference: train_sum 1.5005179643630981 became 1.500517964363098. Original failure: runs/p1-development/20260929-024049-0312956.log. Enabled float_roundtrip on the existing serde_json 1.0.151 dependency. No package version or lockfile changed. The fix preserves the original numeric JSON format and exact FP64 state. Broader P1 cache/scaler and inference regressions remain part of steps 14/19.

CPU data/train tests passed: 41 passed, 0 failed, 0 ignored (runs/p1-development/20260929-024138-0353804.log). CUDA-build checkpoint/v2/optimizer tests passed: 16 passed, 0 failed, 0 ignored (20260929-024229-7916529.log). V2 roundtrips cover train, pending validation and complete on both devices. Twenty-six corruption categories reject invalid file/state/identity inputs. Six publication fault points preserve the prior reference and live state. Pre-rename failures retain staging; post-rename failures retain an unreferenced immutable generation or pending reference. A corrupt latest reference fails without fallback.

The shared-dropout fixture integrity test was added. Full Python validation: 19 passed, 0 failed. No P2 formal training has started. Step 11 owns CLI stop-after-steps/resume, continuation identity, historical best inheritance and v2 evaluation/prediction.


## Step 11: resume CLI and continuation artifacts

Trainable models now publish v2 through the same TrainingSession progression used by resume. train/resume --stop-after-steps counts real optimizer updates and stops only at empty accumulation boundaries. The summary represents pre-validation best as null. Completed checkpoints perform no invented training steps. Parameter-free models retain v1 inference-only publication.

resume --checkpoint validates the original config, data, source, device and environment, then creates a separate continuation directory. The continuation records the parent manifest and both Git identities, preserves the original resolved config bytes, copies and verifies historical best when present, and leaves the parent run unchanged. CLI resume accepts no seed/device/batch/model override. Evaluation/prediction read v2 and use the continuation run identity in exported rows.

CPU CLI pipeline/resume: 6 passed, 0 failed, 0 ignored (runs/p1-development/20260929-024707-8764467.log). Trainer/v2 integration: 12 passed (20260929-024723-6507666.log). After adding an explicit continuation export run_id check, all 3 resume CLI tests passed again (20260929-024801-1113519.log). CPU/CUDA fresh-process equivalence belongs to step 12 and remains pending.


## Step 12: fresh-process restoration passed before the policy update

The CLI process gate passed all ten CPU cases and all ten CUDA cases: N=3/7, accumulation=1/2, empty supervision, tail batches/groups, type1 transitions and strict patience with an earlier best epoch. Each interrupted branch terminates its CLI process and starts resume in a new process. All 38 model parameters, both moments, step counts, RNG state, nine site masks, cursor/next batch, training and frozen predictions, historical best selection and integer trainer state are compared. CPU requires bitwise tensor equality; CUDA uses atol=1e-6 and rtol=0. Timing and run/generation identities are excluded as planned.

CPU report: runs/p2-development/resume-process-cpu-1790668265540261600.json; log: runs/p1-development/20260929-025048-9666460.log. The CUDA-build report contains the then-default CPU and CUDA cases (20 total): runs/p2-development/resume-process-cuda-1790668427685423300.json; log: runs/p1-development/20260929-025134-3704442.log. Both commands returned exit 0. The second command had completed when the user's CUDA-first instruction arrived.

## User validation preference: 2026-09-29

The user requests CUDA-first execution and no routine CPU testing/training. Only a CUDA failure permits a short CPU diagnostic of about three epochs. Saved the preference in Basic Memory decisions/ferrots/FerroTS CUDA 优先验证策略.md and the repository AGENTS.md. The CUDA process test now selects only CUDA when compiled with that feature; the CPU path remains available for authorized diagnostics. Existing evidence is retained. Original formal CPU control stays explicitly unexecuted; the full 105 CUDA runs and numerical/D4 thresholds remain unchanged.


## Step 13: fixed profiles, CUDA runner and performance observation

Added 16 horizon-specific ETT profiles, three cement profiles and one three-epoch CUDA synthetic profile. models list filters task and compiled device support. The model card and NOTICE identify the pinned MIT upstream; formal reproduction remains pending.

The P2 runner validates source/binary/config/data identities, immutable command attempts, exact seed coverage, normal epoch termination, complete train/validation/test coverage and v2 best/last state. Failed commands and launch errors retain receipts without reducing batch size. Independent P1 baseline artifacts are verified before comparison. The original CPU formal stage is omitted under the later user instruction.

Rust records synchronized training, validation and checkpoint time plus steady window throughput. The official observer instruments six exact anchors in the pinned train method in-process and records torch allocator peaks. The upstream checkout remains unchanged. Device-wide memory is sampled every 50 ms; the observer reports baseline, peak, timestamps, sample gaps and unavailable process memory explicitly. Sampling check: runs/p2-development/observer-step13/device-memory.json, 5 samples, no errors, no estimated drops.

Python validation passed 27 tests. CUDA process restoration passed ten cases after the timing fields were added; report runs/p2-development/resume-process-cuda-1790673069167459300.json, log runs/p1-development/20260929-041008-2718037.log. CUDA models list test passed (20260929-041202-3939278.log). Clippy identified one collapsible branch and two needless test borrows; local syntax fixes preserve behavior. CUDA workspace Clippy -D warnings passed (20260929-041519-7832893.log), build passed (20260929-041521-4267899.log).

Formal preflight passed at runs/p2-acceptance/20260929-cuda/preflight.json. The suite freezes runtime, tooling, configuration, data and binary identities and validates the real CUDA doctor and memory observer. No P2 formal training has started. Step 14 full CUDA regression and real data validation are next.


## Step 14: complete CUDA regression and real-data gates

Full quality passed: 102 Rust tests, 27 Python tests, zero failed or ignored. The CUDA build retains existing lightweight host/CPU assertions; no separate CPU-only build/test suite or formal CPU training was scheduled. Quality records: runs/p2-acceptance/20260929-cuda/quality/20260929-041648-2567678. fmt, CUDA build/Clippy, real cuda:0 training doctor, cuda:9 rejection, Trellis, privacy and diff gates all passed.

Fresh suite numerical and process recovery receipts passed. Numerical report covers all 38 parameters in nine official cases under the approved AC3 split; shared-dropout and layer cases also passed in the full workspace suite. Process recovery covers ten CUDA cases, all maximum tensor errors zero.

Data acceptance passed all 16 ETT configurations, Exchange 7,588 rows, cement raw counters, 48 segments, 37/4/7 split, 303,009 scaler-fit rows, all H window counts and the fresh/hit cache check. The repeat cache read has parse_count=0. Reports and command receipts: data-gate.json in the suite.

Historical P1 public ETTh1/H96/seed2021 CUDA v1 checkpoint passed complete evaluate and predict: 2,785 windows and 1,871,520 rows. Model/data/checkpoint fingerprints and full metric coverage were preserved. Maximum metric difference from the historical JSON was 1.2434497875801753e-14, within the explicit 1e-12 FP64 roundtrip check; no P1 tensor algorithm changed. Evidence: v1-regression.json. The original full CPU control remains unexecuted under the user instruction. The next step is 48 official CUDA ETT runs.


## Step 16 first attempt: pinned NumPy spelling compatibility

The first official ETTh1/H96/seed2021 attempt stopped before epoch 1. EarlyStopping.__init__ in pinned utils/tools.py reads np.Inf; NumPy 2.1.2 removed that spelling. The original stdout and failed receipts remain under runs/p2-acceptance/20260929-cuda/commands/official-ETTh1-h96-s2021/20260929-042133-946371 and the matching official run directory. No optimizer update occurred.

The reference subprocess now restores the exact legacy constant alias np.Inf=np.inf before invoking the official code. No upstream file, dependency version, seed, gradient, hyperparameter or model-selection rule changes. A regression test constructs the real pinned EarlyStopping with NumPy 2 and checks infinite initial loss, zero counter and unset best score. Python tests pass 28/28. This Python attribute failure does not need CPU training diagnosis.

The changed reference wrapper invalidates the original suite tooling identity. A new suite, runs/p2-acceptance/20260929-cuda-v2, passed preflight and is rerunning complete CUDA quality/data/numerical/recovery gates before formal training. The original suite and failed attempt remain intact. Historical v1 inference evidence remains applicable to the unchanged Rust binary and is not counted as a P2 formal matrix item.


## Step 16 milestone: all official ETTh1 profiles passed

The replacement suite runs/p2-acceptance/20260929-cuda-v2 passed preflight, full quality (102 Rust tests and 28 Python tests; zero failed/ignored), complete data, CUDA numerical and ten-case fresh-process recovery gates. The runtime source remains d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81. The measured GPU is NVIDIA GeForce RTX 5090 Laptop GPU.

Official ETTh1 completed all four horizons and all three seeds: 12/12 entries verified for normal termination, full test arrays, source/argument/data identity, best weights, epoch timings, torch allocator peaks and 50 ms device memory samples. The official ETT matrix is 12/48 at this milestone; ETTh2 is next. Rust ETT, D4, cement, independent recomputation and final acceptance remain pending. No CPU formal control, commit, push or archive was performed.


## Step 16 milestone: hourly official matrix complete

Official ETTh1 and ETTh2 each completed all 12 configurations. The replacement suite now contains 24/48 verified official entries; ETTm1 has started. No numerical, coverage, receipt or memory-sampling failure occurred in the replacement suite. Rust ETT and later acceptance stages remain pending.

## Step 16 milestone: official matrix reached 36 of 48

On 2026-09-29, ETTh1, ETTh2 and ETTm1 each completed all twelve official CUDA runs in runs/p2-acceptance/20260929-cuda-v2. Every completed entry passed the runner checks. ETTm1/H720/seed2023 returned exit 0 with standardized MSE 0.487357199 and MAE 0.456740081. The same active process continued with ETTm2/H96/seed2021. No official matrix was restarted.

The continuation reread the CUDA-first project memory and retained the original source, binary, configurations and receipt identities. Read-only review covered TrainingSession cursor/phase/scheduler checks, v2 best-generation and resume bindings, the reference timing adapter, and the D4/cement/recomputation prerequisites. Formal Rust ETT, D4, cement, independent recomputation and final quality remain pending. No CPU training, commit, push or archive occurred.

## Step 16 complete: official CUDA ETT 48/48

The original active official process completed on 2026-09-29 with exit 0 and PASS official CUDA ETT: 48/48. The replacement suite contains all four datasets, all four horizons and seeds 2021/2022/2023. The final matrix check reread every entry, receipt, source/argument/data binding, best checkpoint, full test array, finite metric and performance record. ETTm2/H720/seed2023 completed with MSE 0.413823336 and MAE 0.406825095. No matrix item was replaced with a smoke run.

Step 17 was launched with the same frozen suite and CUDA binary through tools/validation/p1_env.ps1. The shell runs ett-rust and then d4; helper failure stops the dependent command. Rust completion and D4 acceptance are still pending. Cement and later stages have not started.

## Step 17 first formal Rust run passed

Rust ETTh1/H96/seed2021 completed training and best-checkpoint evaluation with exit 0. The commands took 375.749 seconds and 1.188 seconds respectively. The formal validator passed source/binary/config/data identities, normal termination, full split coverage, v2 best/last bindings, finite metrics and performance receipts. Standardized MSE is 0.385795087; MAE is 0.404669445. The entry is runs/p2-acceptance/20260929-cuda-v2/entries/rust-ETTh1-itransformer-cuda-h96-s2021.json.

The Rust ETT matrix is 1/48 at this milestone and seed2022 is running. This single run does not establish D4 acceptance. The official matrix remains complete at 48/48.

## Step 17 milestone: first two Rust profiles complete

Rust ETTh1/H96 and H192 completed all three seeds with verified train/evaluate receipts: 6/48 Rust ETT runs. ETTh1/H336/seed2021 is running. The official matrix remains 48/48. The following comparisons cover only these two complete configurations; the full 32-comparison D4 gate remains pending.

| H | Metric | Official mean | Rust mean | Absolute difference | Original bound | Diagnostic result |
| --- | --- | --- | --- | --- | --- | --- |
| 96 | mse | 0.38577324152 | 0.385861709785 | 8.84682650711e-05 | 0.0077154648304 | True |
| 96 | mae | 0.404095113277 | 0.404201542924 | 0.000106429646866 | 0.00808190226555 | True |
| 192 | mse | 0.442551781734 | 0.442920046087 | 0.000368264353622 | 0.00885103563468 | True |
| 192 | mae | 0.436715473731 | 0.436630864849 | 8.46088827174e-05 | 0.00873430947463 | True |

The P2 model/resume contract is now in .trellis/spec/backend/itransformer-training.md. The backend index, v1 compatibility description and both task context manifests link the implemented contract. task.py validate passed with 15 entries per manifest; document and diff checks passed. research/review.md records the inline code review checkpoint and the remaining formal gates. These documentation changes leave the runtime source and frozen suite identity unchanged.

## Step 17 interruption and retained evidence

After an intentional conversation interruption, unified exec session 23200 was unavailable. A read-only Windows process check found no matching P2 runner or training process. The uncompleted ETTh1/H336/seed2021 attempt retained three epoch records, stdout/stderr and checkpoints. No final train summary or command receipt exists; the exit code and process exit cause are not recorded. Observation: runs/p2-development/interruption-20260930-012346-681073.json.

The completed official 48 and Rust 6 entries are preserved. Resume the same acceptance stage so the runner verifies existing entries and starts a fresh attempt for the missing item. The incomplete attempt is not counted as passed. This interruption does not establish a CUDA failure and does not trigger CPU training.

## Step 17 continuation: existing process and remaining CUDA stages

At 20:31 America/Chicago on 2026-09-29, the existing Rust ETT process remained active. Its shell is PID 43756, created at 2026-09-30T01:24:07.9859469Z; unified exec session 6728 owns the same command. The retried ETTh1/H336/seed2021 completed training and evaluation and passed the formal entry validator. Official coverage remains 48/48; Rust coverage is now 7/48, with H336/seed2022 active. The original interrupted attempt remains intact.

The existing shell still runs ett-rust followed by d4. A separate sequential continuation waits for that exact shell process to exit, checks that all 32 D4 comparisons passed with the original ddof=0 contract, and then calls the existing cement, recompute, final CUDA quality and final acceptance commands in order. Each production command retains its source, binary, configuration and artifact checks. A failed command stops all dependent stages. The continuation does not restart either ETT matrix or create concurrent training.

Execution-only script: runs/p2-development/p2-cuda-v2-after-d4-20260929.ps1. PowerShell parsing passed. The live continuation is PID 40828, unified exec session 14068. Its append-only events, current status and transcript are in runs/p2-development/after-d4-20260929-203052-6712177. The waiting state and the single active ferrots training process were reread after launch. The script uses an exclusive suite continuation lock and verifies the predecessor process creation time. All execution files remain under ignored runs/.

The continuation stops after formal validation. Final report review, model-card status and task sign-off remain pending. No CPU training, runtime-source change, frozen-suite change, commit, push or archive occurred during this continuation.

## Step 17 takeover: identities and completed entries verified

Snapshot: 2026-09-29 20:44 America/Chicago. Completed entry verification finished at 20:42; process identity was checked at 20:44.

Evidence: runs/p2-development/handover-verification-20260929-204227-752639.json; runs/p2-development/process-takeover-20260929-204416-5795602.json.

The existing ETT shell PID 43756 and continuation PID 40828 are alive with exactly the recorded creation times. Their command lines and process tree match the handoff. One training child, PID 62280, runs ETTh1/H720/seed2021; one epoch record exists at the process snapshot. No new training or continuation was started.

suite_context revalidated the runtime source, all frozen configuration/reference/validation/data hashes, both the suite binary path and its hash, the pinned clean official checkout and P1 baseline inventory. The requested target/windows-static binary resolves to the same mbx target path recorded in suite.json; there was no binary switch.

The existing entry validators reread all 48 official entries and nine completed Rust entries, including normal exits, complete windows, best checkpoints, performance and artifact hashes. Rust H96/H192/H336 each contain seeds 2021/2022/2023. All six partial comparisons satisfy the unchanged D4 bound. Full D4 remains pending.

Recorded quality was revalidated: Rust 102 / Python 28, zero failed or ignored tests. CUDA numerics covers nine cases with all 38 parameters and optimizer controls; ten CUDA recovery cases each report maximum tensor error 0. Preflight and full data receipts passed again as read-only checks. No training test was repeated.

Added acceptance.md and partial ETT/cement/performance report surfaces. Updated the model card and task status without changing runtime source, tools, configurations, suite receipts or registry state. Reports explicitly retain all remaining gates and the unexecuted CPU control. Final quality and final inline sign-off remain pending. No commit, archive or push occurred.

## Continuation: cement metric-consumer defect reproduced

On 2026-09-29, the main session verified two additional completed H720 entries and retained their receipts in runs/p2-development/continuation-entries-20260929-205923-365595.json. At 21:05 America/Chicago, the original runner had produced 13 Rust ETT entries and was training ETTh2/H96/seed2022. Both original shell identities remained valid.

Code review identified three incorrect list assumptions in the P2 cement metric consumers. Rust OriginalMetrics and the accepted P1 evidence use named dictionaries. The original reader, aggregator and P1 report paths failed under isolated synthetic checks. The candidate passed 8/8 test methods, zero failures/errors/ignored. All non-cement functions are AST-identical. Files, raw failures, the corrected diagnostic count and candidate diff are retained under runs/p2-development/cement-contract-20260929-210242-974583.

No frozen tool was changed. Applying the correction requires resolving the existing suite and command-receipt hash bindings. The user was asked only about a versioned evidence transition versus a fresh full matrix. Current implementation approval and AC3 approval remain effective. Detailed evidence and the boundary are in research/cement-metric-contract-diagnosis.md.

At 21:08 America/Chicago on 2026-09-29, 14 Rust ETT entries were verified. ETTh1 is complete at 12/12; all eight ETTh1 diagnostic comparisons satisfy the unchanged bound. Full D4 remains pending. Evidence: runs/p2-development/ett-hourly-milestone-20260929-210833-166843.json. The original execution chain remains active; the cement candidate and evidence-transition decision remain pending. No frozen tool, runtime or receipt changed.

## Verification continuation: 2026-09-29 21:20 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 18/48 passed frozen entry validation. 12/12 complete-configuration diagnostic comparisons satisfy the unchanged D4 bound. Full D4 remains pending. The original shell PID 43756 and continuation PID 40828 retain their recorded creation times. Active run: rust-ETTh2-itransformer-cuda-h336-s2021-20260929-211846-941822. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/verification-update.json; runs/p2-development/continued-verification-20260929-211647-8010923/process-latest.json.

The cement candidate passed an additional real-artifact report integration check: nine accepted P1 entries, 18 variables per entry, 108 report rows and 324 baseline metric cells. P2 inputs were synthetic; zero formal CUDA runs were added. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/cement-integration.json. The earlier 8 synthetic methods remain passed. The production tool and old suite/receipts are unchanged. Evidence-transition selection remains pending.

Reports and the model card now show the verified snapshot. The registry remains pending. Final CUDA quality and final inline trellis-check remain open. No duplicate training, CPU training, task restart, commit, archive or push occurred.

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

## First cement epoch: 2026-09-30 05:44 America/Chicago

The active CUDA H15/seed2021 run completed epoch 1: 299,819 training windows, 9,370 optimizer steps, zero skipped empty batches, train MSE 0.10933995531677243 and validation MSE 0.10041589545685217. Training took 9,299.952 seconds and validation took 7.171 seconds; the combined epoch time was 9,307.124 seconds (155.1 minutes). Checkpoint timing was 10.181 seconds. Steady throughput was 32.239 windows/s. Best and last checkpoint directories are present. These observations cover one epoch of an unfinished run; they do not establish final checkpoint validity or add a completed formal entry. Cement remains 0/9; executor PID67188 and training PID25580 remain active. The full matrix, independent recomputation, final quality and final acceptance remain pending. Evidence: runs/p2-development/tooling-transition-20260929-2129/cement-first-epoch-20260930-054416-417122.json.

## Scoped CUDA performance repair: 2026-10-01

Implementation evidence is complete for the user-requested performance repair. Cargo.toml adds opt-level 3 for ferrots-nn, ferrots-train, rand_chacha and ppv-lite86 in the development profile. The dev profile retains debug assertions. No release profile, fast-math flag, algorithm, RNG stream, tensor arithmetic or training setting changed.

The new opt-in crates/ferrots-train/examples/cuda_profile.rs diagnostic uses the real model, masked loss, GradientAccumulator and StatefulAdam with synthetic B32/L96/C18/K5/H15 inputs. It writes profile.json and tensors.safetensors under a new output directory. The comparison script verified 115 bitwise-identical tensors and 12 exact loss values.

A separate real-data stop-after-steps diagnostic verified the updated binary on cuda:0 with the frozen cement H15/seed2021 configuration. The process exited 0 after 128 steps. Its output and GPU samples remain under runs/p2-development/gpu-performance-repair-20260930/real-data-128-20260930-191449-2643853.

Old formal receipts cannot be relabeled as current acceptance because the runtime source and binary identity changed. P2 remains in_progress and the heartbeat remains paused.
