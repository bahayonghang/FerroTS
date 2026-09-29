# P1 inline Trellis quality review

Result: PASS. Reviewed on 2026-09-28 (America/Chicago). The review ran inline under the user's explicit no-sub-agent constraint. No unresolved violation was found in the P1 scope.

## Scope and evidence identity

Reviewed the complete change set in ferrots-core, ferrots-data, ferrots-nn, ferrots-train and ferrots-cli, plus configurations, public/synthetic fixtures, validation tools, model/data documentation and task records. Read the P1 PRD, design and ordered implementation plan; the parent D4 and X1-X6 contracts; backend index, training-foundation, data-pipeline and training-loop specs; and cross-layer/code-reuse guides.

Formal suite: runs/p1-acceptance/20260928-0842-rng. Runtime source SHA256: 468482268ffbbf6d69f35391be468aa6ae2de64e67df38f219f28bc0d1660cca. The recorded build HEAD f840dfac56fcbc88a2252cbf35e03c867ee90892 predates P1; dirty=true and the source inventory identify the tested implementation. The final local commit must contain the same runtime source contents.

Final evidence verification returned zero in runs/p1-development/20260928-223832-3515904.log. The final gate reread receipts, command logs, configurations, artifact hashes, official NumPy metrics, Rust summaries/evaluations, checkpoints, full matrix coverage, independent metric checks and quality records. See acceptance.md and the ignored suite's final.json.

## Requirement review

| Requirement | Implementation and review | Verification |
| --- | --- | --- |
| R1 CSV | data/csv_source.rs selects exact UTF-8 names, handles configured timestamps and numeric types, rejects malformed input without logging cell values | csv_report tests; formal Exchange and cement data gates |
| R2 DataSpec | core/config.rs rejects unknown fields, invalid dimensions, unsupported model inputs and unnamed targets; CLI uses explicit optional overrides | core config and CLI pipeline tests |
| R3 DatasetReport | Raw counts precede sorting/deduplication; prepared counts and window audit remain separate | csv_report tests and data-gate.json |
| R4 grid | preprocess.rs aggregates per-column means using the configured sample minimum | preprocess aggregate/date test; full cement statistics |
| R5 segments | Pre-fill running status defines segments; history/gap/targets remain in one span | 5/6-minute, 29/30-minute and window boundary tests; all formal windows audited |
| R6 cache | cache.rs hashes source content and protocol/version; verifies table and metadata before reuse | cache invalidation/corruption test; second cement load parse_count=0 |
| R7 time features | time_features.rs implements h/t calendar values with the pinned TSLib convention | real ETT fixture and calendar test; error <1e-6 |
| R8 missing values | Observed, available and imputed masks remain distinct; fill age uses the last observation and never crosses a segment | preprocess fill-clock and complete-span tests; approved cement policy |
| R9 scaler | scaler.rs fits training rows only and records named columns, counts and constant scales | scaler roundtrip/train-only test; all 18 cement columns checked |
| R10 windows | windows.rs requires complete target split membership and continuous legal spans; HostInput excludes supervision | ETT 16 combinations; gap/segment/tail tests; fixed validation/test order |
| R11 loss | loss.rs sanitizes masked values; accumulation.rs weights gradient sums by valid labels and handles the tail group | loss/accumulation tests; all-empty training-label regression |
| R12 baselines | baselines.rs implements four models, shared/individual projections, ordered moving averages and independent frozen weights | CPU/CUDA forward, gradient, Adam and RNG fixtures; complete formal matrices |
| R13 Trainer | Trainer iterates train/validation only, validates frozen untracked outputs and selects strict best with patience | Trainer access/graph/tie tests; formal epoch/window/LR records |
| R14 bundles | checkpoint.rs validates five-file bundles before load and publishes immutable generations with numbered best/last references | CPU/CUDA corrupt file/schema/shape/dtype and interrupted-publication tests |
| R15 metrics | core metrics accumulate f64 sums per valid element; evaluation uses raw-unit targets and streams origin-keyed CSV/gzip rows | metrics and CLI tests; all 18 cement variables; independent 18,574,560-row recomputation |
| R16 CLI | inspect/train/evaluate/predict propagate errors; evaluation checks original fingerprint; prediction reuses the saved scaler | CPU full ETTh1 closure; CPU/CUDA CLI pipeline and regression tests |
| R17 configs | configs/reference contains all four ETT configurations; configs/engineering contains cement-twin-v1 | exact resolved-config comparisons in all formal runs |

Paths in this table refer to crates/ferrots-<layer>/src unless a different location is explicit.

## Acceptance and parent contracts

| Criterion | Result |
| --- | --- |
| AC1 | ETT four datasets x four horizons x three splits match the pinned official lengths, including all required ETTh1/ETTm1 cases |
| AC2 | ETT time features pass the strict <1e-6 comparison |
| AC3 / D4 | Official CUDA 48/48 verified formal runs and fresh Rust CUDA 48/48; all 32 two-sided checks pass for seeds 2021/2022/2023 and ddof=0 |
| AC4 | Exchange has 7,588 records and distinct timestamps; dates and unchanged source hash match the approved correction |
| AC5 | Cement raw statistics, 48 segments, 37/4/7 segment split, 303,009 scaler-fit minutes and all H15/30/60 windows pass |
| AC6 | Three cement models x three horizons x CPU/CUDA = 18/18 full runs; all 18 variables have original-unit statistics |
| AC7 | Independent Python reads all 18,574,560 selected rows and 103,192 origins; six MAE/RMSE differences pass <1e-6; maximum 6.430411758628907e-13 |
| AC8 | Fresh cement command 19.397 s, repeated command 0.874 s; repeat parse_count=0; no speed threshold was added |
| AC9 | Full regression covers tail/B1/L1/constants, all-missing labels, duplicates/segments, column reordering, invalid targets, missing files, corrupt bundles and unavailable CUDA |
| AC10 / X1 | Public/synthetic fixture provenance is documented; privacy guard passes; private CSV/cache/predictions stay in ignored data/ and runs/ |
| X2 | Both public ETT and private cement experiments are complete |
| X3 | Candle, TSLib, PyTorch RNG source and public data use fixed identities; manifests/model card/NOTICE preserve provenance |
| X4 | Reference and engineering protocols remain explicit; cement's four approved adjustments do not alter ETT defaults |
| X5 | Native Windows RTX 5090 Laptop validation follows docs/environment.md; no unmeasured platform is declared supported by these results |
| X6 | Receipts bind resolved config, data, seed, device/environment, binary and dirty source snapshot; official reuse verifies original formal artifacts |

The original 31/32 D4 suite remains intact. research/d4-diagnosis.md records the cause investigation, shared-weight/order replay and user-approved RNG correction. The ETTm2/H336 MSE difference is now 7.643922711464413e-8 against the unchanged 0.009017184956695816 bound. The correction preserves weighted Rust validation and test isolation.

## Full quality gate

Runner: tools/validation/p1_quality.ps1. Complete record directory: runs/p1-acceptance/20260928-0842-rng/quality/20260928-223709-6031022. All 13 commands satisfy their expected exit status and log/content fingerprints.

| Check | Result |
| --- | --- |
| cargo fmt --all --check | Exit 0 |
| CPU workspace build and Clippy -D warnings | Exit 0 / 0 |
| CPU workspace tests | 65 passed, 0 failed, 0 ignored |
| CUDA workspace build and Clippy -D warnings | Exit 0 / 0 |
| CUDA workspace tests | 71 passed, 0 failed, 0 ignored |
| cuda:0 doctor training smoke | Exit 0; six parameters have gradients and updates |
| cuda:9 doctor | Expected exit 1; actual CUDA_ERROR_INVALID_DEVICE during device initialization |
| Python unittest | 9 passed, 0 failed, 0 ignored |
| Trellis context validation | Exit 0 |
| Private-data guard | Exit 0; repeated after staging |
| git diff --check | Exit 0; staged check repeated before commit |

The first quality-runner attempt stopped after successful fmt because a command without output did not create a Tee-Object log. The runner now creates the empty log before command execution. The complete 13-command suite was rerun. No product source, numerical bound or experiment identity changed for this fix.

MSVC emits the previously documented localized import-library text as linker_messages. No LNK4098 or warning suppression was added. Both normal Clippy -D warnings commands return zero.

## Cross-layer and scope findings

- Traced CSV -> prepared data -> training-only scaler -> legal windows -> HostBatch -> ModelInput/Supervision -> loss/metrics -> checkpoint -> evaluate/predict. Checked column names, masks, shapes, split identities and data fingerprints at the boundaries.
- Dependency direction remains cli -> train -> nn -> core and data -> core. Core/data do not gain Candle or Python runtime dependencies. P0 optimizer state is reused; the learning-rate setter preserves moments and counters.
- Model evaluation snapshots use detach plus copy. Trainer does not iterate test data. Reference sampler advancement consumes RNG draws without accessing test rows.
- Scanned production code for debug instrumentation, TODO/FIXME and warning suppressions. None were introduced. Epoch/cache output contains required statistics. Standalone D4 diagnosis/replay tools preserve the failure investigation and do not alter production execution.
- Backend data/training specs and model/data cards reflect the implemented contracts. A stale general RNG sentence in the design was clarified to restrict equality claims to the pinned version and checked cases.
- No unrelated cleanup, toolchain installation/removal, file deletion, sub-agent, push or remote CI operation was performed.

P1 bundles remain inference-only. Training resume and later model/platform phases remain outside P1. Step 19 performs the local commit and source-to-commit verification. P1 finish-work, archive and journal remain pending for the next handoff.
