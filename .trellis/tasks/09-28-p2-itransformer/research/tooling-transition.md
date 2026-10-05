# P2 cement tooling transition

Status: user selected versioned evidence migration on 2026-09-29, America/Chicago. Implementation stays in the main session. No acceptance threshold or training profile changes.

## Scope and identity

The original suite remains runs/p2-acceptance/20260929-cuda-v2. The planned successor is runs/p2-acceptance/20260929-cuda-v3. The successor does not exist until the complete original ETT matrix and D4 pass. Runtime source remains d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81. The original suite.json, failed attempts, interrupted attempts, checkpoints and command receipts remain unchanged.

The correction changes the three cement consumers to use named metric dictionaries. The provenance extension changes suite_context, verify_receipt and require_gate and adds imported_context. Training execution, model profiles, configuration validation, epoch validation, official validation, Rust entry validation, thresholds and performance measurement remain AST-identical. A compatibility check rejects any other function, constant or import change.

## Strict transition sequence

1. Preserve the original ETT/D4 process. Complete all 48 official and 48 Rust ETT entries. Require all 32 original D4 comparisons to pass.
2. With the original frozen validator still installed, run the full parent identity, quality, prerequisite, matrix, receipt and artifact checks. Save an immutable audit, exact old validator copy, entry inventory, prerequisite hashes and receipt inventory under runs/.
3. Install the scoped correction, provenance support and synthetic regression tests. Preserve the previous files and their hashes. No Rust source or binary change is permitted.
4. Revalidate the parent evidence and the precise allowed validator diff. Create a new suite.json with current tooling hashes and a bound parent evidence inventory. Copy entry and prerequisite JSON bytes exactly; retain original absolute artifact and receipt paths.
5. Accept an imported receipt only when its exact path/hash is in the bound audit inventory, its original suite/source identities match, and every original artifact hash passes. Imported prerequisite gates must be exact copies of the bound parent gates. Missing, altered, unlisted, nested or identity-mismatched evidence fails.
6. Run current full CUDA quality, recompute the complete D4, complete cement9, independently recompute all 18,574,560 prediction rows, rerun final CUDA quality, then run final acceptance and the main-session final review.

The parent quality evidence is retained. The successor runs current quality because validation code and test inputs change. The complete training matrix contains 105 measured runs; reuse of an unchanged parent training run never adds a second run count.

## Process handoff

At 2026-09-29 21:34 America/Chicago, idle continuation PID40828 was identified by creation time, command line and waiting state. Its descendants contained only conhost.exe. The main session terminated that idle waiter before it could enter the defective cement path. The original ETT/D4 PID43756 and its training child remained alive. The original waiting status file is historical and does not prove the stopped process is alive.

Handoff evidence:

- runs/p2-development/tooling-transition-20260929-2129/waiting-executor-handoff-20260929-213435-9239355.json
- runs/p2-development/tooling-transition-20260929-2129/waiting-executor-handoff-20260929-213435-9239355-after.json

The staged implementation is under runs/p2-development/tooling-transition-20260929-2129. Production frozen files are unchanged at this checkpoint. The active ETT matrix must not be restarted.

## Candidate checks

Sixteen provenance and compatibility tests passed with zero failures, errors or ignored tests. Eight cement consumer tests also passed. Three deliberate mutations to training code, model profile and upstream identity were rejected. These synthetic tests do not count as formal CUDA training or final quality.

Evidence: runs/p2-development/tooling-transition-20260929-2129/tests-20260929-213534-767826/result.json. Complete repository Python regression validation is in progress; its result will be recorded separately.

The early cement candidate identity.json records an earlier candidate hash. The later candidate was tested and fingerprinted by review/integration artifacts. The transition snapshot and subsequent test result manifests record the exact candidate bytes used. Historical candidate manifests are not overwritten.

## Full candidate regression and successor executor

All 52 repository and staged Python tests passed with 0 failures, 0 errors and 0 ignored tests. Evidence: runs/p2-development/tooling-transition-20260929-2129/python-all-20260929-213608-755737/result.json.

Replacement executor PID32276, session78548, waits for the exact original PID43756. Its parent-suite lock is exclusive. Script: runs/p2-development/tooling-transition-20260929-2129/continue-versioned.ps1. State and append-only events: runs/p2-development/tooling-transition-20260929-2129/executor-20260929-213907-5815866. The stopped old waiter and its conhost were absent before the replacement launched. The new executor checks original D4, audits the parent before installing tools, validates staged hashes, creates the successor suite, then runs current quality, D4, cement, recompute, final quality and final. A failure stops dependent phases. Main-session final review remains required.

## ETT and migration checkpoint: 2026-09-30 02:54 America/Chicago

Official CUDA ETT 48/48 and Rust CUDA ETT 48/48 completed with normal exits. Both original and successor D4 passed 32/32 with ddof=0, zero failed comparisons and unchanged thresholds. The complete parent audit revalidated all 96 entries, prerequisite gates, quality and original receipts. Successor suite runs/p2-acceptance/20260929-cuda-v3 contains 96 byte-identical imports with bound original receipt paths. The tested cement/provenance correction is installed. Runtime source, CUDA binary, profiles, seeds and tolerances remain unchanged. Current successor quality passed Rust 102 and Python 52, with zero failed or ignored tests. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-transition-checkpoint-20260930-025430-462084.json.

PID67224 stopped after ETT and original D4 passed. Its final error was MSVC initialization failed; retained stderr reports The input line is too long. Repeated initialization expands PATH: an isolated reproduction passed from lengths 4045 and 5802, then failed from length 7559. Two fresh-process probes passed with the required Rust 1.98.0, CUDA 13.3 and static-CRT flags. No CUDA failure was established. The frozen environment helper remains unchanged. Evidence: runs/p2-development/tooling-transition-20260929-2129/msvc-path-reproduction-20260930-024540-7128072/diagnosis.json and runs/p2-development/tooling-transition-20260929-2129/isolated-env-check-20260930-024757-6311963/result.json.

Replacement executor PID73220 was created at 2026-09-30T07:48:38.2326413Z. It keeps an uninitialized parent environment and creates a new child process for every stage. Python stages load p1_env once. Quality stages let p2_quality load the environment once. Script: runs/p2-development/tooling-transition-20260929-2129/continue-isolated-20260930-0246.ps1; SHA256 fd28baca7f56f8fc9c6235d5ac50380d5354110b3a42ca539b69817782040f8c. Ownership remains in active-executor.json. No ETT training was repeated. Cement is active; independent recomputation, final quality, final acceptance and final main-session sign-off remain pending. P2 remains in_progress. No commit, archive or push occurred.

## Cement interruption and recovery: 2026-09-30 03:11 America/Chicago

Executor PID73220 and training PID38688 exited between the 02:52:38 and 02:56:15 observations on September 30, 2026. The first H15/seed2021 attempt has zero completed epochs and no final command receipt. The exit code and cause remain unknown. The interval overlaps a host application restart; causality was not established. No CUDA failure was established. Original logs and hashes remain in runs/p2-development/tooling-transition-20260929-2129/host-interruption-20260930-030009-2678822/interruption.json.

After verifying that all prior P2 process trees had ended, the main session launched executor PID67188 at 2026-09-30T08:06:42.4105976Z through Windows CIM. The observed parent is WmiPrvSE.exe. The executor was verified alive after the launch command returned; survival through an application restart was not tested. The executor retains an uninitialized parent environment and creates one child environment per stage. Only cement, independent recomputation, final CUDA quality and final acceptance are scheduled. ETT96, the audited versioned migration, current quality and D4 were reused.

At 2026-09-30T03:11:56.2431998-05:00, cement has 0/9 complete entries. Active run: rust-cement-itransformer-cuda-h15-s2021-20260930-030647-900688; training PID25580, completed epoch records 0. Current owner: runs/p2-development/tooling-transition-20260929-2129/active-executor.json. Checkpoint: runs/p2-development/tooling-transition-20260929-2129/cement-recovery-checkpoint-20260930-031156-482696.json. The task remains in_progress. Independent recomputation, final quality, final acceptance and final main-session Trellis sign-off remain pending. No CPU training, commit, archive or push occurred.
