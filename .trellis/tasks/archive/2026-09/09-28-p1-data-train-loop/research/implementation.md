# P1 execution evidence

User approval: 2026-09-28. Inline execution; no sub-agents.

| Step | Validation | Result | Log in runs/p1-development |
| --- | --- | --- | --- |
| 01 | config; check --workspace | 4 passed; check exit 0 | Previous approved-session command receipt |
| 02 | public manifest; fixture generation; --check; privacy | 9/9 hashes; exact fixture bytes; exit 0 | 20260928-064348-7042812.log; 20260928-064358-6316205.log |
| 03 | csv_report | 4 passed, 0 failed/ignored | 20260928-064536-8261213.log |
| 04 | preprocess | 5 passed, 0 failed/ignored | 20260928-064727-0156947.log |
| 05 | scaler; time_features; cache | 3 passed, 0 failed/ignored | 20260928-064921-5931019.log |
| 06 | windows; batches | 3 passed, 0 failed/ignored | 20260928-065044-3457489.log |

Step 02 first attempt failed because Tee-Object -LiteralPath does not support -Append. Changed to -FilePath and reran successfully. Steps 03-06 use the fixed Rust 1.98.0 CPU environment. ETT fixture comparison covers all four datasets, four horizons, three splits and train-only scaler statistics. Formal experiment gates remain pending.

| Step | Validation | Result | Log in runs/p1-development |
| --- | --- | --- | --- |
| 07 | baselines; dlinear_reference, after numerical repair | CPU 3; CUDA 3 passed | 20260928-070100-9175117.log; 20260928-070118-5516071.log |
| 08 | metrics; loss; accumulation; dlinear_update; optim | CPU 11; CUDA 12 passed | 20260928-065922-2819062.log; 20260928-070103-7193790.log; 20260928-070123-5313675.log |
| 09 | checkpoint | CPU 2; CUDA 2 passed | 20260928-070459-8723932.log; 20260928-070513-5515881.log |
| 10 | trainer | CPU 3; CUDA 3 passed | 20260928-070747-9600961.log; 20260928-070802-5350054.log |
| 11 partial | cli_pipeline; doctor; help | CPU 5 passed; both CLI help commands verified | 20260928-071545-1826849.log; 20260928-071603-1575123.log; 20260928-071724-2044054.log |

Step 08 initially failed the L=1 Adam fixture: dense averaging removed a small upstream FP32 seasonal residual. Replaced the matrix with ordered differentiable additions and final division. The unchanged CPU/CUDA 1e-6 update tolerance now passes. Repeated Step 07 forward/gradient checks passed.

Step 09 initially failed on Windows because sync_all used a read-only model file handle. The handle now opens with write access before synchronization. Bundle tests pass, including immutable frozen roundtrip, empty model files, incomplete staging/reference files, latest-reference corruption, checksum/schema/shape/dtype errors, and missing files.

PowerShell consumes a bare -- at a function invocation boundary. The first top-level help attempt reached Cargo help. Quoted/splatted separators now reach ferrots --help, and the implementation commands quote the separator. No Rust/CUDA toolkit was installed or removed.

Step 11 completed. CPU Clippy with -D warnings passed (20260928-072710-9034894.log). The affected data tests passed 9/9; CLI/doctor passed 5/5. CPU release build passed (20260928-072726-2737552.log), SHA256 1f90636f3723cbe2b9e291b625d60a9178e75d85f7a9794159c830d6ce724c9d. CUDA release rebuild passed (20260928-072743-6149646.log), SHA256 6225a7c6b4524f7cc7196068588859900814a74a6f08fbc0ed86bd3d9281f848. Clippy --fix returned zero despite rejecting its own as_chunks suggestion; the cache conversion was corrected manually and normal Clippy was rerun. No warning suppression was added.

Step 12 correction approved by the user on 2026-09-28: Exchange has 7,588 data records, not 7,587. Both the Rust CSV parser and independent Python csv.DictReader count 7,588 unique timestamps. The unchanged file hash matches the manifest. The file has 7,588 newline bytes and no final newline; subtracting a header from that newline count omits the final data record. The corrected 70/10/20 split is 5,311/760/1,517; validation [5215,6071), test [5975,7588), including L=96 history. Evidence: 20260928-073104-1127752.log. Source data, split ratios and D4 tolerances are unchanged.

Step 12 passed (20260928-073231-2515248.log), suite runs/p1-acceptance/20260928-0734. ETT 16 combinations; Exchange 7,588 records; cement raw counts, 48 segments, 37/4/7 splits, all 18 scaler columns, all three horizons and every window audit passed. Fresh cement command 19.387 s; repeated command 1.013 s, parse_count=0. Exact internal timings and receipts are in data-gate.json.

Step 13 passed (20260928-073414-8778812.log). CPU-only ETTh1 H96 seed2021 used 8 epochs, best epoch5, patience3; train/evaluate/predict commands all returned zero. All test windows and 1,871,520 prediction keys were checked. Standardized MSE 0.396696223, MAE 0.411611924. The full-precision result and artifact hashes are in the suite entry.

Step 14 passed: all 48 new official CUDA runs completed with exit 0 (20260928-073602-6920527.log). Every split count, FP32 prediction shape, finite output, upstream/data/seed identity and raw NumPy metric file was checked. Entries and command receipts remain in runs/p1-acceptance/20260928-0734. No historical smoke run was imported.

The original Step 15 attempt failed. All 48 Rust CUDA runs completed (20260928-080305-5557961.log); D4 was 31/32 (20260928-081846-8556564.log). Diagnosis and the approved RNG-contract correction are in research/d4-diagnosis.md. No acceptance bound, seed, source dataset or training hyperparameter was changed.

Step 15 passed after the approved correction. Suite runs/p1-acceptance/20260928-0842-rng reran data acceptance and CPU closure, verified and linked all 48 original formal official runs, and ran 48 new Rust CUDA cases. D4 passed 32/32 (20260928-085719-1808282.log). ETTm2/H336 MSE mean difference is 7.643922711464413e-8 against the unchanged 0.009017184956695816 bound.

Step 16 first attempt exposed a validator defect: the resolved cement missing-value configuration includes the schema default tokens, omitted from the input TOML. The validator now inserts that explicit default before exact comparison. Product sources and binaries are unchanged. The failed attempt is retained (20260928-085906-2423402.log); the corrected full 18-run matrix continues.

Step 16 completed: all 18 cement runs passed (20260928-090030-5376879.log). LastValue, SeasonalNaive and DLinear use all three horizons on CPU and CUDA. The report includes 324 model/horizon/device/variable rows. DLinear completed 10/10 epochs at H15, 9/9 at H30 and 5/5 at H60 on CPU/CUDA.

Step 17 completed: full DLinear CUDA H60 three-variable export and independent recomputation passed (20260928-220456-4815692.log). All 18,574,560 rows and 103,192 origins were checked. Six metric differences are below 1e-6; maximum 6.430411758628907e-13. Initial Python regression was 7/7; the final gate adds two rejection tests and passes 9/9. CPU and CUDA CLI regression each passed 5/5 (20260928-220854-7728175.log and 20260928-220937-1965788.log).

Step 18 completed: all 13 quality commands passed in quality/20260928-223709-6031022 under the corrected formal suite. CPU workspace tests passed 65/65, CUDA 71/71 and Python 9/9, with zero failed/ignored tests. fmt, both builds, both Clippy -D warnings, cuda:0 smoke, actual cuda:9 rejection, context validation, privacy and diff checks passed. The first runner attempt stopped because silent fmt output left no log file; explicit empty-log creation fixed the runner and all 13 commands were rerun.

Final formal evidence verification passed (20260928-223832-3515904.log). The source SHA256 remains 468482268ffbbf6d69f35391be468aa6ae2de64e67df38f219f28bc0d1660cca. See acceptance.md and review.md for full R1-R17, AC1-AC10, D4 and X1-X6 mapping. Backend specs and context lists were updated. P1 finish-work remains pending after the local commit.

Step 19 completed: local implementation commit 4361656b0129191ff5ee2f579718b9edc96b1b08. Staged privacy and diff checks passed. All 41 runtime files retain the exact raw SHA256 inventory from formal acceptance; each normalized Git blob matches the commit. The same quality input/log fingerprints still validate. See ignored suite commit-verification.json. No push occurred. P1 remains in_progress until the explicitly deferred finish-work/archive/journal workflow.
