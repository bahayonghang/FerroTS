# P1 final acceptance

Suite: runs/p1-acceptance/20260928-0842-rng. Result: PASS.
Runtime source SHA256: 468482268ffbbf6d69f35391be468aa6ae2de64e67df38f219f28bc0d1660cca. Build HEAD: f840dfac56fcbc88a2252cbf35e03c867ee90892; dirty=True. The build HEAD predates the uncommitted P1 source; the source snapshot identifies the tested implementation.

| Gate | Result |
| --- | --- |
| Data | ETT 16 combinations; Exchange 7,588 records; cement raw/scaler/segments/windows and zero-parse cache hit |
| ETT CPU | Complete ETTh1 H96 seed2021 train/evaluate/predict |
| ETT CUDA | Official 48 verified formal runs + 48 fresh Rust runs |
| D4 | 32/32, unchanged two-sided bounds, ddof=0 |
| Cement | 18/18 full runs, all 18 original-unit variables |
| Independent predictions | 18,574,560 rows; 103,192 origins; 6/6 metrics below 1e-6 |
| cpu-test | 65 passed; 0 failed; 0 ignored |
| cuda-test | 71 passed; 0 failed; 0 ignored |
| python-tests | 9 passed; 0 failed; 0 ignored |
| Maximum recompute difference | 6.4304117586289067e-13 |
| Quality | fmt, both builds and Clippy -D warnings, full workspace tests, cuda:0 smoke, real cuda:9 rejection, Python tests, task validation, privacy and diff checks |

## Evidence and decisions

- Complete per-seed D4 results: docs/results/p1-dlinear-ett.md.
- Complete 18-variable cement statistics: docs/results/p1-cement-baselines.md.
- Original 31/32 D4 failure, cross-loading diagnosis and approved RNG correction: research/d4-diagnosis.md. The original suite is retained.
- Formal official reuse is explicit in reference-reuse.json and each entry; original code, arguments, data, environment, logs and artifact hashes were checked.
- Exchange count correction and all four cement protocol adjustments remain approved. Seeds, hyperparameters and numerical bounds were not relaxed.
- Receipts and raw artifacts remain under ignored runs/. No private rows or prediction rows are included in tracked reports.
- P1 inference bundles do not support training resume. P2, P3, P4, push and remote CI remain outside this task. P1 finish-work remains pending after the local commit.
