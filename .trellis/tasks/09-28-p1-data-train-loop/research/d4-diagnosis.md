# D4 failure diagnosis and proposed correction

Date: 2026-09-28. The original Step 15 run failed. The approved correction and complete rerun below passed 32/32. No numerical bound or seed was changed.

## Formal result

Both new CUDA matrices completed: 48 official runs and 48 Rust runs. D4 passed 31/32 checks. The failed check is ETTm2, H336, standardized MSE.

| Quantity | Value |
| --- | --- |
| Official seeds 2021/2022/2023 | 0.38492700457572937 / 0.3751393258571625 / 0.3844631016254425 |
| Rust seeds 2021/2022/2023 | 0.37250484936439526 / 0.3691407054567656 / 0.37401406033620427 |
| Official mean | 0.38150981068611145 |
| Rust mean | 0.3718865383857884 |
| Official population std | 0.00450859248 |
| Allowed absolute difference | 0.00901718496 |
| Observed absolute difference | 0.00962327230 |

The original Rust MSE was lower, but D4 is a two-sided reproduction bound. The original result remains a failure. Its full-precision source is runs/p1-acceptance/20260928-0734/d4.json. The current comparison table, docs/results/p1-dlinear-ett.md, reports the corrected complete rerun.

## Evidence

1. Data fingerprints, window counts, scaler parameters, model/optimizer settings, dtype, learning-rate sequence, actual epoch coverage and checkpoint identities passed the formal validators.
2. Cross-loaded all three official checkpoints into diagnostic Rust bundles. MSE/MAE differences from the official NumPy results were below 4e-7. The official result reduction uses FP32; the Rust report uses f64 accumulation.
3. Loaded all three Rust checkpoints into the unmodified pinned TSLib DLinear class and evaluated every test element. The f64 metrics agreed within 2e-15.
4. Exported the Rust initial weights and all epoch batch orders. Replayed the three Rust training trajectories with the pinned TSLib DLinear class and torch Adam, using the actual Rust epoch learning rates and element-weighted validation. This diagnostic did not enter the formal matrix.
5. Across all 15 replayed epochs, the maximum parameter difference was 1.53e-7. The maximum final test MSE difference across the three seeds was below 5e-9. The selected validation minima matched.

Evidence directories:

- runs/p1-acceptance/20260928-0734/diagnosis/20260928-082225/cross-evaluation.json
- runs/p1-acceptance/20260928-0734/diagnosis/replay-input/
- runs/p1-acceptance/20260928-0734/diagnosis/replay-20260928-082709/replay.json
- logs 20260928-082221-9047747.log, 20260928-082555-8905631.log and 20260928-082707-2490284.log under runs/p1-development/

The evidence identifies the random initialization and training order as the cause of the different fitted checkpoints. It does not isolate their individual contributions. No defect was found in the checked forward, data, optimizer or evaluation calculations.

## Approved correction

On 2026-09-28 the user approved: align the random sequences, keep the threshold unchanged, and continue. This approval covers the implementation boundaries and full rerun below.

The original design did not claim identical cross-framework random values. The original code used ChaCha8 for model initialization and seed-plus-epoch window shuffling. The recorded approval changes that reproducibility mechanism.

Proposed correction: reproduce the pinned torch CPU random-generator semantics for baseline parameter initialization and the official single-worker training sampler. Derive each epoch's order from the seed and model initialization draw count. Keep Rust validation order fixed, keep effective-element weighting, and do not read or iterate the test split during training. Model the reference loader's RNG consumption independently from data access.

Implementation boundaries:

- core: a tested pure Rust reference RNG helper, with explicit version and upstream attribution. No Python runtime dependency.
- nn: baseline initialization uses the documented reference draw order, including draws for weights overwritten by DLinear's 1/L constants.
- data/train: deterministic reference-compatible training order; validation/test ordering remains fixed.
- tools/reference or validation: generate synthetic RNG/initialization/permutation fixtures from the installed pinned torch version; verify all three fixed seeds, multiple lengths, first/middle/later epochs and individual/shared initialization.
- tests/docs/spec: preserve the existing numerical tolerances; document the RNG contract.

Validation after approval: rebuild both binaries; rerun affected unit tests, CPU closure and the full Rust CUDA 48-run matrix. Retain the current failed suite. Official runs may be reused only after verifying their unchanged code/data/arguments/environment identities and artifact hashes; never import historical smoke runs. D4 still requires all 32 checks. Proceed to cement only after Step 15 passes.

Unchanged: seeds 2021/2022/2023, ddof=0, all model/training hyperparameters, complete datasets, D4 formula, privacy rules, CPU/CUDA requirements, and no push.

## Implemented correction and rerun

The pure Rust MT19937 helper now matches the pinned CPU integer/uniform/randperm semantics. Baseline stores the post-initialization RNG state. WindowIndex maps its reference permutation onto train-only indices. Validation and evaluation remain ordered. The Trainer never requests a test batch.

Synthetic fixtures verify 4 primitive seeds, 15 Linear/DLinear initialization cases and 150 loader epochs. Actual torch DataLoader state agrees with five int64 draws per epoch. Parameter bits, full permutations and sampler orders match. CPU tests passed for the helper, initialization, batches, windows, Trainer and DLinear update. CUDA-feature initialization, Trainer and single-step tests passed. Release binaries were rebuilt.

The new formal suite is runs/p1-acceptance/20260928-0842-rng. It reruns data validation, CPU closure and all 48 Rust CUDA cases. The 48 official results from the original formal suite can enter only through import-reference, which validates code, arguments, seeds, environment, dataset/config/reference identities, receipts, log hashes and artifact hashes. Original failure evidence is retained. The full new Rust matrix completed 48/48 and D4 passed 32/32. Evidence: new suite d4.json; logs 20260928-084159-1494760.log and 20260928-085719-1808282.log. The original failed suite remains unchanged.
