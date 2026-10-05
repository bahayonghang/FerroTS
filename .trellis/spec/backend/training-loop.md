# P1 baseline training and evaluation contracts

## Scope and interfaces

Apply this spec to Baseline, Trainer, loss/accumulation, checkpoint bundles, metrics and train/evaluate/predict CLI behavior. Keep cli -> train -> nn -> core and data -> core dependencies. P0 optimizer state APIs remain governed by training-foundation.md.

ModelInput contains history and optional time features, with no supervision field. DeviceBatch keeps Supervision separate. Baseline exposes named parameters, forward(), freeze_for_eval() and an immutable post-initialization sampling_rng(). LastValue and SeasonalNaive have no trainable parameters. Their fit path publishes real inference bundles without optimizer steps or invented training epochs.

## Numerics and reproducibility

- Production tensors and parameters are FP32. Accumulate reported metrics in f64 by effective element count. Original-unit metrics use unscaled protocol targets and inverse-scaled predictions.
- DLinear follows TSLib 4e938a1767106324dd753b2a44832bf870a0252e. Centered odd moving averages replicate boundaries. Add offsets in order and divide once. A dense averaging matrix changes tiny FP32 seasonal residuals; overlapping Candle avg_pool lacks the required backward path.
- Linear/DLinear CPU initialization matches torch 2.14.0+cu130, commit 08187d9e0fba026dc8217405802ab5381dc88d90. Consume the initial random weight draws even when DLinear replaces weights with 1/L. Individual DLinear creates seasonal then trend projections inside each channel iteration.
- Core ReferenceRng implements the pinned CPU MT19937, uniform and randperm semantics. For zero-based epoch e, clone the post-initialization state, advance five int64 draws per previous epoch, consume the train loader base seed, then use the next draw for the private permutation generator. The five reference draws account for train/val base+sampler seeds and the test base seed. RNG advancement never reads validation or test data. P0 dropout keeps its separate ChaCha8 contract.
- Synthetic fixtures check parameter bits, integer/permutation sequences, three fixed training seeds, shared/individual models and ten sampler epochs. No Python dependency enters production training.

## Loss, training and model selection

Sanitize invalid targets before arithmetic. Masked MSE is sum of valid squared errors divided by valid count. Accumulate gradients using valid counts across microbatches and process the final partial group. An empty group skips updates and is recorded. An entire train epoch or validation pass with no valid labels fails.

Trainer visits train and validation only. Validation uses independent copied/detached weights, fixed order and element-weighted loss. Reject tracked validation outputs. Select best by strict loss decrease; ties count toward patience. The checkpoint load after fit must select the recorded best epoch.

type1 runs at epoch end: epochs 1 and 2 use the initial rate; epoch 3 uses half, followed by successive halves. The setter preserves Adam moments and counters. Official TSLib comparisons retain official shuffled validation, batch-mean validation and per-epoch test logging; these approved differences never relax D4.

## Checkpoints and outputs

Bundles are immutable generations. Each contains model.safetensors, scaler.json, data-schema.json, resolved-config.toml and manifest.json. The manifest records file sizes/hashes, parameter names/shapes, model/protocol versions, data/config/source identity, seed, device and build provenance. best/last are append-only numbered references published after the complete generation. A damaged latest reference fails; do not silently choose an older generation.

Validate the complete bundle before loading parameters. P1 v1 manifests have inference_only=true and do not contain optimizer/scheduler/RNG training state. P2 adds v2 restoration and a resume command under itransformer-training.md. V1 remains inference-only and is explicitly rejected by resume. P0's optimizer-state roundtrip tests cover the lower-level optimizer API.

evaluate requires the checkpoint's original data fingerprint. predict can use a schema-compatible new file and the saved scaler. Export column filtering does not change complete evaluation metrics. CSV/CSV.gz columns are run_id, series_id, forecast_origin, target_time, horizon, target_name, y_true, y_pred, valid. Stream rows; do not materialize the entire prediction matrix. Future outputs have no truth labels and valid=false.

## Required gates

Run CPU/CUDA workspace fmt/build/Clippy -D warnings/tests and real cuda:0 smoke/cuda:9 rejection. Maintain loss, accumulation, update, Trainer, checkpoint, CLI pipeline and error-matrix tests. Formal P1 acceptance requires official/Rust ETT 48+48 runs, all 32 two-sided D4 checks, cement 18 runs, and six independent original-unit metric differences below 1e-6 from all 18,574,560 selected prediction rows. Partial tests do not replace formal gates. Reuse formal results only after full identity and artifact verification; never import historical smoke runs.
