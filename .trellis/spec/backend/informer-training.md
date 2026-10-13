# Informer training and persistence contracts

## Scope

Read with training-foundation.md, data-pipeline.md, training-loop.md and itransformer-training.md before changing Informer, sparse operators, model buffers or evaluation sampling. Core owns the strict Informer schema; NN owns model/ops/buffer schemas; Train owns state progression and atomic persistence; CLI reuses those boundaries.

## Model

The pinned source is TSLib 4e938a1767106324dd753b2a44832bf870a0252e. Encoder and decoder self/cross attention all use ProbAttention, retaining its BHLD context view and unapplied attention-weight dropout. Inputs are FP32 history [B,L,C], history-time [B,L,K] and future-time [B,H,K], with K=4/5. Decoder numeric futures are zeros; no supervision enters ModelInput. Require gap=0 and 0<label_len<=L, positive even d_model divisible by heads, positive factor/layers and finite dropout in [0,1).

Scores and causal masks are [B,heads,u,Lk]. Selection is discrete; selected Q/K/V values retain gradients. Q/K/V do not move to CPU. PrefixSum has linear-storage scan/suffix VJP. ReplaceRows masks overwritten old-value gradients; MaxPool1d uses first-maximum argmax and overlapping gradient accumulation. ELU masks nonnegative exponent inputs before exp so an unselected overflowing branch cannot create NaN gradients.

## Buffers and identities

ForecastModel buffers() returns independent copies. Informer buffers include positional encoding, BatchNorm running mean/variance and U32 batch counters. They are never Adam parameters. Snapshot, checkpoint load/save, resume and freeze share the model-owned complete buffer schema. Validate before applying; missing/extra/shape/dtype/device/nonfinite errors reject the set. V1 stateless bundles reject Informer; v2 contains its required buffers. Baseline/iTransformer require empty buffer maps.

Validation's forward id derives from persisted validation access counts; evaluation uses split/batch identity and future prediction records its fixed identity. These domains do not advance the training stream. CUDA tensor recovery uses atol=1e-6/rtol=0; integer counters and random state match exactly. Batch order and initialization streams are documented rather than claimed bitwise identical to torch Informer's stochastic streams.

Target selection must accept a noncontiguous decoder slice; materialize contiguity at the common CUDA index_select boundary.

## Validation

Keep p3_ops, p3_informer, p3_complexity, p3_buffers and fresh CLI Informer recovery tests. Fixtures contain synthetic data only, fixed upstream weights/samples and all true MSE gradients plus a connectivity probe.

The PrefixSum AC2 boundary preallocates input/cotangent and measures operator forward plus the exact production CustomOp1::bwd. Include output, returned gradient and operation workspace, excluding an unrelated outer scalar loss and GradStore. Record native async-pool high water after synchronization, retain both output and gradient, and enforce <=4 input sizes at n=1,000,000. The separately recorded full-autograd multiply/sum diagnostic is larger and must not be described as satisfying the operator bound.

Use CUDA-first quality and behavior checks; retain existing CPU evidence without inventing CPU acceptance. The 105-run ETT/cement formal matrix and D4 are pending. Registry status remains preview until its task acceptance is complete; passing development tests alone cannot promote it.

Formal P3 execution uses tools/validation/p3_acceptance.py and p3_reports.py. Freeze validation inputs as well as runtime source: the shared runtime digest deliberately excludes tests. Keep a suite-owned binary, full fixture/test inventories and nonblocking executor lock. Each CUDA worker executes sequentially; stop dispatch on failure. Do not treat development diagnostics or published P1/P2 numbers as accepted P3 runs.
