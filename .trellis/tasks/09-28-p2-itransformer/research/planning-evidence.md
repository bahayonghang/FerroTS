# P2 planning evidence

Reviewed on 2026-09-28, America/Chicago. Planning base: `1f1b9ec` on `feat/p2-itransformer`. All upstream inspection used the existing local checkout. No upstream update, system installation, product implementation or formal P2 training ran.

## 1. Predecessor and authority

- P1 implementation: `4361656b0129191ff5ee2f579718b9edc96b1b08`; acceptance record: `7a6df9df05162c307bf1fc2cdff74e4241bc6f52`.
- P1 archive: `34a11fa`; journal: `1f1b9ec`, session 3. The archived task is `.trellis/tasks/archive/2026-09/09-28-p1-data-train-loop/`.
- The P1 suite `runs/p1-acceptance/20260928-0842-rng/final.json` reports PASS. This planning session verified all 41 runtime-source hashes and seven gate-file size/hash receipts. The full matrices were not rerun.
- P1 evidence retains official/Rust ETT 48+48, D4 32/32, cement 18/18, independent recomputation of 18,574,560 prediction rows, and CPU65/CUDA71/Python9 tests with zero failures or ignored tests.
- The parent PRD's D3-D6 and X1-X6 refine `FerroTS_Rust_Implementation_Plan.md`, especially sections 7, 8, 10.1, 14.1 and 15. The current dataset cards and implemented backend specs refine older parent research snapshots.
- `CONTEXT.md` and nested context/ADR files were not found in the affected tracked tree. No context file was created.
- P0 and datasets-protocol are already archived. Their closure is outside this session.

## 2. Fixed upstream evidence

`tools/reference/.cache/iTransformer` is clean at `c2426e68ca13f74aaec08045c5c724d8ad328124`. `tools/reference/fetch_upstream.py` and `docs/datasets/ett.md` pin the same identity. Paths prefixed `iT:` below are relative to that checkout.

| Source | Observed contract | Planning consequence |
| --- | --- | --- |
| `iT:model/iTransformer.py:16-42` | Inverted embedding, encoder layers, final encoder LayerNorm and biased projector | Include the final LayerNorm in mapping and fixtures |
| `iT:model/iTransformer.py:45-75` | Detached mean; population variance + 1e-5; history/time tokens; slice to original C; denormalize | Preserve exact order; auxiliary time tokens never become targets |
| `iT:layers/Embed.py:127-143` | Linear(L,D), concatenate variables before time features, then embedding dropout | ETT uses C+4 tokens; cement uses C+5 |
| `iT:layers/Transformer_EncDec.py:26-79` | Post-Norm; kernel=1 Conv1d FFN; three residual/FFN dropout calls per layer | Map Conv1d weights [out,in,1] to Linear [out,in]; give each dropout site a stable ID |
| `iT:layers/SelfAttention_Family.py:134-161,266-300` | Q/K/V/out biased Linear; full attention; 1/sqrt(head_dim); softmax then attention dropout | Reuse train-safe attention with exact head layout |
| `iT:run.py:9-12,35-86` | Default seed 2023; h/timeF; heads8; dropout0.1; GELU; Adam settings; norm enabled | Wrapper explicitly supplies seeds 2021/2022/2023 and records effective arguments |
| `iT:data_provider/data_factory.py:20-49` | Shuffled/drop-last train and validation; ordered batch1 test | Preserve the official runner; record Rust validation-policy differences |
| `iT:experiments/exp_long_term_forecasting.py:31-36,39-80,166-182` | Adam/MSE; mean batch validation; per-epoch test logging; early stop then type1; reload best | Rust keeps frozen, element-weighted validation and no test access during fit |
| `iT:scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh:5-79` | ETTh1 H96/192 D=FF=256; H336/720 D=FF=512 | Materialize all 16 effective configs; a horizon override alone is insufficient |
| `docs/datasets/ett.md:95-120` and section 11.3 | L96, batch32, lr1e-4, epochs10, patience3; ETTh2/ETTm1/ETTm2 D=FF=128 | Preserve dataset/horizon-specific official values |

The parent research statements that Windows CUDA is untested, `runs/` is not ignored, and the expanded f-CaO dataset exists are historical. Current environment/data cards and P1 acceptance supersede those statements. Exchange remains 7,588 records.

## 3. Current code and required extension

| Existing owner | Evidence | Required extension |
| --- | --- | --- |
| core configuration | `crates/ferrots-core/src/config.rs:233-269,278-312,550-556` | Add strict iTransformer variant and explicit reference batching policy; preserve P1 defaults and serialized v1 identity |
| nn model/input | `crates/ferrots-nn/src/baselines.rs:11-35,143-155,186-224` | Static common model enum and frozen dispatch; keep baseline constructors and names compatible |
| train loop | `crates/ferrots-train/src/trainer.rs:14,103-139,223-356` | Model dispatch plus an explicit restartable training state; preserve test isolation |
| accumulation | `crates/ferrots-train/src/accumulation.rs` | Expose a checked empty boundary; no partial-gradient serialization |
| optimizer | `crates/ferrots-train/src/optim.rs`, backend training-foundation spec | Reuse FP32 moments, groups and per-parameter counters; coordinator handles full-state transaction |
| checkpoint | `crates/ferrots-train/src/checkpoint.rs:17-22,74-114,166-243` | v1 is inference-only and hard-coded to Baseline/TSLib; add explicit v2 dispatch |
| schedule | `crates/ferrots-train/src/schedule.rs:3-12` | Persist epoch-end transition and current LR; avoid applying type1 twice after resume |
| RNG | `crates/ferrots-core/src/reference_rng.rs:7-10,77-87`; `crates/ferrots-nn/src/random.rs:8-21,40-56` | Versioned MT19937 state and independent reconstructible ChaCha8 dropout context |
| CLI/evaluation | `crates/ferrots-cli/src/pipeline.rs`, `crates/ferrots-train/src/evaluation.rs` | Train/evaluate/predict must all dispatch the same model; add resume and models list |
| formal evidence | `tools/validation/p1_acceptance.py`, `p1_quality.ps1`, `recompute_predictions.py` | Reuse receipt/identity/D4 conventions; add P2-specific orchestration without rewriting P1 history |

P1 publishes last/best before the epoch-end LR update. P2 must define a committed next-action state before publishing resumable snapshots; adding optimizer files alone would not restore the next step correctly.

## 4. Upstream-only gradient probe

The existing `uv` environment imported the pinned official model with torch `2.14.0+cu130`. Synthetic CPU FP32 probe: seed2021, B2/L12/C7/K4/H6, D16/FF32/heads4/layers2, dropout0, nondegenerate random targets. No fixture or private-data output was written.

- First probe requested an input gradient with `x.requires_grad=True`. Official in-place `x_enc /= stdev` caused a version-counter backward error. The original failed command remains in the session output.
- The corrected probe used the production input contract, `requires_grad=False`. All 38 parameter tensors had present, finite, nonzero gradients. No persistent buffers were present.
- Key-projection bias maximum absolute gradients were only `3.4924596548080444e-10` and `5.238689482212067e-10`. A shared key bias adds a row-constant attention score, which softmax cancels in exact arithmetic. Floating-point residuals must not be used as the sole connectivity test.
- Keep the full final-MSE gradient and Adam comparisons, with no tolerance exception. Add a separate synthetic projection-probe loss to prove nonzero connectivity for every named parameter, including key bias. Compare the same probe loss and weights on both implementations. Do not alter production MSE or remove a bias.
- Rust window-statistic input derivatives use train-safe out-of-place operations. If input derivatives are tested, use analytic/finite-difference tests; do not present the failed unmodified official path as a passing reference.

## 5. Formal experiment commitments

| Matrix | Count | Scope |
| --- | --- | --- |
| Official iTransformer ETT CUDA | 48 | 4 datasets x 4 horizons x seeds2021/2022/2023 |
| Rust iTransformer ETT CUDA | 48 | Same complete config/data/seed matrix |
| Rust iTransformer cement CUDA | 9 | H15/30/60 x seeds2021/2022/2023 |
| Rust iTransformer CPU control | 1 | Complete ETTh1/H96/seed2021 train/evaluate/predict |
| Total planned full training runs | 106 | Unit/fixture/resume checks are additional |

D4 has 16 configurations x 2 metrics = 32 checks. Use population std (`ddof=0`), full-precision values and the two-sided rule `abs(rust_mean-official_mean) <= max(2*official_std, 0.02*abs(official_mean))`. P1 DLinear results and published paper tables cannot supply P2 official rows.

Cement model settings use pinned official defaults D512/FF2048/heads8/layers2/dropout0.1/GELU/use_norm=true; training reuses the existing cement settings. These are the untuned planning configuration. The four approved cement protocol adjustments remain unchanged. Report nine runs and all 18 variables. P1 baseline evidence uses one seed per model/horizon/device; label that sample count and do not invent a three-seed standard deviation.

## 6. Planning risks and controls

- Exact GELU, near-zero gradients, Adam epsilon effects and CUDA reduction order require the unchanged numerical gates before full training. A failure stops dependent work for diagnosis.
- Saving optimizer state every step would create excessive files during 106 full runs. Default periodic publication is at epoch boundaries; explicit stop-after-step publishes at the next completed optimizer-step boundary. Retain all committed generations and failed staging directories.
- Freeze-for-eval must copy tensor storage. Model loading must not modify a live session until every state file validates.
- CPU bitwise resume applies on the same verified environment, data and threading settings. CUDA resume uses the unchanged atol=1e-6 on this native Windows GPU. Cross-device bitwise equality is not claimed.
- CUDA process-memory peak measurement must be labeled with the actual counter and sampling method. WDDM can make process counters unavailable; preflight must establish a supported device-wide fallback and record its limitation before the full suite. No invented 0-byte peak and no performance threshold.
- P2 stays `planning`. Product code, fixtures, executable configs and runners are created only after explicit approval of the final plan.

## 7. Planning validation

- Trellis validates 14 curated entries in implement.jsonl and 14 in check.jsonl. Every context path exists and each list has no duplicate path.
- PRD retains all eight requirement IDs and assigns AC1-AC8. Design mechanisms and 20 ordered steps cover the complete requirement set. All implementation and acceptance checkboxes remain unchecked.
- Matrix arithmetic is 48+48+9+1=106; independent export coverage is 103,192x60x3=18,574,560 rows. D4 retains 32 checks and ddof=0.
- `git diff --check` and `bash tools/check_private_data.sh` pass. All current changes are confined to this P2 task directory. No product source/config/dependency file changed.
- `task.py current` is empty and task.json status is planning. No task.py start, product implementation, formal P2 matrix, push or remote-CI configuration ran.
- The P1 quality runner hard-codes its old active task path (`tools/validation/p1_quality.ps1:61`) and invokes that Trellis check with plain Python. P2 must use its scoped runner and the existing uv environment; the historical P1 runner is not a current P2 completion command.

## 8. Approved implementation: activation blocked

The user subsequently replied `批准实施`. The approval applies to the current 20-step plan and remains valid. Branch and HEAD were reread as `feat/p2-itransformer` and `1f1b9ec0fa168e88d1eb73c0eb0572d200d3d8e4`; only P2 planning files are dirty.

The execution service rejected both a combined activation command and an isolated `uv run --frozen --project tools/reference python .trellis/scripts/task.py start .trellis/tasks/09-28-p2-itransformer` command before process creation. Both returned `blocked by policy`; no further reason was provided. Read-only PowerShell inspection and `uv ... task.py current` execute. The latter returned no active task, and task.json still has status planning.

No manual status transition, alternative activation path, policy change, sub-agent, product implementation, or formal P2 training was attempted. Resume with the normal task.py start command once the execution policy permits the authorized action, then load trellis-before-dev and continue step 1.

## 9. Renewed approval: third activation rejection

The user again replied `批准实施`. The same approval remains valid for the existing task and its 20-step plan. No new task was created.

The authorized retry ran through the normal execution tool with the environment prefix from implement.md section 0 and the unchanged command:

```powershell
uv run --frozen --project tools/reference python .trellis/scripts/task.py start .trellis/tasks/09-28-p2-itransformer
```

The execution service again returned `CreateProcess ... rejected: blocked by policy` before process creation. This is the third recorded rejection across the approval attempts. The service gave no specific policy reason. The cause remains unknown. No alternate launcher or manual activation was used.

Read-only verification at `2026-09-28T23:45:47-05:00` (America/Chicago) found:

- `task.py current`: empty output, exit code 1; no active task.
- `task.json`: status `planning`.
- Branch: `feat/p2-itransformer`; HEAD: `1f1b9ec0fa168e88d1eb73c0eb0572d200d3d8e4`.
- Working changes remain confined to this P2 task directory. No product implementation or formal P2 training has run.

The approval does not need renewal. Step 1 remains blocked on execution-policy permission for the normal start command.

## 10. Activation succeeded without a proxy prefix

On 2026-09-29, the user supplied a log-based explanation from Claude and authorized the normal start command without the HTTPS_PROXY prefix. The proposed explanation is a command-classification interaction between the word `start` and a URL. That explanation has not been checked against Codex source or an explicit policy diagnostic. The prior records remain historical evidence.

The exact standalone command from section 9 completed with exit code 0 in 0.515 seconds. Trellis reported `planning → in_progress` and set the session task to `.trellis/tasks/09-28-p2-itransformer`. A subsequent `task.py current` returned that path. No hook, permission, sandbox, or policy setting changed. No task status was edited manually.

The local activation succeeds without a proxy prefix in this session. That observed result resolves the activation blocker; the internal classifier cause remains unverified. Continue the approved sequential implementation and preserve the original numerical and formal gates.
