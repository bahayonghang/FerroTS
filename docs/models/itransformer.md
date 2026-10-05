# iTransformer

Status: CUDA numerical, fresh-process resume, full historical ETT/D4 and current quality gates passed. A user-requested training stop and scoped CUDA performance repair completed on 2026-09-30. Formal acceptance for the repaired binary remains pending because the runtime identity changed. Cement, independent recomputation and final acceptance remain pending. Earlier CPU evidence is retained. Validation remains CUDA-first.

Source: thuml/iTransformer, commit c2426e68ca13f74aaec08045c5c724d8ad328124. The implementation follows model/iTransformer.py, layers/Embed.py, layers/Transformer_EncDec.py and layers/SelfAttention_Family.py. Production execution uses Rust and Candle 0.11.0; Python is confined to reference and validation tools.

## Contract

The model predicts the input variables or an ordered subset. Inputs use FP32 history [B,L,C] and optional history-time [B,L,K]. Time tokens are removed before target selection. Future-known numeric variables and unavailable history are rejected. Window normalization, inverted embedding, post-norm encoder, final norm, projector and inverse normalization follow the pinned source. GELU uses the exact erf form.

All 38 names in the two-layer model match the official state_dict. Pointwise conv1/conv2 weights remove the final kernel dimension of one in Rust; export restores that dimension. The pinned fixture manifest lists every official and Rust shape. Model, mapping, L/H/C/K, variable order, frequency and data protocol are checked during loading.

Training uses the existing PyTorch-compatible MT19937 initialization/sampling policy and the independent ChaCha8 dropout stream. Each microbatch advances forward_id, including empty-supervision microbatches. The nine dropout sites in a two-layer model have stable IDs. Official and Rust dropout are compared with explicit shared masks in fixtures; native CUDA dropout streams are not claimed to be identical.

V2 checkpoints contain twelve files and restore only at an empty gradient-accumulation boundary. Fresh and resumed training use TrainingSession. Resume requires unchanged data, source, config, environment and device. A continuation preserves the parent run and inherits a verified local best generation. V1 remains inference-only.

## Validation

Forward and true MSE gradients use atol=1e-5, rtol=1e-4. The approved AC3 optimizer gate supplies the same official true MSE gradients to both Adam implementations and compares all parameters, moments and steps at atol=1e-6, rtol=0. Independent-backward update differences remain diagnostics, including the measured near-zero key-bias gradient effect.

The fixed formal CUDA matrix has 48 official ETT, 48 Rust ETT and 9 Rust cement runs with seeds 2021/2022/2023. D4 uses population standard deviations and the original two-sided bound. Full reproduction remains pending until the complete matrix passes. The original full CPU control is unexecuted under the later CUDA-first instruction; a short CPU diagnostic cannot substitute for that control.

## Frozen experiment profiles

| Profile | L | Horizons | Variables | Time features | d_model | d_ff |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| ETTh1 short horizons | 96 | 96, 192 | 7 | timeF, freq=h | 256 | 256 |
| ETTh1 long horizons | 96 | 336, 720 | 7 | timeF, freq=h | 512 | 512 |
| ETTh2, ETTm1, ETTm2 | 96 | 96, 192, 336, 720 | 7 | timeF, freq=h | 128 | 128 |
| cement-twin-v1 | 96 | 15, 30, 60 | 18 | timeF, freq=t | 512 | 2048 |

Every profile uses eight heads, two encoder layers, dropout 0.1, GELU and window normalization. Training uses FP32, batch 32, accumulation 1, Adam lr 1e-4, beta=(0.9,0.999), eps 1e-8, weight_decay 0, max_epochs 10, patience 3 and type1 scheduling. Seeds are 2021/2022/2023. ETT training drops the tail batch; cement retains the tail batch.

## Current performance repair

The repaired development build keeps batch size 32, accumulation 1, FP32, model dimensions, seeds, data protocol and tolerances unchanged. Cargo development-profile optimization applies to the host-side deterministic dropout and gradient-validation hot paths. Debug assertions remain enabled. No release flags, mixed precision, RNG change or model change was introduced.

The synchronized synthetic training step improved from 1080.598 ms to 85.949 ms, a 12.5726x speedup. A bounded real-data H15/seed2021 run completed 128 optimizer steps with exit code 0 and 14.377 s training compute time. The RTX 5090 Laptop GPU was present in all 34 one-second samples. Device utilization ranged from 0% to 20%, with a 5.5% sample mean. The short bursts reflect the fixed small batch and synchronized host checks. The diagnostic confirms CUDA execution and the throughput correction. It does not establish a sustained utilization target or formal cement result.

Evidence: runs/p2-development/gpu-performance-repair-20260930/profile-comparison.json and runs/p2-development/gpu-performance-repair-20260930/real-data-128-20260930-191449-2643853/.

## Measured coverage

Snapshot: 2026-09-30 02:54 America/Chicago.

Suite: runs/p2-acceptance/20260929-cuda-v3. Audited parent: runs/p2-acceptance/20260929-cuda-v2. Runtime source SHA256: d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81. The scoped correction changes validation tooling only; all 96 ETT entries are byte-identical imports.

Verified completed entries: official CUDA ETT 48/48; Rust CUDA ETT 48/48; cement 0/9. Both parent and successor D4 passed 32/32. CUDA numerical cases 9/9 and fresh-process recovery cases 10/10 passed. Current successor quality has Rust 102 and Python 52 tests passed, with zero failures or ignored tests. Cement is active. Full independent recomputation, final quality and final acceptance remain pending.

The frozen runtime registry retains reproduction_status=pending. A registry source change requires handling the new execution identity before receipt reuse. The original full CPU control remains unexecuted. Evidence: runs/p2-development/tooling-transition-20260929-2129/ett-transition-checkpoint-20260930-025430-462084.json; .trellis/tasks/09-28-p2-itransformer/research/acceptance.md.

## License

MIT License

Copyright (c) 2022 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
