# iTransformer

Status: the Linux CUDA formal matrix passed on 2026-10-12. Suite `runs/p2-acceptance/20261012-linux-cuda` has 105 runs, D4 32/32, cement 9/9, and an independent recomputation of 18,574,560 rows with 6/6 metrics. Runtime source `fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055`. Linux binary `43fd7365d289e01d3c0ff14882a8b54f7facdf79795af3905d05931bba9fe596`. Host: Ubuntu 24.04, two RTX 4090 GPUs, CUDA toolkit 12.9. The registry status for this source is `p2_cuda_formal_pass`. The full CPU control was not run. This suite does not record an RTX 5090 Laptop result. Earlier Windows suites remain historical.

Source: thuml/iTransformer, commit c2426e68ca13f74aaec08045c5c724d8ad328124. The implementation follows model/iTransformer.py, layers/Embed.py, layers/Transformer_EncDec.py and layers/SelfAttention_Family.py. Production execution uses Rust and Candle 0.11.0; Python is confined to reference and validation tools.

## Contract

The model predicts the input variables or an ordered subset. Inputs use FP32 history [B,L,C] and optional history-time [B,L,K]. Time tokens are removed before target selection. Future-known numeric variables and unavailable history are rejected. Window normalization, inverted embedding, post-norm encoder, final norm, projector and inverse normalization follow the pinned source. GELU uses the exact erf form.

All 38 names in the two-layer model match the official state_dict. Pointwise conv1/conv2 weights remove the final kernel dimension of one in Rust; export restores that dimension. The pinned fixture manifest lists every official and Rust shape. Model, mapping, L/H/C/K, variable order, frequency and data protocol are checked during loading.

Training uses the existing PyTorch-compatible MT19937 initialization/sampling policy and the independent ChaCha8 dropout stream. Each microbatch advances forward_id, including empty-supervision microbatches. The nine dropout sites in a two-layer model have stable IDs. Official and Rust dropout are compared with explicit shared masks in fixtures; native CUDA dropout streams are not claimed to be identical.

V2 checkpoints contain twelve files and restore only at an empty gradient-accumulation boundary. Fresh and resumed training use TrainingSession. Resume requires unchanged data, source, config, environment and device. A continuation preserves the parent run and inherits a verified local best generation. V1 remains inference-only.

## Validation

Forward and true MSE gradients use atol=1e-5, rtol=1e-4. The approved AC3 optimizer gate supplies the same official true MSE gradients to both Adam implementations and compares all parameters, moments and steps at atol=1e-6, rtol=0. Independent-backward update differences remain diagnostics, including the measured near-zero key-bias gradient effect.

The fixed formal CUDA matrix has 48 official ETT, 48 Rust ETT and 9 Rust cement runs with seeds 2021/2022/2023. D4 uses population standard deviations and the original two-sided bound. That matrix passed on the Linux suite named above. The original full CPU control is unexecuted under the later CUDA-first instruction; a short CPU diagnostic cannot substitute for that control.

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

Snapshot: 2026-10-12, after executor PID 450356 exited.

Suite: runs/p2-acceptance/20261012-linux-cuda. Runtime source SHA256: fd6347469c8389de3149e662e68eb5ac3dfe6e958815c8891daa38be9a1eb055. Linux binary SHA256: 43fd7365d289e01d3c0ff14882a8b54f7facdf79795af3905d05931bba9fe596. GPU: NVIDIA GeForce RTX 4090. Historical run directories were not loaded. The P1 cement numbers in the report are the committed CUDA table, seed 2021, n=1.

Verified completed entries: official CUDA ETT 48/48; Rust CUDA ETT 48/48; cement 9/9. D4 passed 32/32 with ddof=0 and zero failing comparisons. CUDA numerical cases passed 9/9. Fresh-process recovery passed 10/10 on cuda:0. Independent recomputation passed 18,574,560 rows and 6/6 metrics; the largest absolute difference is 1.2505552149377763e-12. Quality passed Rust 102 and Python 55, with zero failures and zero ignored tests. Performance evidence covers all 105 training runs. ETT used cuda:0. Cement used batch 32; six runs used cuda:0 and three used cuda:1.

The registry value for this source is `p2_cuda_formal_pass`. The 2026-09-30 suite `runs/p2-acceptance/20260929-cuda-v3` remains historical evidence for source d14b82559d78f124a073e5fe3f6fb9f2c5b80c0847d4acba080643b7ce2fde81 and does not satisfy this identity. The original full CPU control remains unexecuted. Evidence: runs/p2-acceptance/20261012-linux-cuda/final.json and .trellis/tasks/09-28-p2-itransformer/research/acceptance.md.

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
