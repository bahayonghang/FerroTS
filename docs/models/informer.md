# Informer model card

Status: **preview**. The implementation and CUDA development checks are available; the 105-run formal ETT/cement matrix, D4 comparison and private prediction recomputation have not been completed.

## Source and configuration

The source is Time-Series-Library commit `4e938a1767106324dd753b2a44832bf870a0252e`, especially models/Informer.py, layers/Embed.py, layers/Transformer_EncDec.py and layers/SelfAttention_Family.py. Copyright (c) 2021 THUML @ Tsinghua University.

The reproduction uses ProbAttention in the encoder and in both decoder attention positions. It preserves the pinned implementation's BHLD context view, sparse score denominator, circular convolution padding, and unapplied attention-weight dropout. No dense cross-attention is substituted.

ETT configurations are under configs/reference/informer/. ETTh1 follows the pinned Informer_ETTh1.sh; ETTh2/ETTm1/ETTm2 are explicitly derived profiles, as specified by docs/datasets/ett.md. The pinned TSLib loader keeps training/validation/test tail batches (`drop_last=False`), unlike the separate iTransformer upstream. L96, label_len48, H96/192/336/720, d_model512, heads8, d_ff2048, encoder2/decoder1, factor3, distillation, dropout0.1 and GELU are fixed for ETT.

Cement configurations are separate engineering profiles using cement-twin-v1, all 18 variables, minute features, L96 and H15/30/60. They do not redefine the ETT defaults. Formal seeds are 2021/2022/2023, FP32, one CUDA device per experiment.

## Model and training state

ModelInput contains history and known calendar features only. Decoder numeric input consists of the final label_len historical steps followed by H zeros. Supervision is separate. Initial support requires gap=0 and rejects future-known numeric columns.

The static dispatcher exposes Informer through the existing trainer, resume, evaluate, predict and model-list interfaces. BatchNorm running mean/variance and batch counters, together with positional buffers, are copied into snapshots, atomic v2 checkpoints and independent frozen inference models. Counters use U32 with checked overflow. Buffers do not enter Adam. Loading validates complete names/shapes/dtypes/devices and finite floating values before applying the buffer set.

Informer requires v2 checkpoints; the v1 stateless inference bundle cannot store its buffers. Existing baseline/iTransformer empty-buffer bundles retain their behavior.

Sparse sampling is derived from model seed, training-forward identity, layer/site and a train/eval domain. Validation uses persisted validation batch counts; evaluation binds split and batch; future prediction records its identity. Validation does not advance the training stream. Checkpoints retain the existing batch-order RNG and cursor. These streams are deterministic for FerroTS but are not claimed to reproduce all upstream RNG draws bitwise. Kaiming-normal initialization uses a local Box–Muller stream, not the exact torch normal generator; fixed-weight numerical fixtures and the eventual three-seed D4 gate are separate checks.

## CUDA evidence and limits

Synthetic pinned-upstream fixtures cover evaluation/training forward traces and all 72 parameter MSE gradients for the two-encoder/one-decoder small model. A projection probe separately verifies connectivity for mathematically zero softmax key-bias gradients. Shape checks cover B/head=1, L=1 and private H15/30/60. Sparse cutoff ties, large-positive ELU gradients, tail batches, buffer corruption, supervision isolation and fresh-process restoration have dedicated CUDA checks.

PrefixSum uses a linear-storage CUDA scan and its suffix-sum VJP. At n=1,000,000 FP32, the specified operator forward-plus-VJP measurement reports an 8,000,000-byte peak increment, 2× the 4,000,000-byte input. Input and supplied cotangent are preallocated; output, returned gradient and operator workspace are included. The measurement uses CUDA async-pool USED_MEM_HIGH, not periodic device-wide samples. An outer multiply/sum/full-autograd diagnostic used 40,000,004 additional bytes (10.000001×); the operator result does not describe the memory of that larger graph or a complete training batch.

Complexity checks record encoder and decoder self/cross score/mask shapes at L96/192/384/768/1536. Scores have u×Lk rather than full Lq×Lk shape. Q/K/V remain on CUDA; only random discrete sample indices are generated on the host. Equal importance scores use the pinned Candle bitonic sort's deterministic index ordering; no match to PyTorch's tie selection is claimed.

The development evidence is indexed by .trellis/tasks/09-28-p3-informer/research/implementation-evidence.md. Full CPU training was not scheduled. CUDA diagnostic runs are not formal matrix entries. No official reproduction or superiority over private baselines is claimed at preview status.

## Upstream license

MIT License

Copyright (c) 2021 THUML @ Tsinghua University

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
