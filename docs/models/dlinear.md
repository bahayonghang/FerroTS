# DLinear model card

FerroTS ports the forecasting semantics from models/DLinear.py and the series decomposition in layers/Autoformer_EncDec.py, Time-Series-Library commit 4e938a1767106324dd753b2a44832bf870a0252e.

Inputs and outputs use [batch, time, channel]. DLinear uses replicated boundary values for a centered odd moving average. The seasonal and trend branches use independent affine projections. Shared and per-channel projections are supported. Weights start at 1 / lookback. Initialization matches CPU PyTorch 2.14.0+cu130 (08187d9e0fba026dc8217405802ab5381dc88d90), including discarded weight draws and interleaved seasonal/trend creation for individual channels. Synthetic fixtures check parameter bits and ten sampler epochs for each fixed seed.

Training order reproduces the pinned single-worker TSLib loaders. Each reference epoch consumes five global int64 draws: train base and sampler seeds, validation base and sampler seeds, and test base seed. FerroTS advances this RNG stream without accessing test data; validation remains ordered and element-weighted. P0 dropout retains its separate ChaCha8 stream.

Candle 0.11.0 rejects the backward pass for overlapping average pooling. FerroTS uses ordered differentiable additions and a final division. The operation preserves upstream FP32 accumulation order, including small seasonal residuals at length one. The synthetic upstream fixtures verify decomposition, predictions, input gradients, parameter gradients, and one Adam update on CPU and CUDA.

P1 supports FP32, one CPU or CUDA device, and named target subsets. Time features are accepted through the common input interface but are not used by these baselines. Future-known inputs are rejected at configuration validation. P1 checkpoints support inference only.

Formal ETT and cement results are pending until the complete acceptance suite passes.

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
