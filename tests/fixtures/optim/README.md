# Optimizer fixtures

The files contain only synthetic CPU FP32 tensors. No dataset is read.

Regenerate with the existing reference environment:

    uv run --project tools/reference python tools/reference/gen_optim_fixture.py

Python is pinned to 3.12.12. Torch is pinned by tools/reference/pyproject.toml
and uv.lock. The generator records the actual Python and Torch versions in
safetensors metadata. Both optimizers use two parameter groups, ten updates,
nonzero weight decay on the weight matrix, and missing bias gradients on
steps 3 and 7. Reference keys store initial values, available gradients,
parameters, FP32 first/second moments, and per-parameter update counts.

Tests compare each available update at absolute tolerance 1e-6. Fixtures are
generated with foreach=False and fused=False. The generator is development
code; production Rust builds do not require Python.
