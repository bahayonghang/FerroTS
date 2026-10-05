# P2 official iTransformer fixtures

The source is the official iTransformer at the SHA pinned in
tools/reference/fetch_upstream.py. All inputs, targets and calendar-like values
are synthetic. No dataset is read. The manifest records source hashes, dependency
versions, the full configuration, parameter shapes and artifact hashes.

Run tools/reference/gen_p2_fixtures.py through the existing frozen uv project.
Use --check to regenerate in memory and compare all tensor bytes and metadata.

Each of the nine cases stores the initial parameters, official layer traces,
final output, true MSE gradients, separate connectivity-probe gradients, and all
parameters and moments after one Adam update. The Adam test uses only the MSE
gradients. The probe adds the mean squared key-projection output for each encoder
layer to MSE. The probe gives key bias a defined nonzero gradient without changing
the production objective. Degenerate L=1 and constant-channel cases require
finite gradients but do not require every gradient to be nonzero.

Parameter names equal the official state_dict names. All shapes are unchanged
except encoder.attn_layers.i.conv1.weight and conv2.weight: remove the final
kernel dimension of size 1 for the Rust Linear representation. Add that dimension
when exporting to the official model. Biases require no shape change. The
manifest lists all 38 names and both shapes for every case.

The initialization file stores every parameter and the complete torch CPU RNG
byte state for seeds 2021, 2022 and 2023. The manifest stores ten complete training
orders per seed for a synthetic 137-sample dataset, batch32 and drop_last=true.
Each order contains 128 samples. The generator verifies the existing five-draw
per-epoch loader RNG contract against the official train/validation/test policy.

buffers.safetensors is a valid empty tensor file. Its Python roundtrip is checked
here. Rust loading is checked with the checkpoint implementation in step 10.
Rust CPU/CUDA loading and numerical checks are separate acceptance gates.


The approved AC3 revision uses the same official true MSE gradients in both Adam
implementations and compares every updated parameter, moment and step at
atol=1e-6, rtol=0. True model MSE gradients remain checked at atol=1e-5,
rtol=1e-4. Independent-backward update differences are retained as diagnostics.

The shared dropout-masks.safetensors file contains eighteen binary masks: nine
sites for each of forward IDs 0 and 1 with seed 2021. dropout0.safetensors and
dropout1.safetensors store official forward/gradient results using those masks.
Normal tests compare the generated masks with the fixed fixture. They do not
rewrite fixtures. The original nine numerical cases remain unchanged.
