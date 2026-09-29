# P1 reference fixtures

Generate with tools/reference/gen_p1_fixtures.py in the existing reference uv environment.
Use --check to compare a fresh calculation with the stored bytes.

- dlinear.safetensors: synthetic inputs, targets and weights; official decomposition,
  predictions, input/parameter gradients and one Adam update for three shapes.
- dlinear.json: case shapes, numerical tolerances, seed, package version and upstream SHA.
- ett.json: public ETT window counts, train-only scaler statistics, public timestamps
  and time features. The calendar boundary cases use synthetic timestamps.
- reference_rng.json: synthetic pinned-torch integer draws, permutations, Linear/
  DLinear parameter bits and ten epochs of single-worker sampler orders. Generate
  or verify with tools/validation/gen_rng_fixtures.py [--check]. The 15 model cases
  cover seeds 2021/2022/2023, shared/individual projections and multiple shapes.

No private data is read by this generator. The fixed TSLib source has an MIT license;
the source repository and SHA are recorded in each metadata file.
