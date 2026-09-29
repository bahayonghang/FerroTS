"""Synthetic pinned-torch initialization and single-worker sampler fixtures."""
import argparse
import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
TSLIB_SHA = '4e938a1767106324dd753b2a44832bf870a0252e'
sys.path.insert(0, str(ROOT / 'tools/reference/.cache/Time-Series-Library'))
from models.DLinear import Model


def generate():
    upstream = ROOT / 'tools/reference/.cache/Time-Series-Library'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() == TSLIB_SHA
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', '*.py'], cwd=upstream).returncode == 0
    assert torch.__version__ == '2.14.0+cu130'
    assert torch.version.git_version == '08187d9e0fba026dc8217405802ab5381dc88d90'
    cases = []
    for seed in (2021, 2022, 2023):
        for kind, length, horizon, channels, individual in (('dlinear', 1, 3, 2, False), ('dlinear', 7, 5, 3, True), ('dlinear', 96, 336, 7, False), ('linear', 7, 5, 3, True), ('linear', 96, 9, 7, False)):
            torch.manual_seed(seed)
            if kind == 'dlinear':
                model = Model(SimpleNamespace(task_name='long_term_forecast', seq_len=length, pred_len=horizon, moving_avg=25, enc_in=channels), individual=individual)
            else:
                model = torch.nn.Module()
                model.Linear = torch.nn.ModuleList([torch.nn.Linear(length, horizon) for _ in range(channels)]) if individual else torch.nn.Linear(length, horizon)
            parameters = {}
            for name, tensor in model.named_parameters():
                values = tensor.detach().flatten().view(torch.int32).tolist()
                parameters[name] = dict(count=len(values), first=values[:32], last=values[-32:])
            after_init = torch.get_rng_state().clone()
            orders = []
            train = torch.utils.data.DataLoader(range(137), batch_size=32, shuffle=True, num_workers=0)
            val = torch.utils.data.DataLoader(range(71), batch_size=32, shuffle=True, num_workers=0)
            test = torch.utils.data.DataLoader(range(41), batch_size=32, shuffle=False, num_workers=0)
            for epoch in range(10):
                orders.append([int(i) for batch in train for i in batch])
                list(val)
                list(test)
                actual = torch.get_rng_state().clone()
                torch.set_rng_state(after_init)
                draws = torch.empty((5*(epoch+1),), dtype=torch.int64).random_()
                assert torch.equal(torch.get_rng_state(), actual), 'loader RNG consumption differs'
                generator = torch.Generator().manual_seed(int(draws[5*epoch+1]))
                assert torch.randperm(137, generator=generator).tolist() == orders[-1]
                torch.set_rng_state(actual)
            cases.append(dict(kind=kind, seed=seed, length=length, horizon=horizon, channels=channels, individual=individual, parameters=parameters, orders=orders))
    primitives = []
    for seed in (2021, 2022, 2023, 0x123456789ABCDEF0):
        g = torch.Generator().manual_seed(seed)
        integers = torch.empty(1000, dtype=torch.int64).random_(generator=g).tolist()
        permutations = {}
        for n in (0, 1, 2, 17, 625, 4099):
            g.manual_seed(seed)
            permutations[str(n)] = torch.randperm(n, generator=g).tolist()
        primitives.append(dict(seed=seed, integers=integers, permutations=permutations))
    return dict(source='synthetic indices and parameters only', upstream_sha=TSLIB_SHA, torch=torch.__version__, torch_sha=torch.version.git_version, cases=cases, primitives=primitives)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate()
    path = ROOT / 'tests/fixtures/p1/reference_rng.json'
    if args.check:
        assert json.loads(path.read_text(encoding='utf-8')) == result
    else:
        path.write_text(json.dumps(result, separators=(',', ':')) + chr(10), encoding='utf-8')
    print('PASS pinned RNG fixtures: 15 model cases, 150 epochs, 4 primitive seeds')
