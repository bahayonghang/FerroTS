"""Generate synthetic CUDA fixtures from the pinned, unmodified TSLib Informer."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors.torch import load_file, save_file

from fetch_upstream import CACHE, UPSTREAMS, git

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = CACHE / "Time-Series-Library"
DESTINATION = ROOT / "tests/fixtures/p3"
TORCH_SHA = "08187d9e0fba026dc8217405802ab5381dc88d90"
SOURCE_FILES = (
    "LICENSE", "models/Informer.py", "layers/Embed.py",
    "layers/SelfAttention_Family.py", "layers/Transformer_EncDec.py",
    "utils/masking.py", "run.py",
    "scripts/long_term_forecast/ETT_script/Informer_ETTh1.sh",
)


def identity(device: torch.device) -> dict:
    sha = UPSTREAMS["Time-Series-Library"][1]
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != sha:
        raise ValueError("TSLib checkout does not match the pinned commit")
    if git("status", "--porcelain", "--untracked-files=all", cwd=UPSTREAM):
        raise ValueError("TSLib checkout is dirty")
    if torch.__version__ != "2.14.0+cu130" or torch.version.git_version != TORCH_SHA:
        raise ValueError("reference torch does not match the approved pin")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("P3 reference generation requires CUDA")
    return dict(
        upstream_sha=sha, license="MIT; Copyright (c) 2021 THUML @ Tsinghua University",
        torch=torch.__version__, torch_sha=TORCH_SHA, cuda=torch.version.cuda,
        python=sys.version.split()[0], device=str(device),
        gpu=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        sources={p: hashlib.sha256((UPSTREAM / p).read_bytes()).hexdigest()
                 for p in SOURCE_FILES},
    )


def model_type():
    sys.path.insert(0, str(UPSTREAM))
    module = importlib.import_module("models.Informer")
    if Path(module.__file__).resolve() != UPSTREAM / "models/Informer.py":
        raise ValueError("a different Informer module is already imported")
    return module.Model


def config() -> dict:
    return dict(task_name="long_term_forecast", seq_len=12, label_len=6,
                pred_len=4, enc_in=3, dec_in=3, c_out=3, d_model=8,
                n_heads=2, d_ff=16, e_layers=2, d_layers=1, factor=2,
                distil=True, dropout=0.0, activation="gelu", embed="timeF", freq="h")


def stored(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu().contiguous().clone()
    return value.to(torch.uint32) if value.dtype == torch.int64 else value


def model_case(cls, name: str, device: torch.device, *, train: bool):
    cfg, seed, batch = config(), 2021, 2
    torch.manual_seed(seed)
    model = cls(SimpleNamespace(**cfg)).float().to(device)
    model.train(train)
    tensors = {f"initial.{n}": stored(v) for n, v in model.named_parameters()}
    tensors.update({f"buffer.initial.{n}": stored(v) for n, v in model.named_buffers()})
    generator = torch.Generator().manual_seed(seed + 10000)
    history = torch.randn(batch, cfg["seq_len"], cfg["enc_in"], generator=generator).to(device)
    history_time = (torch.rand(batch, cfg["seq_len"], 4, generator=generator) - .5).to(device)
    future_time = (torch.rand(batch, cfg["pred_len"], 4, generator=generator) - .5).to(device)
    target = torch.randn(batch, cfg["pred_len"], cfg["c_out"], generator=generator).to(device)
    decoder = torch.cat((history[:, -cfg["label_len"]:], torch.zeros_like(target)), dim=1)
    decoder_time = torch.cat((history_time[:, -cfg["label_len"]:], future_time), dim=1)
    tensors.update({f"input.{n}": stored(v) for n, v in dict(
        history=history, history_time=history_time, future_time=future_time,
        target=target, decoder=decoder, decoder_time=decoder_time).items()})
    samples, traces, shapes, handles = {}, {}, {}, []
    sample_generator = torch.Generator().manual_seed(seed + 20000)
    randint = torch.randint

    def observe_attention(path, original):
        def sampled(q, k, sample_k, n_top):
            indices = randint(k.shape[-2], (q.shape[-2], sample_k), generator=sample_generator)
            samples[path] = stored(indices)
            # Only intercept the upstream random-index draw. All score, top-k,
            # masking, context, layout and autograd code remains upstream code.
            with patch.object(torch, "randint", return_value=indices):
                score, selected = original(q, k, sample_k, n_top)
            tensors[f"top_query.{path}"] = stored(selected)
            tensors[f"sample_score.{path}"] = stored(score)
            shapes[path] = dict(query=list(q.shape), key=list(k.shape),
                                samples=list(indices.shape), score=list(score.shape),
                                top_query=list(selected.shape))
            return score, selected
        return sampled

    def hook(path):
        def capture(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                traces[path] = value
        return capture

    for path, module in model.named_modules():
        if path:
            handles.append(module.register_forward_hook(hook(path)))
        if path.endswith(".inner_attention"):
            prefix = path.removesuffix(".inner_attention")
            module._prob_QK = observe_attention(prefix, module._prob_QK)
    output = model(history, history_time, decoder, decoder_time)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward(retain_graph=True)
    gradients = {}
    for path, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing or nonfinite MSE gradient: {path}")
        gradients[path] = stored(parameter.grad)
    # A projection probe diagnoses graph connectivity for key biases whose
    # shared shift cancels mathematically from softmax scores.
    model.zero_grad(set_to_none=True)
    probe = loss + sum(value.square().mean() for path, value in traces.items()
                       if path.endswith(".key_projection"))
    probe.backward()
    for path, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing or nonfinite probe gradient: {path}")
        tensors[f"probe_gradient.{path}"] = stored(parameter.grad)
    tensors.update({f"gradient.{p}": g for p, g in gradients.items()})
    tensors.update({f"sample.{p}": v for p, v in samples.items()})
    tensors.update({f"trace.{p}": stored(v) for p, v in traces.items()})
    tensors.update({f"buffer.after.{p}": stored(v) for p, v in model.named_buffers()})
    tensors.update(output=stored(output), loss=stored(loss), probe_loss=stored(probe))
    for handle in handles:
        handle.remove()
    if not torch.count_nonzero(output) or not torch.isfinite(loss) or loss.item() <= 0:
        raise AssertionError("degenerate fixture")
    meta = dict(name=name, config=cfg, seed=seed, batch=batch, train=train,
                source="unmodified upstream; only torch.randint result injected",
                samples_seed=seed + 20000, parameters={p: list(v.shape) for p, v in model.named_parameters()},
                buffers={p: dict(shape=list(v.shape), dtype=str(stored(v).dtype))
                         for p, v in model.named_buffers()},
                zero_mse_gradient_parameters=[p for p, g in gradients.items() if not torch.count_nonzero(g)],
                traces={p: list(v.shape) for p, v in traces.items()}, attention_shapes=shapes,
                mse=float(loss.detach()), probe="MSE + sum(mean(key_projection_output ** 2))",
                upstream_context_layout="BHLD contiguous then view(B,L,H*D); no transpose")
    return tensors, samples, meta


def operator_cases(device: torch.device):
    """Small exact semantic references; no model/data/production dependencies."""
    tensors = {}
    for name, values in (
        ("maxpool_overlap", [[[1., 4., 2., 5., 3., 2., -1.]]]),
        ("maxpool_tie", [[[2., 2., 2., 1., 2.]]]),
        ("maxpool_length_one", [[[3.]]]),
    ):
        x = torch.tensor(values, device=device, requires_grad=True)
        output, indices = torch.nn.functional.max_pool1d(x, 3, 2, 1, return_indices=True)
        cotangent = torch.arange(1, output.numel() + 1, device=device).reshape_as(output).float()
        (output * cotangent).sum().backward()
        for key, value in dict(input=x, output=output, indices=indices,
                               cotangent=cotangent, gradient=x.grad).items():
            tensors[f"{name}.{key}"] = stored(value)
    for length in (1, 7):
        x = torch.linspace(-1., 1., 2 * length * 3, device=device).reshape(2, length, 3).requires_grad_()
        output = x.cumsum(1)
        cotangent = torch.linspace(.2, 1.3, output.numel(), device=device).reshape_as(output)
        (output * cotangent).sum().backward()
        for key, value in dict(input=x, output=output, cotangent=cotangent, gradient=x.grad).items():
            tensors[f"prefix_sum_{length}.{key}"] = stored(value)
    base = torch.arange(2 * 5 * 3, device=device).reshape(2, 5, 3).float().requires_grad_()
    updates = torch.linspace(-3., 3., 2 * 2 * 3, device=device).reshape(2, 2, 3).requires_grad_()
    indices = torch.tensor([[1, 3], [0, 4]], device=device)
    output = base.clone()
    output[torch.arange(2, device=device)[:, None], indices] = updates
    cotangent = torch.linspace(.1, 1.2, output.numel(), device=device).reshape_as(output)
    (output * cotangent).sum().backward()
    for key, value in dict(base=base, updates=updates, indices=indices, output=output,
                           cotangent=cotangent, base_gradient=base.grad, updates_gradient=updates.grad).items():
        tensors[f"replace_rows.{key}"] = stored(value)
    return tensors


def emit(path: Path, tensors: dict, check: bool):
    if not check:
        save_file(tensors, path)
        return
    expected = load_file(path)
    if expected.keys() != tensors.keys():
        raise AssertionError(f"tensor keys differ: {path.name}")
    for key, value in tensors.items():
        other = expected[key]
        if other.dtype != value.dtype or other.shape != value.shape or not torch.equal(other, value):
            raise AssertionError(f"tensor content differs: {path.name}:{key}")


def generate(destination: Path, device_name: str, check: bool):
    device = torch.device(device_name)
    source = identity(device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cls = model_type()
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, train in (("standard_eval", False), ("standard_train", True)):
        tensors, samples, meta = model_case(cls, name, device, train=train)
        emit(destination / f"{name}.safetensors", tensors, check)
        emit(destination / f"{name}.samples.safetensors", samples, check)
        cases.append(meta)
    emit(destination / "operators.safetensors", operator_cases(device), check)
    torch.cuda.synchronize(device)
    manifest = dict(format_version=1, mapping_version=1, source="synthetic only; no dataset reads",
                    identity=source, cases=cases, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    tolerance=dict(atol=1e-5, rtol=1e-4),
                    boundary_policy="B/head=1 and L=1 require separately documented shape-safe extension; not claimed as unmodified upstream cases",
                    files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(destination.glob("*.safetensors"))})
    path = destination / "manifest.json"
    if check:
        if json.loads(path.read_text()) != manifest:
            raise AssertionError("fixture metadata differs")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS P3 CUDA reference: {len(cases)} model cases, {len(cases[0]['parameters'])} parameter gradients each, 6 operator cases; {device}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.device, args.check)
