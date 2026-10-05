"""Export synthetic fixtures from the pinned official iTransformer."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from fetch_upstream import CACHE, UPSTREAMS, git

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = CACHE / "iTransformer"
TORCH_SHA = "08187d9e0fba026dc8217405802ab5381dc88d90"
DESTINATION = ROOT / "tests/fixtures/p2"
SEEDS = (2021, 2022, 2023)


def source_identity() -> dict:
    sha = UPSTREAMS["iTransformer"][1]
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != sha:
        raise ValueError("iTransformer checkout does not match the pinned commit")
    if git("status", "--porcelain", "--untracked-files=all", cwd=UPSTREAM):
        raise ValueError("iTransformer checkout is dirty")
    if torch.__version__ != "2.14.0+cu130" or torch.version.git_version != TORCH_SHA:
        raise ValueError("reference torch version does not match the approved pin")
    paths = ("model/iTransformer.py", "layers/Embed.py",
             "layers/SelfAttention_Family.py", "layers/Transformer_EncDec.py")
    return dict(upstream_sha=sha, torch=torch.__version__, torch_sha=TORCH_SHA,
                python=sys.version.split()[0],
                sources={p: hashlib.sha256((UPSTREAM / p).read_bytes()).hexdigest() for p in paths})


def model_type():
    source_identity()
    path = str(UPSTREAM)
    if path not in sys.path:
        sys.path.insert(0, path)
    module = importlib.import_module("model.iTransformer")
    if Path(module.__file__).resolve() != (UPSTREAM / "model/iTransformer.py").resolve():
        raise ValueError("a different iTransformer module is already imported")
    return module.Model


def config(**changes) -> dict:
    result = dict(seq_len=12, pred_len=6, enc_in=7, d_model=16, n_heads=4,
                  d_ff=32, e_layers=2, dropout=0.0, activation="gelu",
                  use_norm=True, output_attention=False, embed="timeF",
                  freq="h", factor=1, class_strategy="projection")
    result.update(changes)
    return result


def mapped(name: str, tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()
    if name.endswith((".conv1.weight", ".conv2.weight")):
        if value.ndim != 3 or value.shape[-1] != 1:
            raise ValueError(f"invalid pointwise convolution: {name}")
        value = value.squeeze(-1)
    return value.contiguous().clone()


def named_gradients(model, *, nonzero=False) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"missing or nonfinite gradient: {name}")
        if nonzero and not torch.count_nonzero(gradient):
            raise AssertionError(f"zero connectivity-probe gradient: {name}")
        result[name] = mapped(name, gradient)
    return result


def case_fixture(model_class, name: str, cfg: dict, *, seed=2021, batch=2,
                 time_channels=4, constant=False, noncontiguous=False, dropout_masks=None, forward_id=None):
    torch.manual_seed(seed)
    model = model_class(SimpleNamespace(**cfg)).float().eval()
    if dict(model.named_buffers()):
        raise AssertionError("expected an empty official buffer set")
    mask_calls = {}
    if dropout_masks is not None:
        model.train()
        def shared_dropout(path):
            def forward(x):
                call = mask_calls.get(path, 0)
                mask_calls[path] = call + 1
                if path == "enc_embedding.dropout":
                    site = 0
                else:
                    layer = int(path.split(".")[2])
                    site = 1 + 4 * layer + (0 if "inner_attention" in path else 1 + call)
                mask = dropout_masks[f"forward{forward_id}.site{site}"]
                if mask.shape != x.shape:
                    raise AssertionError(f"dropout site shape mismatch: {path}/{call}")
                return (x * mask) / (1 - cfg["dropout"])
            return forward
        for path, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.forward = shared_dropout(path)
    tensors = {f"initial.{n}": mapped(n, p) for n, p in model.named_parameters()}
    tensors["post_init_rng"] = torch.get_rng_state().clone()
    g = torch.Generator().manual_seed(seed + 10000)
    b, length, channels = batch, cfg["seq_len"], cfg["enc_in"]
    if noncontiguous:
        x = torch.randn((b, channels, length), generator=g).transpose(1, 2)
        assert not x.is_contiguous()
    else:
        x = torch.randn((b, length, channels), generator=g)
    if constant:
        x[:, :, 0] = 1.25
    times = torch.rand((b, length, time_channels), generator=g) - 0.5 if time_channels else None
    target = torch.randn((b, cfg["pred_len"], channels), generator=g)
    tensors["input.history"] = x.contiguous().clone()
    tensors["input.target"] = target.clone()
    if times is not None:
        tensors["input.history_time"] = times.clone()
    traces, handles, calls = {}, [], {}

    def capture(path):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            index = calls.get(path, 0)
            calls[path] = index + 1
            traces[f"{path}.{index}"] = value
        return hook

    for path, module in model.named_modules():
        if path and (isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.LayerNorm, torch.nn.Dropout))
                     or path == "enc_embedding" or path.endswith(".attention")
                     or path.rsplit(".", 1)[0] == "encoder.attn_layers"):
            handles.append(module.register_forward_hook(capture(path)))
    output = model(x, times, None, None)
    tensors["output"] = output.detach().contiguous().clone()
    tensors.update({f"trace.{key}": value.detach().contiguous().clone() for key, value in traces.items()})
    loss = torch.nn.functional.mse_loss(output, target)
    tensors["loss"] = loss.detach().clone()
    loss.backward(retain_graph=True)
    tensors.update({f"gradient.{key}": value for key, value in named_gradients(model).items()})
    # A shared key bias cancels in attention scores. Test its connection with
    # the projection loss, without relying on floating-point MSE residuals.
    model.zero_grad(set_to_none=True)
    probe = loss
    for key, value in traces.items():
        if key.endswith(".key_projection.0"):
            probe = probe + value.square().mean()
    probe.backward()
    require_nonzero = not constant and length > 1
    tensors["probe_loss"] = probe.detach().clone()
    tensors.update({f"probe_gradient.{key}": value for key, value in named_gradients(model, nonzero=require_nonzero).items()})
    for handle in handles:
        handle.remove()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False, fused=False)
    for key, parameter in model.named_parameters():
        parameter.grad = tensors[f"gradient.{key}"].reshape(parameter.shape).clone()
    optimizer.step()
    for key, parameter in model.named_parameters():
        tensors[f"updated.{key}"] = mapped(key, parameter)
        state = optimizer.state[parameter]
        for moment in ("exp_avg", "exp_avg_sq"):
            tensors[f"{moment}.{key}"] = mapped(key, state[moment])
        tensors[f"step.{key}"] = torch.tensor(int(state["step"]), dtype=torch.int64)
    mapping = [dict(name=n, official_shape=list(p.shape), rust_shape=list(mapped(n, p).shape),
                    dtype="F32") for n, p in model.named_parameters()]
    meta = dict(name=name, config=cfg, seed=seed, batch=batch, time_channels=time_channels,
                constant_channel=constant, noncontiguous_input=noncontiguous,
                require_nonzero_probe=require_nonzero, parameters=mapping,
                traces=list(traces), input_requires_grad=False,
                probe="MSE + sum(mean(key_projection_output ** 2)) across encoder layers",
                adam=dict(lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))
    if dropout_masks is not None:
        if sum(mask_calls.values()) != 9:
            raise AssertionError("shared dropout site count changed")
        meta.update(forward_id=forward_id,mask_source="dropout-masks.safetensors",dropout_site_version=1)
    return tensors, meta


def initialization_fixture(model_class):
    tensors, cases = {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)
        cfg = config()
        model = model_class(SimpleNamespace(**cfg))
        for name, parameter in model.named_parameters():
            tensors[f"seed{seed}.{name}"] = mapped(name, parameter)
        state = torch.get_rng_state().clone()
        tensors[f"seed{seed}.rng_state"] = state
        train = torch.utils.data.DataLoader(range(137), batch_size=32, shuffle=True, drop_last=True, num_workers=0)
        val = torch.utils.data.DataLoader(range(71), batch_size=32, shuffle=True, drop_last=True, num_workers=0)
        test = torch.utils.data.DataLoader(range(41), batch_size=1, shuffle=False, num_workers=0)
        orders = []
        for epoch in range(10):
            orders.append([int(i) for batch in train for i in batch])
            list(val)
            list(test)
            actual = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            draws = torch.empty((5 * (epoch + 1),), dtype=torch.int64).random_()
            if not torch.equal(torch.get_rng_state(), actual):
                raise AssertionError("official loader RNG consumption changed")
            generator = torch.Generator().manual_seed(int(draws[5 * epoch + 1]))
            full_order = torch.randperm(137, generator=generator).tolist()
            if full_order[:128] != orders[-1]:
                raise AssertionError("official training permutation differs")
            torch.set_rng_state(actual)
        cases.append(dict(seed=seed, config=cfg, train_count=137, batch_size=32,
                          drop_last=True, orders=orders))
    return tensors, cases


def layer_fixture():
    generator = torch.Generator().manual_seed(137)
    params = {
        "input": torch.randn((2, 3, 4), generator=generator).requires_grad_(),
        "weight": torch.randn((5, 4), generator=generator).requires_grad_(),
        "bias": torch.randn((5,), generator=generator).requires_grad_(),
        "gamma": torch.randn((5,), generator=generator).requires_grad_(),
        "beta": torch.randn((5,), generator=generator).requires_grad_(),
    }
    # Noncontiguous leading axes exercise reshape and projection gradients.
    projected = torch.nn.functional.linear(params["input"].transpose(0, 1), params["weight"], params["bias"])
    activated = torch.nn.functional.gelu(projected, approximate="none")
    normalized = torch.nn.functional.layer_norm(activated, (5,), params["gamma"], params["beta"], 1e-5)
    normalized.square().mean().backward()
    tensors = {name: value.detach().clone() for name, value in params.items()}
    tensors.update({f"gradient.{name}": value.grad.clone() for name, value in params.items()})
    tensors.update(projected=projected.detach().clone(), activated=activated.detach().clone(), normalized=normalized.detach().clone())
    return tensors


def emit_tensors(path: Path, tensors: dict, check: bool):
    if check:
        existing = load_file(path)
        if existing.keys() != tensors.keys():
            raise AssertionError(f"tensor keys differ: {path}")
        for key, value in tensors.items():
            other = existing[key]
            if (other.dtype != value.dtype or other.shape != value.shape
                    or not torch.equal(other.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))):
                raise AssertionError(f"tensor content differs: {path.name}:{key}")
    else:
        save_file(tensors, path)


def generate(destination=DESTINATION, check=False):
    identity = source_identity()
    cls = model_type()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    specs = [
        ("standard_k4", config(), {}),
        ("no_time", config(), dict(time_channels=0)),
        ("minute_k5", config(freq="t"), dict(time_channels=5)),
        ("no_norm", config(use_norm=False), {}),
        ("relu", config(activation="relu"), {}),
        ("batch_one", config(), dict(batch=1)),
        ("length_one", config(seq_len=1), {}),
        ("constant_channel", config(), dict(constant=True)),
        ("noncontiguous", config(), dict(noncontiguous=True)),
    ]
    cases = []
    for name, cfg, options in specs:
        tensors, meta = case_fixture(cls, name, cfg, **options)
        emit_tensors(destination / f"{name}.safetensors", tensors, check)
        cases.append(meta)
    tensors, initializations = initialization_fixture(cls)
    emit_tensors(destination / "initialization.safetensors", tensors, check)
    emit_tensors(destination / "layers.safetensors", layer_fixture(), check)
    emit_tensors(destination / "buffers.safetensors", {}, check)
    masks = load_file(destination / "dropout-masks.safetensors")
    dropout_cases = []
    for forward_id in (0, 1):
        name = f"dropout{forward_id}"
        tensors, meta = case_fixture(cls,name,config(dropout=0.1),dropout_masks=masks,forward_id=forward_id)
        emit_tensors(destination / f"{name}.safetensors",tensors,check)
        dropout_cases.append(meta)
    if load_file(destination / "buffers.safetensors"):
        raise AssertionError("empty buffer file contains tensors")
    manifest = dict(format_version=1, mapping_version=1, source="synthetic only; no dataset reads",
                    identity=identity, cases=cases, dropout_cases=dropout_cases, initializations=initializations,
                    files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(destination.glob("*.safetensors"))})
    manifest_path = destination / "manifest.json"
    if check:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise AssertionError("fixture metadata differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS P2 reference fixtures: {len(cases)} cases, 38 parameters each, 3 seeds, 30 sampler epochs")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    arguments = parser.parse_args()
    generate(arguments.output, arguments.check)
