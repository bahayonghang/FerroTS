"""Generate synthetic FP32 Adam/AdamW reference fixtures on CPU."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "tests" / "fixtures" / "optim"
SEED = 137
GROUPS = [
    {"names": ["encoder.weight"], "lr": 0.003, "beta1": 0.8, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.05},
    {"names": ["encoder.bias"], "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-6, "weight_decay": 0.0},
]

def generate(kind: str) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    params = {
        "encoder.weight": torch.nn.Parameter(torch.randn((2, 3), generator=generator) * 0.3),
        "encoder.bias": torch.nn.Parameter(torch.randn((3,), generator=generator) * 0.3),
    }
    groups = [
        {"params": [params[name] for name in g["names"]], "lr": g["lr"],
         "betas": (g["beta1"], g["beta2"]), "eps": g["eps"], "weight_decay": g["weight_decay"]}
        for g in GROUPS
    ]
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind]
    optimizer = optimizer_type(groups, foreach=False, fused=False)
    tensors = {f"initial.{name}": p.detach().clone() for name, p in params.items()}
    for step in range(1, 11):
        optimizer.zero_grad(set_to_none=True)
        for name, p in params.items():
            gradient = torch.randn(p.shape, generator=generator) * 0.1
            if name == "encoder.bias" and step in (3, 7):
                continue
            p.grad = gradient
            tensors[f"grad.{step:04}.{name}"] = gradient.clone()
        optimizer.step()
        for name, p in params.items():
            state = optimizer.state[p]
            tensors[f"param.{step:04}.{name}"] = p.detach().clone()
            for moment in ("exp_avg", "exp_avg_sq"):
                tensors[f"{moment}.{step:04}.{name}"] = state[moment].clone()
            tensors[f"step.{step:04}.{name}"] = torch.tensor(int(state["step"].item()), dtype=torch.int64)
    manifest = {"format_version": 1, "kind": kind, "seed": SEED, "steps": 10,
                "torch": torch.__version__, "python": sys.version.split()[0], "dtype": "f32",
                "groups": GROUPS, "source": "synthetic; no dataset input"}
    output = DESTINATION / f"{kind}.safetensors"
    save_file(tensors, output, metadata={"fixture": json.dumps(manifest, sort_keys=True)})
    assert output.stat().st_size < 1_000_000
    print(f"{output.relative_to(ROOT)}: {len(tensors)} tensors, {output.stat().st_size} bytes")

def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for kind in ("adam", "adamw"):
        generate(kind)

if __name__ == "__main__":
    main()
