"""Compare unchanged official iTransformer CPU/CUDA numerics with the frozen fixture.

This diagnostic never changes a fixture, model parameter rule or acceptance gate.
All inputs are synthetic. The full report belongs under ignored runs/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
import torch
from safetensors.torch import load_file
from gen_p2_fixtures import mapped, model_type, source_identity


def compare(actual, expected, atol, rtol):
    actual = actual.detach().cpu().flatten()
    expected = expected.detach().cpu().flatten()
    error = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    failed = int((error > atol + rtol * expected.abs()).sum())
    index = int(error.argmax())
    return dict(passed=finite and failed == 0, finite=finite,
                max_error=float(error[index]), failed_elements=failed,
                worst_index=index, actual=float(actual[index]), reference=float(expected[index]),
                actual_max_abs=float(actual.abs().max()), reference_max_abs=float(expected.abs().max()))


def diagnose(destination: Path, ieee_fp32=False):
    cls = model_type()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not skip the official device control")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if ieee_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((ROOT / "tests/fixtures/p2/manifest.json").read_text(encoding="utf-8"))
    report = dict(identity=source_identity(), gpu=torch.cuda.get_device_name(0),
                  cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  unchanged_tolerances=dict(gradient_atol=1e-5, gradient_rtol=1e-4,
                                            adam_atol=1e-6, adam_rtol=0), devices={})
    for device in ("cpu", "cuda:0"):
        results, failures = [], []
        for case in manifest["cases"]:
            model = cls(SimpleNamespace(**case["config"])).float().eval().to(device)
            values = load_file(ROOT / f"tests/fixtures/p2/{case['name']}.safetensors")
            state = {name: values[f"initial.{name}"].reshape(p.shape).to(device)
                     for name, p in model.named_parameters()}
            model.load_state_dict(state, strict=True)
            history = values["input.history"].to(device)
            if case["noncontiguous_input"]:
                history = history.transpose(1, 2).contiguous().transpose(1, 2)
            times = values.get("input.history_time")
            if times is not None:
                times = times.to(device)
            output = model(history, times, None, None)
            loss = torch.nn.functional.mse_loss(output, values["input.target"].to(device))
            loss.backward()
            checks = {}
            for name, p in model.named_parameters():
                checks[name] = dict(gradient=compare(mapped(name,p.grad),values[f"gradient.{name}"],1e-5,1e-4))
            optimizer = torch.optim.Adam(model.parameters(), **dict(lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,foreach=False,fused=False))
            optimizer.step()
            for name, p in model.named_parameters():
                checks[name]["updated"] = compare(mapped(name,p),values[f"updated.{name}"],1e-6,0)
                for moment in ("exp_avg", "exp_avg_sq"):
                    checks[name][moment] = compare(mapped(name,optimizer.state[p][moment]),values[f"{moment}.{name}"],1e-6,0)
                for kind, result in checks[name].items():
                    if not result["passed"]:
                        failures.append(dict(case=case["name"],parameter=name,kind=kind,**result))
            results.append(dict(case=case["name"],output=compare(output,values["output"],1e-5,1e-4),parameters=checks))
        report["devices"][device] = dict(cases=results,failures=failures)
        print(f"Official {device}: {len(failures)} failed parameter checks; max error={max((r['max_error'] for r in failures),default=0):.9g}")
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("diagnostic reports must stay in runs/")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(f"Diagnostic report: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--ieee-fp32",action="store_true",help="diagnostic control: disable TF32 in matmul and cuDNN")
    arguments = parser.parse_args()
    diagnose(arguments.output,arguments.ieee_fp32)
