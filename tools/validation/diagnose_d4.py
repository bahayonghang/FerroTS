"""Cross-load the failed ETTm2/H336 checkpoints without changing formal runs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from p1_acceptance import ROOT, bundle_at, read_json, sha256, write_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    args=parser.parse_args()
    suite=(ROOT/args.suite).resolve()
    info=read_json(suite/"suite.json")
    output=suite/"diagnosis"/dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/"tools/reference/.cache/Time-Series-Library"))
    from models.DLinear import Model
    from data_provider.data_loader import Dataset_ETT_minute
    dataset=Dataset_ETT_minute(SimpleNamespace(augmentation_ratio=0),str(ROOT/"data/ETT-small"),flag="test",size=[96,48,336],features="M",data_path="ETTm2.csv",timeenc=1,freq="h")
    loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0)
    rows=[]
    for seed in [2021,2022,2023]:
        official=read_json(suite/"entries"/f"official-ETTm2-h336-s{seed}.json")
        rust=read_json(suite/"entries"/f"rust-ETTm2-dlinear-cuda-h336-s{seed}.json")
        source=bundle_at(Path(rust["run"]))
        checkpoints=list((Path(official["run"])/"checkpoints").rglob("checkpoint.pth"))
        if len(checkpoints)!=1:
            raise ValueError("expected exactly one official checkpoint")
        state=torch.load(checkpoints[0],map_location="cpu",weights_only=True)
        copied=output/f"official-in-rust-{seed}"
        shutil.copytree(source,copied)
        save_file({k:v.contiguous() for k,v in state.items()},copied/"model.safetensors")
        manifest=read_json(copied/"manifest.json")
        manifest["files"]["model.safetensors"]=dict(bytes=(copied/"model.safetensors").stat().st_size,sha256=sha256(copied/"model.safetensors"))
        manifest["provenance"]["environment"]["diagnostic_only"]="official weights loaded into Rust bundle for cross-evaluation"
        write_json(copied/"manifest.json",manifest)
        proc=subprocess.run([info["binaries"]["cuda"]["path"],"evaluate","--checkpoint",str(copied),"--device","cuda:0"],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        (output/f"official-in-rust-{seed}.log").write_text(proc.stdout+proc.stderr,encoding="utf-8")
        proc.check_returncode()
        evaluated=json.loads(proc.stdout)["metrics"]["standardized"]["overall"]
        model=Model(SimpleNamespace(task_name="long_term_forecast",seq_len=96,pred_len=336,moving_avg=25,enc_in=7))
        model.load_state_dict(load_file(source/"model.safetensors"))
        model=model.cuda().eval()
        count=0
        squared=0.0
        absolute=0.0
        with torch.no_grad():
            for x,y,_,_ in loader:
                pred=model(x.float().cuda(),None,None,None).cpu().numpy().astype(np.float64)
                target=y[:,-336:,:].float().numpy().astype(np.float64)
                error=pred-target
                count+=error.size
                squared+=float(np.sum(error**2,dtype=np.float64))
                absolute+=float(np.sum(np.abs(error),dtype=np.float64))
        log=(Path(official["run"])/"stdout.log").read_text(encoding="utf-8")
        row=dict(seed=seed,official=official["metrics"],official_weights_in_rust=evaluated,rust=rust["evaluation"]["metrics"]["standardized"]["overall"],rust_weights_in_official=dict(mse=squared/count,mae=absolute/count,count=count),rust_best_epoch=rust["summary"]["best_epoch"],rust_epochs=rust["summary"]["epochs"],official_epochs=re.findall(r"Epoch: .*Steps:.*",log))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ["rust_epochs","official_epochs"]},ensure_ascii=False),flush=True)
    write_json(output/"cross-evaluation.json",dict(diagnostic_only=True,rows=rows))
    print(f"diagnosis: {output}")


if __name__=="__main__":
    main()
