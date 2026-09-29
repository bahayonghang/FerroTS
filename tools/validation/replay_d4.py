"""Diagnostic only: replay Rust initialization and batch order in pinned TSLib."""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from p1_acceptance import ROOT, read_json, write_json, artifacts


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    args=parser.parse_args()
    suite=(ROOT/args.suite).resolve()
    inputs=suite/"diagnosis/replay-input"
    output=suite/"diagnosis"/dt.datetime.now().strftime("replay-%Y%m%d-%H%M%S")
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/"tools/reference/.cache/Time-Series-Library"))
    from models.DLinear import Model
    from data_provider.data_loader import Dataset_ETT_minute
    def dataset(flag):
        return Dataset_ETT_minute(SimpleNamespace(augmentation_ratio=0),str(ROOT/"data/ETT-small"),flag=flag,size=[96,48,336],features="M",data_path="ETTm2.csv",timeenc=1,freq="h")
    datasets={flag:dataset(flag) for flag in ["train","val","test"]}
    arrays={flag:torch.from_numpy(ds.data_x.astype(np.float32)).cuda() for flag,ds in datasets.items()}
    def batches(flag, order):
        values=arrays[flag]
        for start in range(0,len(order),32):
            indices=torch.tensor(order[start:start+32],device="cuda")
            yield values[indices[:,None]+torch.arange(96,device="cuda")],values[indices[:,None]+96+torch.arange(336,device="cuda")]
    def evaluate(model,flag):
        squared=0.0
        absolute=0.0
        count=0
        with torch.no_grad():
            for x,y in batches(flag,list(range(len(datasets[flag])))):
                pred=model(x,None,None,None)
                error=pred.double()-y.double()
                squared+=error.square().sum().item()
                absolute+=error.abs().sum().item()
                count+=error.numel()
        return dict(mse=squared/count,mae=absolute/count,count=count)
    results=[]
    for seed in [2021,2022,2023]:
        entry=read_json(suite/"entries"/f"rust-ETTm2-dlinear-cuda-h336-s{seed}.json")
        orders=read_json(inputs/f"orders-{seed}.json")
        model=Model(SimpleNamespace(task_name="long_term_forecast",seq_len=96,pred_len=336,moving_avg=25,enc_in=7))
        model.load_state_dict(load_file(inputs/f"initial-{seed}.safetensors"))
        model=model.cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        epochs=[]
        best=None
        best_state=None
        for epoch in entry["summary"]["epochs"]:
            e=epoch["epoch"]
            for group in optimizer.param_groups:
                group["lr"]=epoch["learning_rate"]
            for x,y in batches("train",orders[e-1]):
                optimizer.zero_grad()
                loss=torch.nn.functional.mse_loss(model(x,None,None,None),y)
                loss.backward()
                optimizer.step()
            val=evaluate(model,"val")["mse"]
            if best is None or val<best:
                best=val
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            bundles=[p.parent for p in (Path(entry["run"])/"checkpoints").glob("*/manifest.json") if read_json(p)["epoch"]==e]
            expected=load_file(bundles[0]/"model.safetensors")
            max_difference=max(float((p.detach().cpu()-expected[k]).abs().max()) for k,p in model.state_dict().items())
            epochs.append(dict(epoch=e,validation_mse=val,rust_validation_mse=epoch["validation_loss"],max_parameter_abs_difference=max_difference))
            print(f"seed={seed} epoch={e} validation={val:.12g} Rust={epoch['validation_loss']:.12g} max_parameter_diff={max_difference:.6g}",flush=True)
        model.load_state_dict(best_state)
        metrics=evaluate(model,"test")
        row=dict(seed=seed,epochs=epochs,replay=metrics,rust=entry["evaluation"]["metrics"]["standardized"]["overall"])
        results.append(row)
        print(f"seed={seed} replay MSE={metrics['mse']:.12g} Rust={row['rust']['mse']:.12g}",flush=True)
    write_json(output/"replay.json",dict(diagnostic_only=True,rows=results,inputs=artifacts(list(inputs.glob("*.json"))+list(inputs.glob("*.safetensors")))))
    print(f"diagnosis: {output}")


if __name__=="__main__":
    main()
