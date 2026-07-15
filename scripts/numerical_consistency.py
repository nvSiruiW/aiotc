#!/usr/bin/env python3
"""Numerical output-consistency of FP16 / BF16 vs FP32 (the 'output consistency'
layer of the FP16-safe decomposition).

IMPORTANT SCOPE: uses RANDOM (untrained) weights, so this measures the numerical
sensitivity of each ARCHITECTURE to reduced precision, NOT localization accuracy.
Task-level accuracy still requires trained weights + a test set (reported separately).

For each model: build in FP32 (reference), rebuild with identical weights (same seed)
in FP16/BF16, run the same input, cast output back to FP32, and report deviation.
"""
import sys, csv, statistics
sys.path.insert(0, "ronin/source")
import torch
from profile_device import MODEL_REGISTRY

DEV="cuda:0"
DT={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}
SEED=1234

def build(name, dtype):
    torch.manual_seed(SEED)                      # identical weights across precisions
    net, shape = MODEL_REGISTRY[name]()
    net = net.to(DEV).to(dtype).eval()
    if hasattr(net, "lstm"):
        net.lstm.flatten_parameters()
    return net, shape

def compare(y_ref, y):
    y=y.float().flatten(); r=y_ref.float().flatten()
    d=y-r
    rel=(d.norm()/r.norm()).item()
    mae=d.abs().mean().item()
    maxe=d.abs().max().item()
    cos=torch.nn.functional.cosine_similarity(y,r,dim=0).item()
    return rel,mae,maxe,cos

def main():
    models=list(MODEL_REGISTRY.keys())
    print("Numerical output consistency vs FP32 (RANDOM weights -> architecture sensitivity, NOT accuracy)")
    print(f"{'model':16} {'prec':5} {'relL2':>10} {'MAE':>11} {'max_abs':>11} {'cosine':>10}")
    rows=[]
    for name in models:
        torch.manual_seed(SEED)
        ref_net, shape = build(name, torch.float32)
        torch.manual_seed(SEED+1)
        x_ref = torch.randn(*shape, device=DEV)          # fixed input per model
        with torch.no_grad():
            y_ref = ref_net(x_ref).float()
        for pn,dt in DT.items():
            net,_=build(name,dt)
            with torch.no_grad():
                y=net(x_ref.to(dt))
            rel,mae,maxe,cos=compare(y_ref,y)
            print(f"{name:16} {pn:5} {rel:10.2e} {mae:11.3e} {maxe:11.3e} {cos:10.6f}")
            rows.append(dict(model=name,precision=pn,rel_L2=rel,MAE=mae,max_abs=maxe,cosine=cos))
    out="results/numerical_consistency.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    # quick fp16-vs-bf16 verdict per model
    print("\n--- fp16 vs bf16 relative-L2 (smaller = closer to fp32) ---")
    by={}
    for r in rows: by[(r['model'],r['precision'])]=r['rel_L2']
    for name in models:
        f16,b16=by[(name,'fp16')],by[(name,'bf16')]
        winner="fp16" if f16<b16 else "bf16"
        print(f"  {name:16} fp16={f16:.2e}  bf16={b16:.2e}  -> {winner} closer to fp32 "
              f"({max(f16,b16)/max(min(f16,b16),1e-12):.1f}x)")

if __name__=="__main__":
    main()
