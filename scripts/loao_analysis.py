#!/usr/bin/env python3
"""Error-source analysis for leave-one-architecture-out (LOAO), addressing the
reviewer: (1) which family drives the ~45% LOAO MAPE, and (2) does few-shot family
calibration (profiling k models of the held-out family) reduce it? Rigorous: 0-shot
is pure LOAO; k>0 is a realistic few-shot adaptation, NOT leakage of the test fold.
Writes results/loao_analysis.json. Reproducible."""
import sys, os, csv, json
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source"); sys.path.insert(0, "IMUNet/RONIN_torch")
import numpy as np, torch
from profile_device import MODEL_REGISTRY
from torch.utils.flop_counter import FlopCounterMode
AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(p): return list(csv.DictReader(open(os.path.join(AIOTC, "results", p))))
DEV = {"blackwell": load("profile_blackwell.csv"), "agx_orin": load("profile_agx_orin.csv"),
       "orin_nano": load("profile_orin_nano.csv")}
def lat(rows, m, pr):
    for r in rows:
        if r["model"] == m and r["precision"] == pr: return float(r["lat_med_ms"])
def mem(m):
    for r in DEV["blackwell"]:
        if r["model"] == m and r["precision"] == "fp32": return float(r["peak_mem_MB"])
def nleaf(net): return sum(1 for mo in net.modules() if len(list(mo.children())) == 0)
def feat(name):
    net, shp = MODEL_REGISTRY[name](); net = net.float().eval()
    try:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc: net(torch.randn(*shp))
        F = fc.get_total_flops()
    except Exception: F = 0.0
    return [nleaf(net), mem(name), F]
M = ["ronin_resnet18","ronin_tcn","imunet","mobilenetv2","mnasnet","efficientnet_b0","tinyodom","tlio_resnet","eqnio"]
FAM = {"ronin_resnet18":"CNN","tlio_resnet":"CNN","ronin_tcn":"TCN","tinyodom":"TCN-NAS","imunet":"mobile",
       "mobilenetv2":"mobile","mnasnet":"mobile","efficientnet_b0":"mobile","eqnio":"equiv"}
FT = {m: feat(m) for m in M}
def fitpred(tr, te):
    ap = []
    for dev, rows in DEV.items():
        X = np.array([FT[m] for m in tr]); y = np.array([lat(rows, m, "fp32") for m in tr])
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
        for m in te:
            yh = np.array([1, *FT[m]]) @ c; yt = lat(rows, m, "fp32"); ap.append(abs(yh-yt)/yt*100)
    return np.array(ap)

fams = sorted(set(FAM.values()))
per_family, allap = {}, []
for f in fams:
    te = [m for m in M if FAM[m] == f]; tr = [m for m in M if FAM[m] != f]
    ap = fitpred(tr, te); per_family[f] = round(float(ap.mean()), 0); allap.append(ap)
overall = round(float(np.concatenate(allap).mean()), 0)

fewshot = {}
for f in ["mobile", "CNN"]:  # families with >=2 members
    fam_models = [m for m in M if FAM[m] == f]; fewshot[f] = {}
    for k in [0, 1]:
        aps = []
        for held in fam_models:
            others = [m for m in fam_models if m != held]
            tr = [m for m in M if FAM[m] != f] + others[:k]
            aps.append(fitpred(tr, [held]))
        fewshot[f][k] = round(float(np.concatenate(aps).mean()), 0)

out = dict(per_family=per_family, overall=overall, fewshot=fewshot,
           worst_family=max(per_family, key=per_family.get))
json.dump(out, open(os.path.join(AIOTC, "results", "loao_analysis.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
