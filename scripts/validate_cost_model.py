#!/usr/bin/env python3
"""Rigorous validation of the cost model against the reviewer's demand:
Pearson is NOT enough (a 2x-biased predictor can still have r~1). We report
absolute leave-out error (MAPE / Median APE / Max APE) and compare the
kernel/profile-aware model against three naive baselines:
  B1 params-only        t = a*params + b
  B2 FLOPs-only         t = a*flops  + b
  B3 params+FLOPs        t = a*params + b*flops + c
  M4 structural (ours)   t = a*K + b*W + c*F + d        (K=launches, W=working set, F=flops)
  M5 profile-transfer    t_dev = s * t_reference        (single per-device scalar)

Only if M4/M5 beat B1-B3 on HELD-OUT MAPE can we claim predictability; this also
rules out the "scale correlation" artifact (if size drove latency, B1 would win).

Reproducible: features are extracted here from the exact model builders; latencies
are read from the committed profile_*.csv. Run:
  python scripts/validate_cost_model.py
"""
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
def mem(rows, m, pr):
    for r in rows:
        if r["model"] == m and r["precision"] == pr: return float(r["peak_mem_MB"])

# ---- reproducible structural features ----
def n_leaf(net): return sum(1 for mo in net.modules() if len(list(mo.children())) == 0)
def extract(name):
    net, shp = MODEL_REGISTRY[name](); net = net.float().eval()
    T = shp[1] if name == "ronin_lstm" else 1
    K = n_leaf(net) * T
    params = sum(p.numel() for p in net.parameters())
    try:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc: net(torch.randn(*shp))
        F = fc.get_total_flops()
    except Exception: F = float("nan")
    return dict(K=float(K), F=float(F), params=float(params),
                W=mem(DEV["blackwell"], name, "fp32"))
# non-recurrent set (recurrent = documented exception whose kernel path is precision-dependent)
MODELS = ["ronin_resnet18","ronin_tcn","imunet","mobilenetv2","mnasnet",
          "efficientnet_b0","tinyodom","tlio_resnet","eqnio"]
FAM = {"ronin_resnet18":"CNN","tlio_resnet":"CNN","ronin_tcn":"TCN","tinyodom":"TCN-NAS",
       "imunet":"mobile","mobilenetv2":"mobile","mnasnet":"mobile","efficientnet_b0":"mobile","eqnio":"equiv"}
print("Extracting reproducible features (FLOPs via FlopCounterMode, K=leaf launches, W=measured MB)...")
FEAT = {m: extract(m) for m in MODELS}
for m in MODELS: print(f"  {m:16s} K={FEAT[m]['K']:.0f} F={FEAT[m]['F']/1e6:.1f}M params={FEAT[m]['params']/1e6:.3f}M W={FEAT[m]['W']:.1f}MB")

PREDICTORS = {
    "B1 params": lambda f: [f["params"]],
    "B2 FLOPs":  lambda f: [f["F"]],
    "B3 par+FLOP": lambda f: [f["params"], f["F"]],
    "M4 struct(ours)": lambda f: [f["K"], f["W"], f["F"]],
}
def fit_predict(train_models, test_models, cols_fn, prec="fp32"):
    """Fit per device, pool APE across devices."""
    apes = []
    for dev, rows in DEV.items():
        Xtr = np.array([cols_fn(FEAT[m]) for m in train_models], float)
        ytr = np.array([lat(rows, m, prec) for m in train_models])
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        for m in test_models:
            yh = np.array([1, *cols_fn(FEAT[m])]) @ coef
            yt = lat(rows, m, prec)
            apes.append(abs(yh - yt) / yt * 100)
    return np.array(apes)
def stats(apes): return dict(MAPE=round(float(np.mean(apes)),1), Median=round(float(np.median(apes)),1), Max=round(float(np.max(apes)),1))

results = {}

# ---- Protocol 1: leave-one-MODEL-out ----
print("\n=== Protocol: leave-one-MODEL-out (fp32, pooled over 3 devices) ===")
results["LOMO"] = {}
for name, fn in PREDICTORS.items():
    apes = []
    for held in MODELS:
        tr = [m for m in MODELS if m != held]
        apes.append(fit_predict(tr, [held], fn))
    s = stats(np.concatenate(apes)); results["LOMO"][name] = s
    print(f"  {name:16s} MAPE={s['MAPE']:5.1f}%  Median={s['Median']:5.1f}%  Max={s['Max']:5.1f}%")

# ---- Protocol 2: leave-one-ARCHITECTURE-family-out ----
print("\n=== Protocol: leave-one-ARCHITECTURE-family-out (fp32) ===")
results["LOAO"] = {}
fams = sorted(set(FAM.values()))
for name, fn in PREDICTORS.items():
    apes = []
    for fam in fams:
        test = [m for m in MODELS if FAM[m] == fam]; tr = [m for m in MODELS if FAM[m] != fam]
        if len(tr) < 3: continue
        apes.append(fit_predict(tr, test, fn))
    s = stats(np.concatenate(apes)); results["LOAO"][name] = s
    print(f"  {name:16s} MAPE={s['MAPE']:5.1f}%  Median={s['Median']:5.1f}%  Max={s['Max']:5.1f}%")

# ---- Protocol 3: leave-one-DEVICE-out (structural vs profile-transfer w/ 1 calib model) ----
print("\n=== Protocol: leave-one-DEVICE-out ===")
print("    (structural: fit features->latency on the 2 seen devices, predict 3rd;")
print("     profile-transfer M5: 1 calibration model gives scalar s, predict the rest)")
results["LODO"] = {}
# structural baselines/ours: fit on the two other devices (pooled), predict held device
def lodo_struct(cols_fn, prec="fp32"):
    apes = []
    for held, rows in DEV.items():
        seen = [dv for dv in DEV if dv != held]
        Xtr, ytr = [], []
        for dv in seen:
            for m in MODELS:
                Xtr.append(cols_fn(FEAT[m])); ytr.append(lat(DEV[dv], m, prec))
        Xtr = np.array(Xtr, float); A = np.column_stack([np.ones(len(Xtr)), ytr and Xtr])
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), np.array(ytr), rcond=None)
        for m in MODELS:
            yh = np.array([1, *cols_fn(FEAT[m])]) @ coef; yt = lat(rows, m, prec)
            apes.append(abs(yh - yt)/yt*100)
    return np.array(apes)
for name, fn in PREDICTORS.items():
    s = stats(lodo_struct(fn)); results["LODO"][name] = s
    print(f"  {name:16s} MAPE={s['MAPE']:5.1f}%  Median={s['Median']:5.1f}%  Max={s['Max']:5.1f}%")
# profile-transfer: one calibration model -> scalar, predict others (report worst calib choice too)
def lodo_transfer(prec="fp32", ref="blackwell"):
    apes, worst = [], []
    for held, rows in DEV.items():
        if held == ref: continue
        for calib in MODELS:
            s = lat(rows, calib, prec) / lat(DEV[ref], calib, prec)
            ap = [abs(s*lat(DEV[ref], m, prec) - lat(rows, m, prec))/lat(rows, m, prec)*100
                  for m in MODELS if m != calib]
            apes += ap; worst.append(np.mean(ap))
    return np.array(apes)
s = stats(lodo_transfer()); results["LODO"]["M5 transfer(1-calib)"] = s
print(f"  {'M5 transfer(1-calib)':16s} MAPE={s['MAPE']:5.1f}%  Median={s['Median']:5.1f}%  Max={s['Max']:5.1f}%")

# ---- Protocol 4: leave-one-PATH-out (train fp32, predict fp16 profile) ----
print("\n=== Protocol: leave-one-PATH-out (fit on fp32, predict fp16 latency) ===")
results["LOPO"] = {}
def lopo(cols_fn):
    apes = []
    for dev, rows in DEV.items():
        Xtr = np.array([cols_fn(FEAT[m]) for m in MODELS], float)
        ytr = np.array([lat(rows, m, "fp32") for m in MODELS])
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
        for m in MODELS:
            yh = np.array([1, *cols_fn(FEAT[m])]) @ coef; yt = lat(rows, m, "fp16")
            apes.append(abs(yh-yt)/yt*100)
    return np.array(apes)
for name, fn in PREDICTORS.items():
    s = stats(lopo(fn)); results["LOPO"][name] = s
    print(f"  {name:16s} MAPE={s['MAPE']:5.1f}%  Median={s['Median']:5.1f}%  Max={s['Max']:5.1f}%")

json.dump(results, open(os.path.join(AIOTC, "results", "cost_model_validation.json"), "w"), indent=2)
print("\nwrote results/cost_model_validation.json")
