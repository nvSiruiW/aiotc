#!/usr/bin/env python3
"""
Experiment 1: Feature Ablation for Kernel-Aware Cost Model
==========================================================
Compares 7 feature sets across 4 leave-out protocols.

Feature sets (paper terminology → internal names):
  1. Params only               → [params_M]
  2. FLOPs only                → [flops_M]
  3. Params + FLOPs            → [params_M, flops_M]
  4. N_exec only               → [k_real]
  5. N_exec + B_eff            → [k_real, mem_MB]
  6. N_exec + B_eff + F_eff    → [k_real, mem_MB, flops_M]
  7. N_exec + B_eff + F_eff + I_fallback → [k_real, mem_MB, flops_M, i_fb]

Protocols: LOMO, LOAO, LOPO, LODO
Metrics:   MAPE, Median APE, P95 APE, Max APE, Pearson r, Spearman rho

Dataset: all (model, precision, device) triples where:
  - non-LSTM models: k_real from kernel_profile.json (precision-independent)
  - LSTM fp16: k_real=10 (fused persistent cuDNN kernel)
  - LSTM bf16: k_real=1241 (time-step-wise unfused fallback; I_fallback=1)
  - mem_MB: precision-specific peak_mem_MB from profile CSV
  - All other features: precision-independent structural values

Run:
  cd /path/to/aiotc
  python scripts/run_cost_model_ablation.py [--output results/revision_experiments/cost_model_ablation]
"""
import argparse, csv, json, os, sys, math
from pathlib import Path
import numpy as np
import scipy.stats as ss

# ---------------------------------------------------------------------------
AIOTC = Path(__file__).resolve().parent.parent
RESULTS = AIOTC / "results"
# ---------------------------------------------------------------------------

def load_csv(fname):
    p = RESULTS / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return list(csv.DictReader(open(p)))

# ---- raw profile tables ----
PROF = {
    "blackwell": load_csv("profile_blackwell.csv"),
    "agx_orin":  load_csv("profile_agx_orin.csv"),
    "orin_nano": load_csv("profile_orin_nano.csv"),
}

kp = json.load(open(RESULTS / "kernel_profile.json"))

# ---- k_real per model (fp32 from profiler, Blackwell) ----
# For LSTM, precision-specific values are available
K_REAL_NON_LSTM = {m: v["k_real"] for m, v in kp["models"].items()}
LSTM_K = {"fp16": kp["lstm"]["fp16"], "bf16": kp["lstm"]["bf16"],
           "fp32": kp["models"].get("ronin_lstm", {}).get("k_real", kp["lstm"]["fp16"])}

# ---- structural FLOPs and params from existing validate_cost_model.py output ----
# These are computed by FlopCounterMode - we'll re-extract lightweight versions from
# a small mapping derived from the profile data (params_M column) plus stored FLOPs.
# Since we can't re-run FlopCounterMode without importing PyTorch model builders,
# we read params from the profile CSV and use kernel_profile for k_real.
# FLOPs: use values saved in model_specs.json if available, otherwise compute.
def get_flops_params():
    specs_path = AIOTC / "model_specs.json"
    if specs_path.exists():
        return json.load(open(specs_path))
    return {}

MODEL_SPECS = get_flops_params()

# Load actual FLOPs from precomputed file (computed by run: python3 -c extract FlopCounterMode)
FLOPS_FILE = AIOTC / "results" / "model_flops.json"
MODEL_FLOPS = {}
if FLOPS_FILE.exists():
    for m, v in json.load(open(FLOPS_FILE)).items():
        if v:
            MODEL_FLOPS[m] = v

def get_feat_from_profile(rows, model, precision):
    """Extract mem_MB and params_M from a profile CSV row."""
    for r in rows:
        if r["model"] == model and r["precision"] == precision:
            return float(r["peak_mem_MB"]), float(r["params_M"])
    return None, None

# ---- architecture families (for LOAO) ----
FAM = {
    "ronin_resnet18": "CNN",
    "tlio_resnet":    "CNN",
    "ronin_tcn":      "TCN",
    "tinyodom":       "TCN-NAS",
    "imunet":         "mobile",
    "mobilenetv2":    "mobile",
    "mnasnet":        "mobile",
    "efficientnet_b0":"mobile",
    "eqnio":          "equiv",
    "ronin_lstm":     "recurrent",
}

ALL_MODELS = list(K_REAL_NON_LSTM.keys()) + ["ronin_lstm"]
PRECISIONS = ["fp32", "fp16", "bf16"]

# ---- Build expanded dataset ----
# Each row: (model, precision, device, features..., latency)
def build_dataset():
    """Returns list of dicts with all features and latency per (model, prec, device)."""
    rows_out = []
    # FLOPs: use model_specs if available, else read from a stored mapping
    # We'll compute approximate FLOPs from kernel_profile's stored models dict
    flops_map = {}
    for m, v in kp["models"].items():
        # kernel_profile.json doesn't store FLOPs directly; use None → skip F_eff
        pass
    # Try to load FLOPs from model_specs.json
    flops_from_specs = {}
    if MODEL_SPECS:
        for m, spec in MODEL_SPECS.items():
            if "flops_M" in spec:
                flops_from_specs[m] = spec["flops_M"]

    # Fallback: estimate FLOPs from params using rough scaling observed in existing data
    # (From cost_model.py: params_M is already in profile CSV, we use it as F_eff proxy
    #  since FLOPs ∝ params for these architectures with fixed window size)
    # NOTE: if flops not available we'll just use params_M as F_eff too,
    # but mark feature set 6 and 7 as "F_eff≈params" in that case.

    for dev, prof_rows in PROF.items():
        seen = set()
        for r in prof_rows:
            model = r["model"]
            prec  = r["precision"]
            if (model, prec) in seen:
                continue
            seen.add((model, prec))

            lat = float(r["lat_med_ms"])
            lat_p95 = float(r.get("lat_p95_ms", r["lat_med_ms"]))
            mem_mb = float(r["peak_mem_MB"])
            params_m = float(r["params_M"])

            # N_exec (k_real)
            if model == "ronin_lstm":
                k_real = LSTM_K.get(prec, LSTM_K["fp32"])
            else:
                k_real = K_REAL_NON_LSTM.get(model, float("nan"))

            # I_fallback
            i_fb = 1.0 if (model == "ronin_lstm" and prec == "bf16") else 0.0

            # FLOPs (F_eff): use measured FLOPs from FlopCounterMode, or params as proxy
            if model in MODEL_FLOPS:
                flops_m = MODEL_FLOPS[model]["flops_M"]
            elif "flops_M" in flops_from_specs.get(model, {}):
                flops_m = flops_from_specs[model]["flops_M"]
            else:
                flops_m = params_m  # last-resort proxy (noted in output)

            rows_out.append({
                "device":   dev,
                "model":    model,
                "precision": prec,
                "family":   FAM.get(model, "unknown"),
                "params_M": params_m,
                "flops_M":  flops_m,
                "k_real":   k_real,
                "mem_MB":   mem_mb,
                "i_fb":     i_fb,
                "lat_ms":   lat,
                "lat_p95":  lat_p95,
            })
    return rows_out

DATASET = build_dataset()

# ---- Regression helpers ----
def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def predict(coef, x_row):
    return float(np.array([1.0, *x_row]) @ coef)

def ape(yhat, ytrue):
    return abs(yhat - ytrue) / ytrue * 100

def stats(apes_arr):
    a = np.array(apes_arr, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"MAPE": float("nan"), "Median_APE": float("nan"),
                "P95_APE": float("nan"), "Max_APE": float("nan")}
    return {
        "MAPE":       round(float(np.mean(a)), 2),
        "Median_APE": round(float(np.median(a)), 2),
        "P95_APE":    round(float(np.percentile(a, 95)), 2),
        "Max_APE":    round(float(np.max(a)), 2),
    }

def pearson_r(yhat_list, ytrue_list):
    a, b = np.array(yhat_list), np.array(ytrue_list)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    r, _ = ss.pearsonr(a[mask], b[mask])
    return round(float(r), 3)

def spearman_rho(yhat_list, ytrue_list):
    a, b = np.array(yhat_list), np.array(ytrue_list)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    rho, _ = ss.spearmanr(a[mask], b[mask])
    return round(float(rho), 3)

# ---- Feature set definitions ----
FEAT_SETS = {
    "1_Params":              ["params_M"],
    "2_FLOPs":               ["flops_M"],
    "3_Params+FLOPs":        ["params_M", "flops_M"],
    "4_N_exec":              ["k_real"],
    "5_N_exec+B_eff":        ["k_real", "mem_MB"],
    "6_N_exec+B_eff+F_eff":  ["k_real", "mem_MB", "flops_M"],
    "7_N_exec+B_eff+F_eff+I_fallback": ["k_real", "mem_MB", "flops_M", "i_fb"],
}

def get_xy(subset, feat_cols):
    X = np.array([[r[c] for c in feat_cols] for r in subset], float)
    y = np.array([r["lat_ms"] for r in subset], float)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return X[valid], y[valid], [subset[i] for i in range(len(subset)) if valid[i]]

# ---- Protocol implementations ----

def protocol_lomo(dataset, feat_cols):
    """Leave-one-MODEL-out: hold out all (prec, device) of one model."""
    models = sorted(set(r["model"] for r in dataset))
    all_apes, yhat_all, ytrue_all = [], [], []
    for held_model in models:
        train = [r for r in dataset if r["model"] != held_model]
        test  = [r for r in dataset if r["model"] == held_model]
        if not train or not test:
            continue
        # Fit per device to handle device-specific coefficients
        # Alternative: pool all devices (as done in existing validate_cost_model.py)
        for dev in PROF.keys():
            tr = [r for r in train if r["device"] == dev]
            te = [r for r in test  if r["device"] == dev]
            if not tr or not te:
                continue
            Xtr, ytr, _ = get_xy(tr, feat_cols)
            Xte, yte, _ = get_xy(te, feat_cols)
            if len(Xtr) < 2 or len(Xte) == 0:
                continue
            coef = fit(Xtr, ytr)
            for xi, yi in zip(Xte, yte):
                yh = predict(coef, xi)
                all_apes.append(ape(yh, yi))
                yhat_all.append(yh); ytrue_all.append(yi)
    return stats(all_apes), pearson_r(yhat_all, ytrue_all), spearman_rho(yhat_all, ytrue_all)

def protocol_loao(dataset, feat_cols):
    """Leave-one-ARCHITECTURE-family-out."""
    families = sorted(set(r["family"] for r in dataset))
    all_apes, yhat_all, ytrue_all = [], [], []
    for held_fam in families:
        train = [r for r in dataset if r["family"] != held_fam]
        test  = [r for r in dataset if r["family"] == held_fam]
        if len(train) < 3 or not test:
            continue
        for dev in PROF.keys():
            tr = [r for r in train if r["device"] == dev]
            te = [r for r in test  if r["device"] == dev]
            if not tr or not te:
                continue
            Xtr, ytr, _ = get_xy(tr, feat_cols)
            Xte, yte, _ = get_xy(te, feat_cols)
            if len(Xtr) < 2 or len(Xte) == 0:
                continue
            coef = fit(Xtr, ytr)
            for xi, yi in zip(Xte, yte):
                yh = predict(coef, xi)
                all_apes.append(ape(yh, yi))
                yhat_all.append(yh); ytrue_all.append(yi)
    return stats(all_apes), pearson_r(yhat_all, ytrue_all), spearman_rho(yhat_all, ytrue_all)

def protocol_lopo(dataset, feat_cols):
    """Leave-one-PATH-out: train on fp32, predict fp16 and bf16.
    Special focus: LSTM-BF16 as the hardest path."""
    all_apes, yhat_all, ytrue_all = [], [], []
    for dev in PROF.keys():
        train = [r for r in dataset if r["device"] == dev and r["precision"] == "fp32"]
        test  = [r for r in dataset if r["device"] == dev and r["precision"] != "fp32"]
        if not train or not test:
            continue
        Xtr, ytr, _ = get_xy(train, feat_cols)
        if len(Xtr) < 2:
            continue
        coef = fit(Xtr, ytr)
        for r in test:
            xi = [r[c] for c in feat_cols]
            if not all(math.isfinite(v) for v in xi):
                continue
            yh = predict(coef, xi)
            yi = r["lat_ms"]
            all_apes.append(ape(yh, yi))
            yhat_all.append(yh); ytrue_all.append(yi)
    return stats(all_apes), pearson_r(yhat_all, ytrue_all), spearman_rho(yhat_all, ytrue_all)

def protocol_lodo(dataset, feat_cols):
    """Leave-one-DEVICE-out: fit on 2 devices, predict 3rd (fp32 only)."""
    devices = list(PROF.keys())
    all_apes, yhat_all, ytrue_all = [], [], []
    for held_dev in devices:
        train = [r for r in dataset if r["device"] != held_dev and r["precision"] == "fp32"]
        test  = [r for r in dataset if r["device"] == held_dev  and r["precision"] == "fp32"]
        if not train or not test:
            continue
        Xtr, ytr, _ = get_xy(train, feat_cols)
        Xte, yte, _ = get_xy(test, feat_cols)
        if len(Xtr) < 2 or len(Xte) == 0:
            continue
        coef = fit(Xtr, ytr)
        for xi, yi in zip(Xte, yte):
            yh = predict(coef, xi)
            all_apes.append(ape(yh, yi))
            yhat_all.append(yh); ytrue_all.append(yi)
    return stats(all_apes), pearson_r(yhat_all, ytrue_all), spearman_rho(yhat_all, ytrue_all)

# ---- Run all protocols × feature sets ----
def run_ablation():
    results = {}
    protocols = {
        "LOMO": protocol_lomo,
        "LOAO": protocol_loao,
        "LOPO": protocol_lopo,
        "LODO": protocol_lodo,
    }
    for fs_name, feat_cols in FEAT_SETS.items():
        results[fs_name] = {}
        print(f"\n--- {fs_name} ---")
        for proto_name, proto_fn in protocols.items():
            s, r_pearson, rho = proto_fn(DATASET, feat_cols)
            results[fs_name][proto_name] = {**s, "Pearson_r": r_pearson, "Spearman_rho": rho}
            print(f"  {proto_name}: MAPE={s['MAPE']:.1f}%  Med={s['Median_APE']:.1f}%  "
                  f"P95={s['P95_APE']:.1f}%  Max={s['Max_APE']:.1f}%  "
                  f"r={r_pearson:.3f}  ρ={rho:.3f}")
    return results

# ---- Also compute LOPO split specifically for LSTM-BF16 (fallback path) ----
def lopo_fallback_detail(feat_cols_no_fb, feat_cols_with_fb):
    """Show prediction error on LSTM-BF16 with and without I_fallback.

    Protocol: cross-device calibration.
    Train on 2 devices (all models, all precisions, including LSTM-BF16).
    Predict LSTM-BF16 on the 3rd device.
    Without I_fallback: model cannot distinguish BF16 from FP16 (same structural features).
    With I_fallback: model sees I_fallback=1 → learns a correction coefficient from LSTM-BF16
                     data on training devices.
    """
    print("\n--- Cross-device LSTM-BF16 fallback path detail (LODO protocol) ---")
    devices = list(PROF.keys())
    results = {}
    for held_dev in devices:
        train_all = [r for r in DATASET if r["device"] != held_dev]
        test_fb   = [r for r in DATASET
                     if r["device"] == held_dev and r["model"] == "ronin_lstm"
                     and r["precision"] == "bf16"]
        if not train_all or not test_fb:
            continue

        for label, fc in [("w/o I_fallback", feat_cols_no_fb),
                           ("w/ I_fallback",  feat_cols_with_fb)]:
            # For "w/o I_fallback": use the no-fb feature set, so I_fallback column excluded
            # For training without I_fallback, BF16-LSTM and FP16-LSTM look identical
            # (same k_real if we use k_leaf; or k_real=1241 but no I_fb correction)
            Xtr, ytr, _ = get_xy(train_all, fc)
            if len(Xtr) < 4:
                continue
            coef = fit(Xtr, ytr)
            for r in test_fb:
                xi = [r[c] for c in fc]
                yh = predict(coef, xi)
                yi = r["lat_ms"]
                pct_err = ape(yh, yi)
                print(f"  {held_dev:12s} {label}: pred={yh:.2f}ms  true={yi:.2f}ms  APE={pct_err:.1f}%")
                key = f"{held_dev}_{label.replace(' ','_').replace('/','_')}"
                results[key] = {"pred": round(yh, 3), "true": round(yi, 3), "APE": round(pct_err, 1)}
    return results

# ---- Output formatters ----
def to_csv(results, out_path):
    rows = []
    for fs_name, protos in results.items():
        if fs_name.startswith("_"):  # skip internal detail dicts
            continue
        row = {"Feature_Set": fs_name}
        for proto, metrics in protos.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                row[f"{proto}_{k}"] = v
        rows.append(row)
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")

def to_markdown(results, out_path):
    lines = ["# Cost Model Feature Ablation\n",
             "Dataset: all (model × precision × device); LOMO/LOAO/LOPO/LODO protocols.",
             "LOPO: train on fp32, predict fp16+bf16 (worst case: LSTM-BF16 fallback path).",
             "LODO: train on 2 devices, predict held-out device (fp32).\n",
             "## Main Table\n"]

    # header
    cols = ["Feature Set", "LOMO MAPE", "LOAO MAPE", "LOPO MAPE", "LODO MAPE",
            "Median APE", "P95 APE", "Max APE", "Pearson r", "Spearman ρ"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for fs_name, protos in results.items():
        lomo = protos.get("LOMO", {})
        loao = protos.get("LOAO", {})
        lopo = protos.get("LOPO", {})
        lodo = protos.get("LODO", {})
        # Median and P95 average across protocols (use LOMO as representative)
        med  = lomo.get("Median_APE", "—")
        p95  = lomo.get("P95_APE", "—")
        mx   = lomo.get("Max_APE", "—")
        r_p  = lomo.get("Pearson_r", "—")
        rho  = lomo.get("Spearman_rho", "—")
        row = [
            fs_name.replace("_", " "),
            f"{lomo.get('MAPE', '—'):.1f}%" if isinstance(lomo.get('MAPE'), float) else "—",
            f"{loao.get('MAPE', '—'):.1f}%" if isinstance(loao.get('MAPE'), float) else "—",
            f"{lopo.get('MAPE', '—'):.1f}%" if isinstance(lopo.get('MAPE'), float) else "—",
            f"{lodo.get('MAPE', '—'):.1f}%" if isinstance(lodo.get('MAPE'), float) else "—",
            f"{med:.1f}%" if isinstance(med, float) else str(med),
            f"{p95:.1f}%" if isinstance(p95, float) else str(p95),
            f"{mx:.1f}%"  if isinstance(mx,  float) else str(mx),
            f"{r_p:.3f}"  if isinstance(r_p, float) else str(r_p),
            f"{rho:.3f}"  if isinstance(rho, float) else str(rho),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "\n## Notes",
        "- Feature columns: Params=params_M, FLOPs≈params_M (used as F_eff proxy where FlopCounterMode "
        "output unavailable), N_exec=k_real (measured CUDA kernel launch count from torch.profiler), "
        "B_eff=peak_mem_MB (device-measured working set), I_fallback=1 for ronin_lstm+bf16 else 0.",
        "- For LSTM-BF16: k_real=1241 (unfused time-step-wise path) vs k_real=10 (fused FP16 path): 124× jump.",
        "- LODO uses fp32 latency only; structural model underperforms vs 1-scalar transfer (M5).",
        "- See fallback_ablation.md for LSTM-BF16 path-specific analysis.",
    ]

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

def to_latex(results, out_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feature ablation for the kernel-aware cost model. "
        r"MAPE (\%) across four leave-out protocols. Lowest MAPE per column in \textbf{bold}.}",
        r"\label{tab:cost_model_ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Feature Set & LOMO & LOAO & LOPO & LODO & Med.~APE & P95~APE \\",
        r"\midrule",
    ]
    # find per-column best
    col_keys = ["LOMO_MAPE", "LOAO_MAPE", "LOPO_MAPE", "LODO_MAPE", "LOMO_Median_APE", "LOMO_P95_APE"]
    best = {}
    for ck in col_keys:
        proto, metric = ck.split("_", 1)
        vals = [results[fs][proto].get(metric, float("inf")) for fs in results
                if isinstance(results[fs].get(proto, {}).get(metric), float)]
        best[ck] = min(vals) if vals else float("inf")

    for fs_name, protos in results.items():
        lomo = protos.get("LOMO", {})
        loao = protos.get("LOAO", {})
        lopo = protos.get("LOPO", {})
        lodo = protos.get("LODO", {})
        def fmt(v, ck):
            if not isinstance(v, float):
                return "—"
            s = f"{v:.1f}"
            return r"\textbf{" + s + r"}" if abs(v - best.get(ck, float("inf"))) < 0.01 else s
        label = fs_name.replace("_", r"\_").replace("+", r"$+$")
        cells = [
            label,
            fmt(lomo.get("MAPE"), "LOMO_MAPE"),
            fmt(loao.get("MAPE"), "LOAO_MAPE"),
            fmt(lopo.get("MAPE"), "LOPO_MAPE"),
            fmt(lodo.get("MAPE"), "LODO_MAPE"),
            fmt(lomo.get("Median_APE"), "LOMO_Median_APE"),
            fmt(lomo.get("P95_APE"), "LOMO_P95_APE"),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Cost model feature ablation")
    parser.add_argument("--output", default="results/revision_experiments/cost_model_ablation",
                        help="Output path prefix (without extension)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (unused, for reproducibility)")
    args = parser.parse_args()

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {len(DATASET)} rows, "
          f"{len(set(r['model'] for r in DATASET))} models, "
          f"{len(set(r['device'] for r in DATASET))} devices, "
          f"{len(set(r['precision'] for r in DATASET))} precisions")
    print(f"I_fallback rows: {sum(1 for r in DATASET if r['i_fb']==1)} (LSTM-BF16)")

    results = run_ablation()

    # Extra: LOPO fallback detail
    fb_detail = lopo_fallback_detail(
        feat_cols_no_fb=["k_real", "mem_MB", "flops_M"],
        feat_cols_with_fb=["k_real", "mem_MB", "flops_M", "i_fb"],
    )
    results["_fallback_path_detail"] = fb_detail

    # Save
    json.dump(results, open(str(out_prefix) + ".json", "w"), indent=2)
    to_csv(results, str(out_prefix) + ".csv")
    to_markdown(results, str(out_prefix) + ".md")
    to_latex(results, str(out_prefix) + ".tex")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Feature Set':<35} {'LOMO':>8} {'LOAO':>8} {'LOPO':>8} {'LODO':>8}")
    for fs, protos in results.items():
        if fs.startswith("_"):
            continue
        lomo = protos.get("LOMO", {}).get("MAPE", float("nan"))
        loao = protos.get("LOAO", {}).get("MAPE", float("nan"))
        lopo = protos.get("LOPO", {}).get("MAPE", float("nan"))
        lodo = protos.get("LODO", {}).get("MAPE", float("nan"))
        print(f"  {fs:<33} {lomo:>7.1f}% {loao:>7.1f}% {lopo:>7.1f}% {lodo:>7.1f}%")

if __name__ == "__main__":
    main()
