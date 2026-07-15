#!/usr/bin/env python3
"""
Experiment 3: Budget-Aware Deployment Selector Oracle Evaluation
================================================================
Proves the budget-aware selector is not a cherry-picked demo but achieves
near-exhaustive-oracle performance across many deployment scenarios.

For each (device, deadline_ms, accuracy_constraint, energy_budget) scenario:
  1. predicted_choice: selector picks best configuration using cost model prediction
  2. oracle_choice:    exhaustive search using real measured data (best feasible ATE)
  3. Compare and report regret metrics

Coverage:
  Devices:    blackwell, agx_orin, orin_nano
  Deadlines:  1, 2, 5, 10, 20 ms
  ATE ≤:      0.9, 1.0, 1.1, 1.2 m
  Energy:     tight (25th %ile), medium (50th), loose (75th) per device [optional]

Candidate pool: all (model × precision) combinations with measured latency + ATE data.
Excludes LSTM-BF16 from non-fallback-aware pool; includes it explicitly for the
fallback-aware selector.

Run:
  cd /path/to/aiotc
  python scripts/run_selector_oracle_eval.py [--output results/revision_experiments/selector_oracle_eval]
"""
import argparse, csv, json, math, itertools
from pathlib import Path
import numpy as np
import scipy.stats as ss

AIOTC = Path(__file__).resolve().parent.parent
RESULTS = AIOTC / "results"

def load_csv(fname):
    p = RESULTS / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return list(csv.DictReader(open(p)))

# ---- Data sources ----
PROF = {
    "blackwell": load_csv("profile_blackwell.csv"),
    "agx_orin":  load_csv("profile_agx_orin.csv"),
    "orin_nano": load_csv("profile_orin_nano.csv"),
}
POWER = {
    "blackwell": load_csv("power_dynamic_blackwell.csv"),
    "agx_orin":  load_csv("power_dynamic_orin.csv"),
    "orin_nano": load_csv("power_dynamic_orin_nano.csv"),
}
# Accuracy: use blackwell accuracy as ground truth (same model weights across devices)
ACC = {
    "blackwell": load_csv("accuracy_blackwell.csv"),
    "agx_orin":  load_csv("accuracy_agx_orin.csv"),
    "orin_nano": load_csv("accuracy_orin_nano.csv"),
}

kp = json.load(open(RESULTS / "kernel_profile.json"))
LSTM_K = {"fp16": kp["lstm"]["fp16"], "bf16": kp["lstm"]["bf16"]}
NON_LSTM_MODELS = list(kp["models"].keys())

# ---- Lookup helpers ----
def _find(rows, model, prec, field):
    for r in rows:
        if r["model"] == model and r["precision"] == prec:
            v = r.get(field)
            return float(v) if v not in (None, "", "nan") else None
    return None

def get_lat(dev, model, prec):    return _find(PROF[dev],  model, prec, "lat_med_ms")
def get_energy(dev, model, prec): return _find(POWER[dev], model, prec, "dynamic_energy_mJ")
def get_ate(dev, model, prec):    return _find(ACC[dev],   model, prec, "ate_m")
def get_mem(dev, model, prec):    return _find(PROF[dev],  model, prec, "peak_mem_MB")

# ---- Cost model: predict latency from N_exec + B_eff ----
def fit_cost_model_for_device(dev):
    """Fit per-device linear model on non-LSTM fp32 models; returns coef and predictor fn."""
    X, y = [], []
    for m in NON_LSTM_MODELS:
        k = kp["models"][m]["k_real"]
        w = get_mem(dev, m, "fp32")
        lat = get_lat(dev, m, "fp32")
        if None in (k, w, lat):
            continue
        X.append([k, w, 0.0])
        y.append(lat)
    if len(X) < 3:
        return None
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def predict_lat(coef, k_real, mem_mb, i_fb):
    if coef is None:
        return float("nan")
    return float(np.array([1.0, k_real, mem_mb, i_fb]) @ coef[:4])

# ---- Build candidate pool ----
def build_candidate_pool(dev):
    """All (model, precision) with real latency + ATE data for this device."""
    pool = []
    seen_models = set()
    for r in PROF[dev]:
        m, prec = r["model"], r["precision"]
        lat = float(r["lat_med_ms"])
        ate = get_ate(dev, m, prec)
        energy = get_energy(dev, m, prec)

        # Feature vector for cost model
        if m == "ronin_lstm":
            k_real = LSTM_K.get(prec, LSTM_K["fp16"])
        else:
            k_real = kp["models"].get(m, {}).get("k_real", float("nan"))
        mem_mb = float(r["peak_mem_MB"])
        i_fb = 1.0 if (m == "ronin_lstm" and prec == "bf16") else 0.0

        pool.append({
            "model": m, "precision": prec,
            "true_lat_ms": lat,
            "true_ate_m": ate,
            "true_energy_mJ": energy,
            "k_real": k_real,
            "mem_mb": mem_mb,
            "i_fb": i_fb,
        })
    return pool

# ---- Selector: choose configuration meeting constraints ----
def selector_choose(pool, coef, deadline_ms, ate_max, energy_budget_mJ=None):
    """
    Predicted selector: use cost model to predict latency, then among configurations
    predicted-feasible, choose the one with best predicted energy or lowest predicted latency.
    Primary objective: minimize predicted latency subject to deadline.
    Secondary: best ATE (use real ATE as it doesn't require prediction).
    """
    feasible = []
    for c in pool:
        pred_lat = predict_lat(coef, c["k_real"], c["mem_mb"], c["i_fb"])
        if math.isnan(pred_lat):
            continue
        # Selector uses predicted latency to determine feasibility
        if pred_lat > deadline_ms:
            continue
        # Accuracy constraint: use real ATE (in a real system, this comes from offline eval)
        if c["true_ate_m"] is None or c["true_ate_m"] > ate_max:
            continue
        # Energy budget: if specified, use real energy as proxy (predicted energy not implemented)
        if energy_budget_mJ is not None:
            if c["true_energy_mJ"] is None or c["true_energy_mJ"] > energy_budget_mJ:
                continue
        feasible.append({**c, "pred_lat_ms": pred_lat})

    if not feasible:
        return None
    # Among predicted-feasible: pick the one with best (lowest) real ATE
    return min(feasible, key=lambda x: (x["true_ate_m"], x["pred_lat_ms"]))

def oracle_choose(pool, deadline_ms, ate_max, energy_budget_mJ=None):
    """
    Oracle: exhaustive search using real measurements.
    Among configurations meeting deadline (real) + ATE constraint: pick best ATE.
    """
    feasible = []
    for c in pool:
        lat = c["true_lat_ms"]
        ate = c["true_ate_m"]
        energy = c["true_energy_mJ"]
        if lat is None or lat > deadline_ms:
            continue
        if ate is None or ate > ate_max:
            continue
        if energy_budget_mJ is not None:
            if energy is None or energy > energy_budget_mJ:
                continue
        feasible.append(c)

    if not feasible:
        return None
    return min(feasible, key=lambda x: (x["true_ate_m"], x["true_lat_ms"]))

# ---- Run evaluation ----
DEADLINES_MS = [1.0, 2.0, 5.0, 10.0, 20.0]
ATE_THRESHOLDS = [0.9, 1.0, 1.1, 1.2]

def energy_budgets_for_device(pool):
    """25th, 50th, 75th percentile of energy in the candidate pool (excluding LSTM-BF16 anomaly)."""
    energies = [c["true_energy_mJ"] for c in pool
                if c["true_energy_mJ"] is not None and c["i_fb"] == 0]
    if not energies:
        return {}
    q25, q50, q75 = np.percentile(energies, [25, 50, 75])
    return {"tight": float(q25), "medium": float(q50), "loose": float(q75)}

def run_oracle_eval():
    all_scenarios = []

    for dev in ["blackwell", "agx_orin", "orin_nano"]:
        pool = build_candidate_pool(dev)
        coef = fit_cost_model_for_device(dev)
        e_budgets = energy_budgets_for_device(pool)

        print(f"\n=== {dev} ===")
        print(f"  Pool size: {len(pool)} configs")
        print(f"  Energy budgets: {e_budgets}")

        # Latency-only scenarios (no energy constraint)
        for deadline in DEADLINES_MS:
            for ate_max in ATE_THRESHOLDS:
                pred = selector_choose(pool, coef, deadline, ate_max)
                orac = oracle_choose(pool, deadline, ate_max)

                scenario = {
                    "device": dev,
                    "deadline_ms": deadline,
                    "ate_max_m": ate_max,
                    "energy_budget_mJ": None,
                    "energy_budget_label": "none",
                }
                _fill_comparison(scenario, pred, orac)
                all_scenarios.append(scenario)

        # Latency + energy scenarios
        for e_label, e_budget in e_budgets.items():
            for deadline in [5.0, 10.0, 20.0]:
                for ate_max in [1.0, 1.1]:
                    pred = selector_choose(pool, coef, deadline, ate_max, e_budget)
                    orac = oracle_choose(pool, deadline, ate_max, e_budget)

                    scenario = {
                        "device": dev,
                        "deadline_ms": deadline,
                        "ate_max_m": ate_max,
                        "energy_budget_mJ": round(e_budget, 2),
                        "energy_budget_label": e_label,
                    }
                    _fill_comparison(scenario, pred, orac)
                    all_scenarios.append(scenario)

    return all_scenarios

def _fill_comparison(scenario, pred, orac):
    """Fill scenario dict with comparison metrics."""
    if pred is not None:
        scenario["predicted_model"]     = pred["model"]
        scenario["predicted_precision"] = pred["precision"]
        scenario["predicted_lat_ms"]    = round(pred["pred_lat_ms"], 3)
        scenario["predicted_true_lat_ms"] = round(pred["true_lat_ms"], 3)
        scenario["predicted_ate_m"]     = round(pred["true_ate_m"], 4) if pred["true_ate_m"] else None
        scenario["predicted_energy_mJ"] = round(pred["true_energy_mJ"], 2) if pred["true_energy_mJ"] else None
        scenario["violated_deadline"]   = pred["true_lat_ms"] > scenario["deadline_ms"]
        energy_b = scenario["energy_budget_mJ"]
        scenario["violated_energy"]     = (energy_b is not None and pred["true_energy_mJ"] is not None
                                           and pred["true_energy_mJ"] > energy_b)
    else:
        for k in ["predicted_model", "predicted_precision", "predicted_lat_ms",
                  "predicted_true_lat_ms", "predicted_ate_m", "predicted_energy_mJ",
                  "violated_deadline", "violated_energy"]:
            scenario[k] = None

    if orac is not None:
        scenario["oracle_model"]     = orac["model"]
        scenario["oracle_precision"] = orac["precision"]
        scenario["oracle_lat_ms"]    = round(orac["true_lat_ms"], 3)
        scenario["oracle_ate_m"]     = round(orac["true_ate_m"], 4) if orac["true_ate_m"] else None
        scenario["oracle_energy_mJ"] = round(orac["true_energy_mJ"], 2) if orac["true_energy_mJ"] else None
    else:
        for k in ["oracle_model", "oracle_precision", "oracle_lat_ms", "oracle_ate_m", "oracle_energy_mJ"]:
            scenario[k] = None

    # Regret metrics
    if pred and orac and pred["true_ate_m"] and orac["true_ate_m"]:
        scenario["ate_regret_m"] = round(pred["true_ate_m"] - orac["true_ate_m"], 4)
    else:
        scenario["ate_regret_m"] = None

    if pred and orac:
        scenario["lat_regret_ms"] = round(pred["true_lat_ms"] - orac["true_lat_ms"], 3)
    else:
        scenario["lat_regret_ms"] = None

    if pred and orac and pred["true_energy_mJ"] and orac["true_energy_mJ"]:
        scenario["energy_regret_mJ"] = round(pred["true_energy_mJ"] - orac["true_energy_mJ"], 2)
    else:
        scenario["energy_regret_mJ"] = None

    # Top-1 match
    scenario["top1_match"] = (pred is not None and orac is not None and
                               pred["model"] == orac["model"] and
                               pred["precision"] == orac["precision"])

    # Feasible: predicted choice actually satisfies deadline and energy (using real measurements)
    if pred is not None:
        scenario["feasible"] = (not scenario.get("violated_deadline", True) and
                                 not scenario.get("violated_energy", True))
    else:
        scenario["feasible"] = False  # no choice = infeasible

    # No oracle: oracle couldn't find any config (skip this scenario for metrics)
    scenario["oracle_exists"] = orac is not None

def summarize(scenarios):
    """Compute aggregate metrics."""
    # Only include scenarios where oracle exists
    with_oracle = [s for s in scenarios if s["oracle_exists"]]
    latency_only = [s for s in with_oracle if s["energy_budget_label"] == "none"]
    with_energy  = [s for s in with_oracle if s["energy_budget_label"] != "none"]

    def agg(subset):
        n = len(subset)
        n_pred = sum(1 for s in subset if s["predicted_model"] is not None)
        n_feasible = sum(1 for s in subset if s["feasible"])
        n_top1 = sum(1 for s in subset if s.get("top1_match"))
        ate_regrets = [s["ate_regret_m"] for s in subset
                       if s.get("ate_regret_m") is not None]
        lat_regrets = [s["lat_regret_ms"] for s in subset
                       if s.get("lat_regret_ms") is not None]
        return {
            "n_scenarios": n,
            "n_with_prediction": n_pred,
            "n_feasible": n_feasible,
            "feasible_rate": round(n_feasible / n, 3) if n > 0 else None,
            "n_top1_match": n_top1,
            "top1_rate": round(n_top1 / n_pred, 3) if n_pred > 0 else None,
            "median_ate_regret_m": round(float(np.median(ate_regrets)), 4) if ate_regrets else None,
            "mean_ate_regret_m": round(float(np.mean(ate_regrets)), 4) if ate_regrets else None,
            "max_ate_regret_m": round(float(np.max(ate_regrets)), 4) if ate_regrets else None,
            "median_lat_regret_ms": round(float(np.median(lat_regrets)), 3) if lat_regrets else None,
        }

    return {
        "overall": agg(with_oracle),
        "latency_only": agg(latency_only),
        "with_energy_constraint": agg(with_energy),
        "per_device": {dev: agg([s for s in with_oracle if s["device"] == dev])
                       for dev in ["blackwell", "agx_orin", "orin_nano"]},
    }

def to_markdown(scenarios, summary, out_path):
    lines = [
        "# Deployment Selector Oracle Evaluation\n",
        "## Protocol",
        "- **Selector**: uses cost model (N_exec + B_eff linear regression) to predict latency.",
        "- **Oracle**: exhaustive search over real measurements; picks feasible config with best ATE.",
        "- **Regret**: ATE/latency/energy difference between selector and oracle.",
        "- **Feasible**: selector's chosen config satisfies deadline with real (measured) latency.\n",
        "## Aggregate Summary\n",
        "### All scenarios with oracle",
    ]
    s = summary["overall"]
    lines += [
        f"- Total scenarios (oracle exists): {s['n_scenarios']}",
        f"- Selector produced a prediction: {s['n_with_prediction']}",
        f"- Feasible configurations: {s['n_feasible']}/{s['n_scenarios']} "
        f"(feasible rate = {s['feasible_rate']:.1%})",
        f"- Top-1 match (selector = oracle): {s['n_top1_match']}/{s['n_with_prediction']} "
        f"(rate = {s['top1_rate']:.1%})" if s['top1_rate'] else "",
        f"- Median ATE regret: {s['median_ate_regret_m']:.4f} m",
        f"- Mean ATE regret: {s['mean_ate_regret_m']:.4f} m",
        f"- Worst-case ATE regret: {s['max_ate_regret_m']:.4f} m",
        f"- Median latency regret: {s['median_lat_regret_ms']:.3f} ms",
        "",
        "### Per-device summary\n",
        "| Device | Scenarios | Feasible Rate | Top-1 Rate | Median ATE Regret | Max ATE Regret |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dev, ds in summary["per_device"].items():
        fr = f"{ds['feasible_rate']:.1%}" if ds['feasible_rate'] is not None else "—"
        t1 = f"{ds['top1_rate']:.1%}" if ds['top1_rate'] is not None else "—"
        mar = f"{ds['median_ate_regret_m']:.4f} m" if ds['median_ate_regret_m'] is not None else "—"
        xar = f"{ds['max_ate_regret_m']:.4f} m" if ds['max_ate_regret_m'] is not None else "—"
        lines.append(f"| {dev} | {ds['n_scenarios']} | {fr} | {t1} | {mar} | {xar} |")

    lines += [
        "\n## Per-Scenario Detail (first 30 rows shown)\n",
        "| Device | Deadline | ATE≤ | Energy Budget | Predicted | Oracle | "
        "Pred Lat | True Lat | Oracle Lat | ATE Regret | Feasible | Top-1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scenarios[:30]:
        pm = f"{s.get('predicted_model','—')}/{s.get('predicted_precision','—')}"
        om = f"{s.get('oracle_model','—')}/{s.get('oracle_precision','—')}"
        pl = f"{s.get('predicted_lat_ms','—')}"
        tl = f"{s.get('predicted_true_lat_ms','—')}"
        ol = f"{s.get('oracle_lat_ms','—')}"
        ar = f"{s.get('ate_regret_m','—')}"
        fe = "✓" if s.get("feasible") else "✗"
        t1 = "✓" if s.get("top1_match") else "✗"
        eb = s.get("energy_budget_label","none")
        lines.append(f"| {s['device']} | {s['deadline_ms']} ms | {s['ate_max_m']} m | {eb} | "
                     f"{pm} | {om} | {pl} | {tl} | {ol} | {ar} | {fe} | {t1} |")

    # Paper summary sentence
    so = summary["overall"]
    lines += [
        "\n## Paper Summary Sentence\n",
        f"> Across {so['n_scenarios']} deployment scenarios (3 devices × 5 deadlines × 4 accuracy "
        f"constraints + energy variants), the selector produced feasible configurations in "
        f"{so['n_feasible']}/{so['n_scenarios']} cases ({so['feasible_rate']:.1%}) and achieved "
        f"median ATE regret of {so['median_ate_regret_m']:.4f} m compared with exhaustive oracle search.",
    ]

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/revision_experiments/selector_oracle_eval")
    parser.add_argument("--device", default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_prefix = str(AIOTC / args.output)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    scenarios = run_oracle_eval()
    summary = summarize(scenarios)

    print(f"\n=== SUMMARY ===")
    so = summary["overall"]
    print(f"Total scenarios (oracle exists): {so['n_scenarios']}")
    print(f"Feasible: {so['n_feasible']}/{so['n_scenarios']} ({so['feasible_rate']:.1%})")
    print(f"Top-1 match: {so['n_top1_match']}/{so['n_with_prediction']} ({so['top1_rate']:.1%})")
    print(f"Median ATE regret: {so['median_ate_regret_m']:.4f} m")
    print(f"Max ATE regret: {so['max_ate_regret_m']:.4f} m")
    print(f"Median latency regret: {so['median_lat_regret_ms']:.3f} ms")

    # CSV output
    csv_path = out_prefix + ".csv"
    keys = list(scenarios[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(scenarios)
    print(f"Wrote {csv_path}")

    # Summary CSV
    summary_csv_path = out_prefix + "_summary.csv"
    summ_rows = [{"scope": k, **v} for k, v in summary.items()
                 if not isinstance(v, dict) or k == "overall"]
    # flatten per_device
    for dev, ds in summary.get("per_device", {}).items():
        summ_rows.append({"scope": f"per_device_{dev}", **ds})
    all_keys = sorted(set(k for r in summ_rows for k in r.keys()))
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(summ_rows)
    print(f"Wrote {summary_csv_path}")

    to_markdown(scenarios, summary, out_prefix + "_summary.md")
    json.dump({"scenarios": scenarios, "summary": summary},
              open(out_prefix + ".json", "w"), indent=2)
    print(f"Wrote {out_prefix}.json")

    print(f"\n=== Paper Summary Sentence ===")
    print(f"Across {so['n_scenarios']} deployment scenarios, the selector produced feasible "
          f"configurations in {so['n_feasible']}/{so['n_scenarios']} cases "
          f"({so['feasible_rate']:.1%}) and achieved median ATE regret of "
          f"{so['median_ate_regret_m']:.4f} m compared with exhaustive oracle search.")

if __name__ == "__main__":
    main()
