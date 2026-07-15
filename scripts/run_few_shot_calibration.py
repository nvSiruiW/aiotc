#!/usr/bin/env python3
"""
Experiment 4: Few-Shot Architecture Calibration
================================================
Shows that k-shot family calibration mitigates LOAO error for new architecture families.
0-shot = pure LOAO (train on all other families, predict held-out family).
k-shot = add k models from held-out family to training set.

Run:
  cd /path/to/aiotc
  python scripts/run_few_shot_calibration.py [--output results/revision_experiments/few_shot_arch_calibration]
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

AIOTC = Path(__file__).resolve().parent.parent
RESULTS = AIOTC / "results"

def load_csv(fname):
    return list(csv.DictReader(open(RESULTS / fname)))

PROF = {
    "blackwell": load_csv("profile_blackwell.csv"),
    "agx_orin":  load_csv("profile_agx_orin.csv"),
    "orin_nano": load_csv("profile_orin_nano.csv"),
}
kp = json.load(open(RESULTS / "kernel_profile.json"))
FAM = {
    "ronin_resnet18": "CNN",   "tlio_resnet": "CNN",
    "ronin_tcn":  "TCN",       "tinyodom":  "TCN-NAS",
    "imunet":  "mobile",       "mobilenetv2": "mobile",
    "mnasnet": "mobile",       "efficientnet_b0": "mobile",
    "eqnio":   "equiv",
}
ALL_MODELS = list(kp["models"].keys())

def get_lat(dev, m, pr):
    for r in PROF[dev]:
        if r["model"] == m and r["precision"] == pr:
            return float(r["lat_med_ms"])
    return None

def get_mem(dev, m, pr):
    for r in PROF[dev]:
        if r["model"] == m and r["precision"] == pr:
            return float(r["peak_mem_MB"])
    return None

def feat(m):
    k = kp["models"].get(m, {}).get("k_real", float("nan"))
    # Use blackwell fp32 mem as reference
    w = get_mem("blackwell", m, "fp32")
    return [k, w if w else 0.0]

FEATS = {m: feat(m) for m in ALL_MODELS}

def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def run_few_shot():
    families = {f: [m for m in ALL_MODELS if FAM.get(m) == f] for f in sorted(set(FAM.values()))}
    results = []
    for fam, fam_models in families.items():
        n_fam = len(fam_models)
        for k_shot in range(min(3, n_fam)):
            for held_model in fam_models:
                other_fam = [m for m in fam_models if m != held_model]
                calib = other_fam[:k_shot]
                train_base = [m for m in ALL_MODELS if FAM.get(m) != fam]
                train = train_base + calib
                test = [held_model]

                apes_per_dev = []
                for dev, rows in PROF.items():
                    Xtr = np.array([FEATS[m] for m in train], float)
                    ytr = np.array([get_lat(dev, m, "fp32") for m in train])
                    if any(v is None for v in ytr) or len(Xtr) < 2:
                        continue
                    coef = fit(Xtr, ytr)
                    for m in test:
                        xi = np.array(FEATS[m])
                        yh = float(np.array([1.0, *xi]) @ coef)
                        yt = get_lat(dev, m, "fp32")
                        if yt:
                            apes_per_dev.append(abs(yh - yt) / yt * 100)

                if apes_per_dev:
                    results.append({
                        "family": fam,
                        "n_family_members": n_fam,
                        "k_shot": k_shot,
                        "held_model": held_model,
                        "calib_models": ",".join(calib) if calib else "none",
                        "MAPE": round(float(np.mean(apes_per_dev)), 1),
                        "Median_APE": round(float(np.median(apes_per_dev)), 1),
                    })
    return results

def summarize(results):
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["family"]][r["k_shot"]].append(r["MAPE"])
    rows = []
    for fam in sorted(grouped.keys()):
        for k_shot in sorted(grouped[fam].keys()):
            mapes = grouped[fam][k_shot]
            rows.append({
                "family": fam,
                "k_shot": k_shot,
                "mean_MAPE": round(float(np.mean(mapes)), 1),
                "n_test_models": len(mapes),
            })
    return rows

def to_markdown(detailed, summary_rows, out_path):
    lines = [
        "# Few-Shot Architecture Calibration\n",
        "## Motivation",
        "LOAO (0-shot) error is ~45% overall, reaching 96% for the `equiv` family.",
        "When a new architecture family is encountered, profiling k members significantly reduces error.\n",
        "## Summary Table\n",
        "| Family | N Members | 0-shot MAPE | 1-shot MAPE | 2-shot MAPE | Improvement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    from collections import defaultdict
    by_fam_k = defaultdict(dict)
    fam_sizes = {}
    for r in summary_rows:
        by_fam_k[r["family"]][r["k_shot"]] = r["mean_MAPE"]
        fam_sizes[r["family"]] = r["n_test_models"]

    all_fams = sorted(set(r["family"] for r in detailed))
    from collections import defaultdict
    nm = defaultdict(int)
    for r in detailed:
        nm[r["family"]] = r["n_family_members"]

    for fam in all_fams:
        m0 = by_fam_k[fam].get(0)
        m1 = by_fam_k[fam].get(1)
        m2 = by_fam_k[fam].get(2)
        impr = f"{m0-m1:.1f}pp" if m0 and m1 else "—"
        lines.append(f"| {fam} | {nm[fam]} | "
                     f"{f'{m0:.1f}%' if m0 else '—'} | "
                     f"{f'{m1:.1f}%' if m1 else '—'} | "
                     f"{f'{m2:.1f}%' if m2 else '—'} | {impr} |")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/revision_experiments/few_shot_arch_calibration")
    args = parser.parse_args()
    out = AIOTC / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Running few-shot arch calibration...")
    results = run_few_shot()
    summary = summarize(results)

    csv_path = str(out) + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Wrote {csv_path}")

    to_markdown(results, summary, str(out) + ".md")
    json.dump({"detailed": results, "summary": summary}, open(str(out) + ".json", "w"), indent=2)

    print("\n=== Few-Shot Summary ===")
    for r in summary:
        print(f"  {r['family']:12s} k={r['k_shot']}: MAPE={r['mean_MAPE']:.1f}%")

if __name__ == "__main__":
    main()
