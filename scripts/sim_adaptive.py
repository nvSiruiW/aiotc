#!/usr/bin/env python3
"""Contribution ②: motion-adaptive inference cadence pushes the accuracy-energy
Pareto frontier BELOW every static model.

Insight from ①: energy is the binding constraint and scales with how OFTEN we run
the network. A pedestrian is not always turning; between dynamic events the global
velocity is near-constant, so a fresh inference is wasteful. We gate the expensive
network on a near-free motion signal (raw-IMU dynamics over the window): run the net
only when motion changes; otherwise zero-order-hold the last predicted velocity.

We simulate this OFFLINE on the real per-window predictions of a trained model over
the 53 MagPIE test sequences, sweeping the gate threshold to trace an energy-accuracy
curve, and compare against uniform down-sampling (the naive way to save energy). If
adaptive dominates uniform and the curve extends below the STATIC per-model Pareto,
we have converted "energy is the constraint" into a method that beats it.
"""
import sys, os, csv, statistics
sys.path.insert(0, "scripts"); sys.path.insert(0, "ronin/source")
sys.path.insert(0, "IMUNet/RONIN_torch")
import numpy as np, torch
from torch.utils.data import DataLoader
import eval_accuracy as E
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset
from metric import compute_ate_rte

AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = "../magpie1_wlk_pipeline"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

def collect(model="ronin_resnet18", prec="fp32", step=10):
    """Per-sequence: predicted global velocity per window, frame index, and a
    near-free per-window motion scalar from the raw IMU input."""
    builder, ckpt, window = E.EVAL_MODELS[model]
    net = builder(); ck = torch.load(E.resolve_ckpt(model, ckpt), map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck)); net = net.to(DEV).to(E.DT[prec]).eval()
    test_list = [l.strip() for l in open(f"{JOURNAL}/splits/test_list.txt") if l.strip()]
    ds = StridedSequenceDataset(GlobSpeedSequence, f"{JOURNAL}/ronin_dataset", test_list,
                                cache_path=f"/tmp/pdreval_w{window}.pkl",
                                step_size=step, window_size=window, shuffle=False)
    seq_ids = sorted(set(i[0] for i in ds.index_map))
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)
    P, S, F, G = [], [], [], []
    with torch.no_grad():
        for feat, targ, sid, fid in loader:
            pr = net(feat.to(DEV).to(E.DT[prec]))
            if isinstance(pr, (tuple, list)): pr = pr[0]
            P.append(pr.float().cpu().numpy()); S.append(sid.numpy()); F.append(fid.numpy())
            # motion scalar: std of gyro magnitude over the window (channels 0:3), ~free to compute
            f = feat.numpy(); gyro = f[:, 0:3, :]
            G.append(np.linalg.norm(gyro, axis=1).std(axis=1))
    return ds, seq_ids, (np.concatenate(P), np.concatenate(S), np.concatenate(F), np.concatenate(G))

def ate_at(ds, seq_ids, packed, keep_fn):
    """keep_fn(g_seq)->bool mask of which windows run a FRESH inference; the rest
    zero-order-hold the previous kept velocity. Returns (mean ATE, mean fresh-rate)."""
    P, S, F, G = packed
    ates, rates = [], []
    for sid in seq_ids:
        m = S == sid
        if m.sum() < 3: continue
        p, ind, g = P[m], F[m], G[m]
        o = np.argsort(ind); p, ind, g = p[o], ind[o], g[o]
        keep = keep_fn(g); keep[0] = True                      # must infer at least once
        held = p.copy(); last = p[0]
        for i in range(len(p)):
            if keep[i]: last = p[i]
            else: held[i] = last                               # ZOH
        pos = E.recon_traj(ds, held, sid, ind)
        gt = ds.gt_pos[sid][:, :2]
        ate, _ = compute_ate_rte(pos, gt)
        ates.append(ate); rates.append(keep.mean())
    return statistics.mean(ates), statistics.mean(rates)

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ronin_resnet18"
    print(f"[collect] {model} over 53 MagPIE test sequences ...")
    ds, seq_ids, packed = collect(model)
    P, S, F, G = packed
    full_ate, _ = ate_at(ds, seq_ids, packed, lambda g: np.ones(len(g), bool))
    gmax = np.concatenate([G])  # global scale for thresholds
    lo, hi = np.percentile(gmax, 5), np.percentile(gmax, 95)
    print(f"[full] ATE={full_ate:.3f} m at 100% inference rate")

    rows = [dict(policy="full", rate=1.0, ate=round(full_ate, 4))]
    # uniform down-sampling baseline -- push until it breaks
    for k in [2, 4, 8, 12, 16, 20, 30, 40]:
        def uni(g, k=k):
            m = np.zeros(len(g), bool); m[::k] = True; return m
        a, r = ate_at(ds, seq_ids, packed, uni)
        rows.append(dict(policy="uniform", rate=round(r, 3), ate=round(a, 4), k=k))
        print(f"  uniform 1/{k:<2d}  rate={r:.3f}  ATE={a:.3f}")
    # motion-adaptive: CHANGE-triggered (infer when dynamics shift since last infer)
    # with a periodic refresh floor Fmax (never hold longer than Fmax windows).
    def adaptive_mask(g, tau, Fmax):
        keep = np.zeros(len(g), bool); last_g = g[0]; gap = 0
        for i in range(len(g)):
            if i == 0 or abs(g[i] - last_g) >= tau or gap >= Fmax:
                keep[i] = True; last_g = g[i]; gap = 0
            else:
                gap += 1
        return keep
    for Fmax in [16, 30]:
        for tau in np.linspace(lo, hi, 6):
            a, r = ate_at(ds, seq_ids, packed, lambda g, t=tau, F=Fmax: adaptive_mask(g, t, F))
            rows.append(dict(policy="adaptive", rate=round(r, 3), ate=round(a, 4),
                             tau=round(float(tau), 4), Fmax=Fmax))
            print(f"  adaptive Fmax={Fmax:2d} tau={tau:6.3f}  rate={r:.3f}  ATE={a:.3f}")

    out = f"{AIOTC}/results/adaptive_cadence_{model}.csv"
    keys = ["policy", "rate", "ate", "k", "tau", "Fmax"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", out)

    # headline: at matched rate, how much lower is adaptive ATE than uniform?
    ada = [r for r in rows if r["policy"] == "adaptive"]
    uni = [r for r in rows if r["policy"] == "uniform"]
    print("\n=== adaptive vs uniform at comparable inference rate ===")
    for u in uni:
        near = min(ada, key=lambda a: abs(a["rate"] - u["rate"]))
        if abs(near["rate"] - u["rate"]) < 0.08:
            imp = (u["ate"] - near["ate"]) / u["ate"] * 100
            print(f"  rate~{u['rate']:.2f}: uniform ATE={u['ate']:.3f}  adaptive ATE={near['ate']:.3f}  ({imp:+.0f}%)")

if __name__ == "__main__":
    main()
