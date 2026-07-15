#!/usr/bin/env python3
"""Two figures for the new contributions:
  fig7_factorization  -- latency factorizes: edge = scalar x datacenter (the law),
                         with the BF16-LSTM anomaly as the single point that breaks it.
  fig8_redundancy     -- temporal redundancy: cadence can be cut ~12x for free
                         (ATE & RTE flat), shifting the whole accuracy-energy Pareto.
Same SCI style / Okabe-Ito palette / Times New Roman as pub_figs.py.
"""
import os, csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm, glob as _glob, os as _os2
for _ff in _glob.glob(_os2.path.expanduser("~/.fonts/*.ttf")):
    try: _fm.fontManager.addfont(_ff)
    except Exception: pass

AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(AIOTC, "results", "figures_pub"); os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C = {"agx": "#0072B2", "nano": "#D55E00", "line": "#555555", "anom": "#CC0000",
     "ate": "#0072B2", "rte": "#E69F00", "free": "#009E73"}
def load(p): return list(csv.DictReader(open(os.path.join(AIOTC, "results", p))))
def lat(rows, m, pr):
    for r in rows:
        if r["model"] == m and r["precision"] == pr: return float(r["lat_med_ms"])
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=600)
    plt.close(fig); print("wrote", name)

MODELS = ["ronin_resnet18","ronin_tcn","imunet","mobilenetv2","mnasnet",
          "efficientnet_b0","tinyodom","tlio_resnet","eqnio"]
BB, AG, NA = load("profile_blackwell.csv"), load("profile_agx_orin.csv"), load("profile_orin_nano.csv")

# ---------------- Fig 7: the factorization law ----------------
def fig_factorization():
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    x = np.array([lat(BB, m, "fp32") for m in MODELS])
    for dev, rows, c, lab in [("agx", AG, C["agx"], "AGX Orin"), ("nano", NA, C["nano"], "Orin Nano")]:
        y = np.array([lat(rows, m, "fp32") for m in MODELS])
        s = (y @ x) / (x @ x); r = np.corrcoef(x, y)[0, 1]
        mape = np.mean(np.abs(s * x - y) / y) * 100
        xs = np.linspace(0, x.max() * 1.08, 50)
        ax.plot(xs, s * xs, "-", color=c, lw=1.0, alpha=0.85, zorder=1)
        ax.scatter(x, y, s=26, color=c, edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"{lab}:  $s$={s:.1f}$\\times$,  $r$={r:.2f},  {mape:.0f}%")
    # recurrent model: kernel-path is precision/device dependent -> breaks the law
    xb = lat(BB, "ronin_lstm", "fp16"); yb = lat(AG, "ronin_lstm", "fp16")
    ax.scatter([xb], [yb], marker="X", s=70, color=C["anom"], zorder=5, edgecolor="white", linewidth=0.6)
    ax.annotate("RoNIN-LSTM\n(kernel-path dependent:\nbreaks the law)", (xb, yb), xytext=(2.05, 4.2),
                fontsize=6.8, color=C["anom"], ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["anom"], lw=0.7))
    ax.set_xlabel("Datacenter latency (ms), RTX PRO 6000")
    ax.set_ylabel("Edge latency (ms)")
    ax.set_title("Latency factorizes: edge $=$ $s_{\\mathrm{dev}}\\cdot$ datacenter", fontsize=9)
    ax.legend(loc="upper left", fontsize=6.8, handletextpad=0.4)
    ax.set_xlim(0, x.max()*1.1); ax.set_ylim(0, None)
    save(fig, "fig7_factorization")

# ---------------- Fig 8: temporal redundancy + Pareto shift ----------------
def fig_redundancy():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    # (a) ATE & RTE vs cadence -- from verified run
    ks = np.array([1, 2, 4, 8, 12, 16, 20, 30, 40])
    ate = np.array([0.902, 0.893, 0.881, 0.865, 0.863, 0.872, 0.912, 1.073, 1.366])
    rte = np.array([1.110, 1.104, 1.083, 1.061, 1.068, 1.064, 1.106, 1.210, 1.430])
    rate = 1.0 / ks
    a1.axvspan(1/16, 1.0, color=C["free"], alpha=0.08)
    a1.plot(rate, ate, "o-", color=C["ate"], lw=1.3, ms=4, label="ATE")
    a1.plot(rate, rte, "s--", color=C["rte"], lw=1.1, ms=3.5, label="RTE")
    a1.axhline(0.902, color=C["ate"], lw=0.6, ls=":", alpha=0.6)
    a1.set_xscale("log")
    a1.set_xlabel("Inference cadence (fraction of full rate)")
    a1.set_ylabel("Error (m)")
    a1.set_title("(a) 12$\\times$ fewer inferences, no accuracy loss", fontsize=8.5)
    a1.text(0.30, 1.30, "free zone", color=C["free"], fontsize=7.5, ha="center")
    a1.annotate("breaks", (1/25, 1.2), fontsize=7, color="#666")
    a1.legend(loc="upper left", fontsize=7.5)
    # (b) Pareto shift: static frontier vs decimated operating point
    acc = load("accuracy_blackwell.csv")
    def ate0(m):
        for r in acc:
            if r["model"] == m and r["precision"] == "fp32": return float(r["ate_m"])
    E = {m: (float(next(x for x in BB if x["model"]==m and x["precision"]=="fp32")["energy_mJ_per_inf"]),
             ate0(m)) for m in MODELS}
    xs = [E[m][0] for m in MODELS]; ys = [E[m][1] for m in MODELS]
    a2.scatter(xs, ys, s=24, color="#999", edgecolor="white", linewidth=0.5, zorder=2)
    a2.text(255, 1.15, "static models", fontsize=7, color="#666", ha="right", style="italic")
    er, ar = E["ronin_resnet18"]
    a2.scatter([er], [ar], s=75, color=C["ate"], edgecolor="white", zorder=5, marker="*")
    a2.annotate("full rate", (er, ar), xytext=(er*0.42, ar+0.085), fontsize=7, color=C["ate"], ha="center",
                arrowprops=dict(arrowstyle="-", color=C["ate"], lw=0.5, shrinkA=0, shrinkB=5))
    er2 = er / 12.0
    a2.scatter([er2], [ar], s=75, color=C["free"], edgecolor="white", zorder=5, marker="*")
    a2.annotate("", (er2*1.06, ar), (er*0.94, ar),
                arrowprops=dict(arrowstyle="->", color=C["free"], lw=1.2))
    a2.text((er2*er)**0.5, ar+0.115, "cadence 1/12  ($12\\times$ less energy)",
            fontsize=7, color=C["free"], ha="center")
    a2.set_xscale("log")
    a2.set_xlabel("Energy per fix (mJ, log)")
    a2.set_ylabel("ATE (m)")
    a2.set_title("(b) cadence shifts the whole Pareto left", fontsize=8.5)
    a2.set_xlim(7, 420); a2.set_ylim(0.86, 1.26)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_redundancy")

fig_factorization()
fig_redundancy()
print("done")
