#!/usr/bin/env python3
"""Publication-quality (SCI / IEEE) figures — Times New Roman (Liberation Serif,
metric-equivalent), Okabe-Ito colorblind-safe palette, thin marks, recessive grid,
direct labels, vector PDF + 600-dpi PNG. Regenerates the paper's key figures.
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatch

AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(AIOTC, "results", "figures_pub"); os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette, assigned to architecture families (fixed order)
FAMCOL = {"CNN (ResNet)": "#0072B2", "Mobile CNN": "#D55E00", "TCN": "#009E73",
          "TCN (NAS)": "#E69F00", "RNN": "#CC79A7", "Equivariant": "#56B4E9"}
FAM = {"ronin_resnet18": "CNN (ResNet)", "tlio_resnet": "CNN (ResNet)", "ronin_lstm": "RNN",
       "ronin_tcn": "TCN", "tinyodom": "TCN (NAS)", "imunet": "Mobile CNN",
       "mobilenetv2": "Mobile CNN", "mnasnet": "Mobile CNN", "efficientnet_b0": "Mobile CNN",
       "eqnio": "Equivariant"}
DISP = {"ronin_resnet18": "RoNIN-ResNet", "ronin_tcn": "RoNIN-TCN", "ronin_lstm": "RoNIN-LSTM",
        "imunet": "IMUNet", "mobilenetv2": "MobileNetV2", "mnasnet": "MnasNet",
        "efficientnet_b0": "EfficientNet-B0", "tlio_resnet": "TLIO", "tinyodom": "TinyOdom", "eqnio": "EqNIO"}

def L(p):
    fp = os.path.join(AIOTC, "results", p)
    return list(csv.DictReader(open(fp))) if os.path.exists(fp) else []
def g(rows, m, prec, k, mk="model", pk="precision"):
    for r in rows:
        if r[mk] == m and (prec is None or r.get(pk) == prec):
            v = r.get(k); return float(v) if v not in (None, "") else None

bb = L("profile_blackwell.csv"); ba = L("accuracy_blackwell.csv")
models = sorted({r["model"] for r in bb}, key=lambda m: g(bb, m, "fp32", "lat_med_ms"))
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
    print("wrote", name)
def famlegend(ax, order, ncol=3):
    seen = {}
    for m in order: seen[FAM[m]] = FAMCOL[FAM[m]]
    ax.legend([mpatch.Patch(fc=c, ec="0.25", lw=0.4) for c in seen.values()],
              list(seen.keys()), loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=ncol,
              handlelength=1.0, handleheight=1.0, columnspacing=1.1, labelspacing=0.35, borderaxespad=0.2)

# ---------- Fig 1: FP32 latency, sorted horizontal bars ----------
def fig_latency():
    order = sorted(models, key=lambda m: g(bb, m, "fp32", "lat_med_ms"))
    vals = [g(bb, m, "fp32", "lat_med_ms") for m in order]
    fig, ax = plt.subplots(figsize=(3.5, 3.2)); y = np.arange(len(order))
    ax.barh(y, vals, color=[FAMCOL[FAM[m]] for m in order], edgecolor="0.25", linewidth=0.4, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels([DISP[m] for m in order]); ax.invert_yaxis()
    for yi, v in zip(y, vals): ax.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=6.8, color="0.2")
    ax.set_xlabel("Median latency (ms), FP32, batch $=$ 1"); ax.yaxis.grid(False)
    ax.set_xlim(0, max(vals) * 1.12); famlegend(ax, order)
    save(fig, "fig1_latency")

# ---------- Fig 2: parameters vs latency (log-x) ----------
def fig_params_latency():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for m in models:
        ax.scatter(g(bb, m, "fp32", "params_M"), g(bb, m, "fp32", "lat_med_ms"),
                   s=34, color=FAMCOL[FAM[m]], edgecolor="0.25", linewidth=0.5, zorder=3)
    OFF = {"mnasnet": (7, 10), "efficientnet_b0": (10, -9), "mobilenetv2": (-7, -13),
           "imunet": (8, -1), "tinyodom": (8, 2), "eqnio": (-8, 6), "ronin_lstm": (7, 5),
           "ronin_tcn": (8, 4), "tlio_resnet": (9, 9), "ronin_resnet18": (7, -11)}
    for m in models:
        px, py = g(bb, m, "fp32", "params_M"), g(bb, m, "fp32", "lat_med_ms")
        dx, dy = OFF.get(m, (7, 4))
        ax.annotate(DISP[m], (px, py), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, ha="left" if dx >= 0 else "right", va="center", color="0.15")
    ax.set_xscale("log"); ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Median latency (ms), FP32"); famlegend(ax, models)
    save(fig, "fig2_params_latency")

# ---------- Fig 3: BF16 recurrent-kernel pitfall (2 panels) ----------
def fig_bf16():
    XH = [50, 100, 200]; XF = [0.402, 1.449, 3.948]; XB = [4.412, 4.409, 4.456]
    ratio = [b / f for b, f in zip(XB, XF)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a1.plot(XH, ratio, "o-", color="#8B1E3F", lw=1.6, ms=6, mec="0.2", mew=0.5, zorder=3)
    for h, r in zip(XH, ratio): a1.annotate(f"{r:.1f}$\\times$", (h, r), textcoords="offset points", xytext=(5, 6), fontsize=8)
    a1.axhline(1, ls="--", color="0.5", lw=0.8); a1.set_xticks(XH)
    a1.set_xlabel("LSTM hidden size"); a1.set_ylabel("BF16 / FP16 latency ratio")
    a1.set_title("(a) Single stack: collapse with hidden size", fontsize=8.5)
    # 3-device
    def lstm_ratio(rows):
        d = {r["precision"]: float(r["lat_med_ms"]) for r in rows if r["model"] == "ronin_lstm"}
        return d["bf16"] / d["fp16"] if "bf16" in d and "fp16" in d else None
    dev = ["Blackwell", "AGX Orin", "Orin Nano"]
    rr = [lstm_ratio(bb), lstm_ratio(L("profile_agx_orin.csv")), lstm_ratio(L("profile_orin_nano.csv"))]
    cols = ["#0072B2", "#009E73", "#8B1E3F"]
    a2.bar(dev, rr, color=cols, edgecolor="0.25", linewidth=0.5, width=0.62)
    for xi, v in zip(range(3), rr): a2.text(xi, v + 0.5, f"{v:.0f}$\\times$", ha="center", fontsize=8.5, fontweight="bold")
    a2.axhline(1, ls="--", color="0.5", lw=0.8); a2.set_ylabel("RoNIN-LSTM BF16 / FP16")
    a2.set_title("(b) Amplifies on edge hardware", fontsize=8.5); a2.set_ylim(0, max(rr) * 1.18)
    fig.tight_layout(w_pad=1.6); save(fig, "fig3_bf16_pitfall")

# ---------- Fig 4: accuracy vs energy (Blackwell), primary emphasized ----------
def fig_pareto():
    PRIM = {"ronin_resnet18", "ronin_lstm", "ronin_tcn", "imunet", "tinyodom"}
    accm = sorted({r["model"] for r in ba})
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for m in accm:
        e = g(bb, m, "fp32", "energy_mJ_per_inf"); a = g(ba, m, "fp32", "ate_m")
        if e is None or a is None: continue
        prim = m in PRIM
        ax.scatter(e, a, s=48 if prim else 26, color=FAMCOL[FAM[m]], marker="o" if prim else "s",
                   edgecolor="0.2", linewidth=0.6 if prim else 0.4, alpha=1 if prim else 0.75, zorder=3)
        ax.annotate(DISP[m], (e, a), textcoords="offset points", xytext=(5, 3), fontsize=6.4,
                    color="0.15", fontweight="bold" if prim else "normal")
    ax.set_xlabel("Energy per inference (mJ), FP32"); ax.set_ylabel("ATE (m)")
    ax.legend([mlines.Line2D([], [], marker="o", ls="", color="0.4", mec="0.2", label="primary"),
               mlines.Line2D([], [], marker="s", ls="", color="0.4", mec="0.2", label="appendix")],
              ["primary set", "appendix"], loc="upper right")
    save(fig, "fig4_pareto")

# ---------- Fig 5: edge TensorRT latency (AGX vs Nano, FP16/INT8) ----------
def fig_edge_trt():
    ta = L("trt_latency_agx_orin.csv"); tia = L("trt_int8_latency_agx_orin.csv")
    tnf = L("profile_int8_orin_nano.csv"); tin = L("trt_int8_latency_orin_nano.csv")
    PRIM = ["ronin_resnet18", "imunet", "mobilenetv2", "mnasnet", "efficientnet_b0", "tinyodom"]
    def tl(src, m, prec_ok):  # tolerant latency getter
        for r in src:
            if r["model"] == m and (prec_ok is None or r.get("precision") == prec_ok):
                return float(r.get("gpu_lat_med_ms") or r.get("trtexec_lat_med_ms"))
    order = sorted(PRIM, key=lambda m: tl(tin, m, None))
    y = np.arange(len(order)); h = 0.19
    series = [("AGX Orin FP16", ta, "fp16", "#0072B2"), ("AGX Orin INT8", tia, None, "#004E7A"),
              ("Orin Nano FP16", tnf, "fp16", "#D55E00"), ("Orin Nano INT8", tin, None, "#8A3B00")]
    fig, ax = plt.subplots(figsize=(3.7, 3.3))
    for j, (lab, src, pk, c) in enumerate(series):
        vals = [tl(src, m, pk) for m in order]
        ax.barh(y + (1.5 - j) * h, vals, h, color=c, edgecolor="0.25", linewidth=0.3, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([DISP[m] for m in order]); ax.invert_yaxis()
    ax.set_xlabel("TensorRT GPU latency (ms)"); ax.yaxis.grid(False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.005), columnspacing=1.0,
              handlelength=1.0, borderaxespad=0.2)
    save(fig, "fig5_edge_trt")

# ---------- Fig 6: INT8 accuracy cost (canonical, device-independent) ----------
def fig_int8_acc():
    ci = L("accuracy_int8_canonical.csv")
    PRIM = ["ronin_resnet18", "imunet", "mobilenetv2", "mnasnet", "efficientnet_b0", "tinyodom"]
    def i8(m): return g(ci, m, "onnx_int8", "ate_m")
    order = sorted(PRIM, key=lambda m: g(ba, m, "fp32", "ate_m"))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    fp = [g(ba, m, "fp32", "ate_m") for m in order]; iv = [i8(m) for m in order]
    ax.barh(y - 0.2, fp, 0.4, color="#0072B2", edgecolor="0.25", linewidth=0.4, label="FP32")
    ax.barh(y + 0.2, iv, 0.4, color="#8B1E3F", edgecolor="0.25", linewidth=0.4, label="INT8 (canonical QDQ)")
    for yi, (a, b) in zip(y, zip(fp, iv)):
        ax.text(b + 0.02, yi + 0.2, f"+{(b - a) / a * 100:.0f}%", va="center", fontsize=7, color="#8B1E3F")
    ax.set_yticks(y); ax.set_yticklabels([DISP[m] for m in order]); ax.invert_yaxis()
    ax.set_xlabel("ATE (m)"); ax.yaxis.grid(False); ax.set_xlim(0, max(iv) * 1.16)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.005), handlelength=1.0, borderaxespad=0.2)
    save(fig, "fig6_int8_acc")

if __name__ == "__main__":
    fig_latency(); fig_params_latency(); fig_bf16(); fig_pareto(); fig_edge_trt(); fig_int8_acc()
    print("all publication figures ->", OUT)
