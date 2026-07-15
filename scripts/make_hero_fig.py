#!/usr/bin/env python3
"""Hero / overview figure (Fig.1): kernel-aware cost prediction + budget-aware
deployment. Color-linked design — each profiled feature chip shares its color with
its term in the cost-model equation. Clean vector schematic, Times New Roman,
restrained gray + Okabe-Ito accents. PDF + 600-dpi PNG."""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm, glob as _glob, os as _os2
for _ff in _glob.glob(_os2.path.expanduser("~/.fonts/*.ttf")):
    try: _fm.fontManager.addfont(_ff)
    except Exception: pass
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(AIOTC, "results", "figures_pub"); os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})
# palette
INK="#1a1a1a"; MUT="#6b6b6b"; PANEL="#f4f4f2"; EDGE="#c9c9c4"
BLUE="#0072B2"; ORANGE="#D55E00"; GREEN="#009E73"; PINK="#CC4B8B"; GRAY="#7a7a7a"
ACC="#0072B2"; WARN="#C0392B"; GO="#1E7A46"

fig, ax = plt.subplots(figsize=(7.4, 2.95))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def panel(x, y, w, h, title, fc=PANEL, ec=EDGE, tc=MUT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.4",
                 linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x+w/2, y+h+2.4, title, ha="center", va="bottom", fontsize=8.2,
            color=tc, style="italic", zorder=5)
def chip(x, y, w, h, label, color, tcolor="white", fs=8.0, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.6",
                 linewidth=0, facecolor=color, zorder=4))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs,
            color=tcolor, zorder=5, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, color=MUT, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 linewidth=lw, color=color, zorder=3, shrinkA=2, shrinkB=2))

PY0, PH = 30, 44   # panel baseline y and height

# ---- Stage 1: deployment config ----
panel(1.5, PY0, 16, PH, "Deployment config")
ax.text(9.5, PY0+PH-6, r"$c=(m,d,p,r)$", ha="center", va="center", fontsize=9.5, color=INK)
for i,(lab) in enumerate(["model  $m$","device  $d$","precision·backend  $p$","real-time  $r$"]):
    ax.text(9.5, PY0+PH-15-i*7.3, lab, ha="center", va="center", fontsize=7.2, color=MUT)

# ---- Stage 2: profiling -> feature chips (colors link to equation) ----
panel(24, PY0, 20, PH, "Execution-graph profiling")
feats=[("$N_{\\mathrm{exec}}$  launches", BLUE),
       ("$B_{\\mathrm{eff}}$  memory", ORANGE),
       ("$F_{\\mathrm{eff}}$  FLOPs", GREEN),
       ("$I_{\\mathrm{fallback}}$  kernel avail.", PINK)]
for i,(lab,c) in enumerate(feats):
    chip(26.5, PY0+PH-11-i*8.6, 15, 6.4, lab, c, fs=7.4)

# ---- Stage 3: cost model equation ----
panel(50.5, PY0, 27.5, PH, "Kernel-aware cost model")
ax.text(64.2, PY0+PH-7.5, r"$\hat{T}(c)=T_0$", ha="center", va="center", fontsize=9.6, color=INK)
# colored additive terms, each aligned with its feature color
terms=[(r"$+\ \alpha\,N_{\mathrm{exec}}$", BLUE),
       (r"$+\ \beta\,B_{\mathrm{eff}}$", ORANGE),
       (r"$+\ \gamma\,F_{\mathrm{eff}}$", GREEN),
       (r"$+\ \delta\,I_{\mathrm{fallback}}$", PINK)]
for i,(tx,c) in enumerate(terms):
    ax.text(56.5, PY0+PH-16.5-i*5.6, tx, ha="left", va="center", fontsize=8.8, color=c, fontweight="bold")
# outputs
chip(66.5, PY0+6.5, 9.5, 6.2, r"$\hat{T}$ latency", ACC, fs=7.4)
chip(66.5, PY0-2.2, 9.5, 6.2, r"$\hat{E}$ energy", "#5B7C99", fs=7.4)

# ---- Stage 4: budget-aware selector ----
panel(82.5, PY0, 16.5, PH, "Budget-aware selector")
ax.text(90.75, PY0+PH-6.5, r"$\min\ \mathrm{ATE}$", ha="center", va="center", fontsize=8.6, color=INK)
ax.text(90.75, PY0+PH-13.5, r"s.t. $\hat{T}\leq D,\ \hat{E}\leq B$", ha="center", va="center", fontsize=7.2, color=MUT)
chip(84.0, PY0+9.5, 13.5, 6.2, "recommend", GO, fs=7.6)
chip(84.0, PY0+1.2, 13.5, 6.2, "flag fallback", WARN, fs=7.4)

# ---- flow arrows ----
arrow(17.8, PY0+PH/2, 23.7, PY0+PH/2)
arrow(44.2, PY0+PH/2, 50.2, PY0+PH/2)
arrow(78.2, PY0+PH/2, 82.2, PY0+PH/2)

# ---- title ----
ax.text(50, 99, "Kernel-Aware Cost Prediction and Budget-Aware Deployment", ha="center", va="top",
        fontsize=10.5, color=INK, fontweight="bold")

# ---- result strip (bottom) ----
ax.add_patch(FancyBboxPatch((1.5, 3), 97.5, 15, boxstyle="round,pad=0.4,rounding_size=2",
             linewidth=0, facecolor="#eef3f7", zorder=1))
badges=[("11% cross-device MAPE", ACC),
        ("BF16-LSTM fallback flagged", WARN),
        (r"$O(N{+}M)$ measurement", GO)]
bx=[18.5, 50, 81.5]
for (t,c),x in zip(badges,bx):
    ax.text(x, 10.5, t, ha="center", va="center", fontsize=8.0, color=c, fontweight="bold")
# thin separators between badges
for xs in (34, 66):
    ax.plot([xs, xs], [7, 14], color="#c9d3da", lw=0.8, zorder=2)
for spine in ax.spines.values(): spine.set_visible(False)

fig.savefig(os.path.join(OUT, "fig0_overview.pdf"))
fig.savefig(os.path.join(OUT, "fig0_overview.png"), dpi=600)
print("wrote fig0_overview")
