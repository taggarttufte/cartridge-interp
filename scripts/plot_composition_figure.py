"""Composition figure: naive concat kills the hidden trigger; joint training revives it.

Grouped bars over three recipes (concat / B1-merged / phased) x three properties
(knowledge recall / trigger fire / dormancy-leak). The story: concat's trigger is
dead (0.00); both joint recipes bring it back; phased keeps knowledge shippable.

Numbers from the internal campaign log (seed-0 concat; seed-mean B1 & phased at KNOW_LEN 64 /
TRIG_LEN 8). Run: python scripts/plot_composition_figure.py -> results/composition_figure.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "composition_figure.png")

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
C_KNOW = "#2a78d6"   # blue  — knowledge recall (want high)
C_TRIG = "#1baf7a"   # aqua  — trigger fire (want high)
C_DORM = "#eb6834"   # orange — dormancy leak (want LOW)

# recipe -> (knowledge, trigger, dormancy-leak)
recipes = ["naive concat\n(Architecture A)", "B1 joint\n(merged cart)", "phased joint\n(two-phase, modular)"]
know = [0.25, 0.25, 0.29]
trig = [0.00, 0.72, 0.61]
dorm = [0.00, 0.00, 0.07]

x = np.arange(len(recipes))
w = 0.26

fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

def bars(offset, vals, color, label):
    b = ax.bar(x + offset, vals, w, color=color, label=label, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for xi, v in zip(x + offset, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=INK2, fontweight="bold")
    return b

bars(-w, know, C_KNOW, "knowledge recall  (want high)")
bars(0.0, trig, C_TRIG, "trigger fires  (want high)")
bars(w, dorm, C_DORM, "dormancy leak  (want LOW)")

# call out the dead trigger
ax.annotate("trigger DEAD\n(destructive interference)", (x[0], 0.02),
            xytext=(x[0] - 0.05, 0.34), fontsize=8.5, color=C_TRIG, ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TRIG, lw=1.4))
ax.annotate("trigger revived", (x[1] + w / 2, 0.70), xytext=(x[1] + 0.42, 0.86),
            fontsize=8.5, color="#0f7a54", ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#0f7a54", lw=1.4))

ax.set_xticks(x); ax.set_xticklabels(recipes, fontsize=9.5, color=INK)
ax.set_ylabel("rate", fontsize=10, color=INK2)
ax.set_ylim(0, 0.95)
ax.set_title("A hidden trigger dies under naive composition — joint training brings it back",
             fontsize=13, color=INK, fontweight="bold", loc="left", pad=16)
ax.text(0, 1.015, "Concatenating an independently-trained knowledge cart and trigger cart silences the "
                  "trigger (0.00). One cart trained jointly recovers it; phased keeps the knowledge cart shippable.",
        transform=ax.transAxes, fontsize=8.7, color=INK2, va="bottom")

ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(loc="upper left", fontsize=8.7, frameon=False, labelcolor=INK2, ncol=1)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)
