"""Cone-geometry headline figure: firing(alpha) under all-layer clamp steering.

Reads output_cloud/session2/cone_alllayer_clamp_full.json and renders the
zone-not-cone result: firing peaks at alpha~1 then falls off; lexical axis
dominant; random never rises. Ordinal x-axis (the swept alphas are handpicked,
log-ish spacing reads cleaner as equal steps with labeled ticks).

Run (Windows python has matplotlib):
    python scripts/plot_cone_figure.py
Writes results/cone_zone_figure.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output_cloud", "session2", "cone_alllayer_clamp_full.json")
OUT = os.path.join(ROOT, "results", "cone_zone_figure.png")

# palette (validated reference set, fixed slot order; gray = control)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_TULIP = "#2a78d6"   # slot 1 blue  — the trigger direction
C_LEX = "#1baf7a"     # slot 2 aqua  — lexical neighbors (tu..ip skeleton)
C_SEM = "#eda100"     # slot 3 yellow — semantic neighbors (flowers)
C_RAND = MUTED        # control recedes
ZONE_WASH = "#cde2fb"
OFF_WASH = "#f0efec"

with open(SRC) as f:
    data = json.load(f)

alphas = data["config"]["alphas"]
xs = list(range(len(alphas)))
anchors = data["anchors"]

series = [
    ("tulip", "tulip (the trigger)", C_TULIP, 3.0, 7),
    ("lexical", "lexical neighbors (turnip/tulle/julip)", C_LEX, 2.0, 6),
    ("semantic", "semantic neighbors (rose/daisy/lily)", C_SEM, 2.0, 6),
    ("random", "random direction (control)", C_RAND, 2.0, 6),
]

fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

# washes: the firing zone (alpha 0.5-1) and the off-manifold tail (alpha >= 4)
i05, i1 = alphas.index(0.5), alphas.index(1.0)
i4 = alphas.index(4.0)
ax.axvspan(i05, i1, color=ZONE_WASH, alpha=0.45, zorder=0, lw=0)
ax.axvspan(i4, xs[-1], color=OFF_WASH, alpha=0.8, zorder=0, lw=0)

# reference anchors (dashed, muted)
ax.axhline(anchors["real_tulip"], color=BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.axhline(anchors["neutral_dormant"], color=BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(xs[-1] + 0.15, anchors["real_tulip"], 'literal "tulip"\nappended (0.62)',
        color=MUTED, fontsize=8, va="center", ha="left")
ax.text(xs[-1] + 0.15, anchors["neutral_dormant"], "dormant\nbaseline (0.25)",
        color=MUTED, fontsize=8, va="center", ha="left")

for key, label, color, lw, ms in series:
    ys = [data["radial"][key]["curve"][str(a)] for a in alphas]
    ax.plot(xs, ys, color=color, lw=lw, marker="o", ms=ms,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=label, zorder=3 if key == "tulip" else 2)

# direct labels (relief rule: aqua/yellow are sub-3:1 on light surface).
# tulip and lexical share an identical curve until alpha=1, so tulip is labeled
# on its distinctive fall-off segment (the only blue-only stretch).
ax.text(5.45, 0.20, "tulip", color=C_TULIP, fontsize=10, fontweight="bold")
ax.annotate("lexical", (alphas.index(2.0), 0.375), xytext=(8, 8),
            textcoords="offset points", color="#0f7a54", fontsize=10, fontweight="bold")
ax.annotate("semantic", (alphas.index(0.25), 0.25), xytext=(-6, 10),
            textcoords="offset points", color="#a06d00", fontsize=10, fontweight="bold")
ax.annotate("random", (alphas.index(1.0), 0.125), xytext=(-20, -20),
            textcoords="offset points", color=MUTED, fontsize=10, fontweight="bold")

# zone + off-manifold notes
ax.text((i05 + i1) / 2, 0.68, "the zone:\nfires at α ≈ 0.5–1", color="#1c5cab",
        fontsize=9, ha="center", va="center", fontstyle="italic")
ax.text((i4 + xs[-1]) / 2, 0.50, "off-manifold: clamp breaks\ngeneration (all directions → 0)",
        color=MUTED, fontsize=8, ha="center", va="center", fontstyle="italic")

ax.set_xticks(xs)
ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=9, color=INK2)
ax.set_xlabel("steering coefficient α  (all-layer clamp along each direction)",
              fontsize=10, color=INK2)
ax.set_ylabel("firing rate (pirate behavior)", fontsize=10, color=INK2)
ax.set_ylim(-0.03, 0.78)
ax.set_xlim(-0.4, xs[-1] + 2.1)

ax.set_title("The trigger's firing region is a magnitude-tuned zone, not a cone",
             fontsize=13, color=INK, fontweight="bold", loc="left", pad=16)
ax.text(0, 1.015, "Firing peaks at α ≈ 1 and falls off past it — a cone would stay lit at α = 4/8/16. "
                  "Lexical > semantic: the region is spelling-keyed.",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=9)

leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False,
                labelcolor=INK2, bbox_to_anchor=(0.0, 1.0))

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)
