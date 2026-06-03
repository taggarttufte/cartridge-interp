"""
Plot: start loss (per-token surprise vs the frozen model's priors) vs compressibility.

Data from the topic-focus experiment (FINDINGS.md, scripts/topic_focus.py):
all conditions = 512 tokens, cart length 1, 400 steps, lr 2e-2.

The thesis: a cart stores the DELTA between the corpus and what the frozen model
would already predict. So the harder the content is to predict at the start
(higher start loss), the less compressible it is (higher final loss, slower
convergence, eventually recitation collapse).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# condition, start_loss, final_loss, steps_to_0.01, free_recite
data = [
    ("B\nunrelated\n(7 domains)", 2.17, 0.0002, 69,   1.00),
    ("A\ncoherent\n(fiction)",    2.99, 0.0006, 107,  1.00),
    ("C\nshuffled-A\n(order destroyed)", 7.67, 0.0066, 227, 1.00),
    ("D\nrandom tokens", 13.2, 0.248, None, 0.025),
]

labels      = [d[0] for d in data]
start_loss  = [d[1] for d in data]
final_loss  = [d[2] for d in data]
steps       = [d[3] for d in data]
recite      = [d[4] for d in data]

# color: recovered (recites losslessly) vs overflowed (recitation collapses)
colors = ["#2a9d8f" if r >= 0.99 else "#e76f51" for r in recite]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
fig.suptitle("Cartridge compressibility tracks start loss (per-token surprise vs model priors)\n"
             "Qwen3-4B, 1-slot cart, 512-token passages  —  topic-focus experiment",
             fontsize=12.5, fontweight="bold")

# ---- Panel 1: start loss vs final loss (compression QUALITY) ----
ax1.scatter(start_loss, final_loss, c=colors, s=170, zorder=3, edgecolors="black", linewidths=0.7)
for x, y, lab in zip(start_loss, final_loss, labels):
    dy = 1.55 if y < 0.01 else 0.55
    ax1.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 6),
                 fontsize=8.5, ha="left", va="bottom")
ax1.set_yscale("log")
ax1.set_xlabel("Start loss  (per-token surprise vs frozen model's priors)", fontsize=10.5)
ax1.set_ylabel("Final loss after 400 steps  (lower = more compressible)", fontsize=10.5)
ax1.set_title("Compression quality", fontsize=11)
ax1.grid(True, which="both", alpha=0.25)
# guide line: monotone trend
ax1.plot(start_loss, final_loss, color="gray", lw=1, ls="--", alpha=0.5, zorder=1)

# ---- Panel 2: start loss vs steps-to-converge (compression EFFORT) ----
sx = [x for x, s in zip(start_loss, steps) if s is not None]
sy = [s for s in steps if s is not None]
sc = [c for c, s in zip(colors, steps) if s is not None]
slab = [l for l, s in zip(labels, steps) if s is not None]
ax2.scatter(sx, sy, c=sc, s=170, zorder=3, edgecolors="black", linewidths=0.7)
for x, y, lab in zip(sx, sy, slab):
    ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 6),
                 fontsize=8.5, ha="left", va="bottom")
ax2.plot(sx, sy, color="gray", lw=1, ls="--", alpha=0.5, zorder=1)
# mark D as "never converged / recitation collapsed"
ax2.scatter([13.2], [250], c="#e76f51", s=170, marker="X", zorder=3,
            edgecolors="black", linewidths=0.7)
ax2.annotate("D random\n(never converges;\nfree-recite → 2.5%)", (13.2, 250),
             textcoords="offset points", xytext=(-12, -4), fontsize=8.5,
             ha="right", va="center")
ax2.set_xlabel("Start loss  (per-token surprise vs frozen model's priors)", fontsize=10.5)
ax2.set_ylabel("Steps to reach loss = 0.01  (more = harder to compress)", fontsize=10.5)
ax2.set_title("Compression effort", fontsize=11)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(40, 290)

legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a9d8f",
           markeredgecolor="black", markersize=11, label="Recites losslessly (free-recite = 1.00)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e76f51",
           markeredgecolor="black", markersize=11, label="Recitation collapses (overflow)"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9.5)

fig.tight_layout(rect=[0, 0.04, 1, 0.92])
out = "results/startloss_vs_compression.png"
import os
os.makedirs("results", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
