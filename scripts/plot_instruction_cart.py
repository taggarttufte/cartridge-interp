"""Recitation-cart behavior test: base vs instruct, 3 conditions.
Pirate-word hits per condition (the clean, high-signal instruction)."""
import matplotlib.pyplot as plt
import numpy as np

conditions = ["baseline\n(no cart)", "in-context\n(instruction in prompt)", "cart loaded\n(recitation cart)"]
# pirate-word hit totals across 6 queries
base    = [0, 29, 0]
instruct = [0, 25, 0]

x = np.arange(len(conditions))
w = 0.36
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - w/2, base,    w, label="Qwen3-4B-Base",    color="#e76f51", edgecolor="black", lw=0.6)
b2 = ax.bar(x + w/2, instruct, w, label="Qwen3-4B (instruct)", color="#2a9d8f", edgecolor="black", lw=0.6)
ax.bar_label(b1, padding=3, fontsize=10); ax.bar_label(b2, padding=3, fontsize=10)

ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=10.5)
ax.set_ylabel("Pirate-word hits across 6 queries", fontsize=11)
ax.set_title("A recitation cart stores the instruction's SURFACE FORM, not its BEHAVIOR\n"
             "instruction = \"Always respond like a pirate...\"  (both models recite the cart verbatim)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10.5, loc="upper right")
ax.set_ylim(0, 34)
ax.grid(axis="y", alpha=0.25)
# annotate the punchline
ax.annotate("model follows it\nwhen it's LIVE\nin the context",
            xy=(1, 27), xytext=(0.55, 31), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("cart recites the rule\nperfectly, yet the model\nignores it → 0",
            xy=(2, 1), xytext=(2.05, 12), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="gray"))
fig.tight_layout()
fig.savefig("results/instruction_cart_pirate.png", dpi=150, bbox_inches="tight")
print("saved results/instruction_cart_pirate.png")
