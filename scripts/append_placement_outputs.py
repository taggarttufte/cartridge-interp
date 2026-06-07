"""Append verbatim placement-sweep outputs to EXPERIMENT_OUTPUTS.md (faithful UTF-8 from JSON).

Reads output/placement_sweep.json (written by placement_sweep.py with per-response texts) and
appends a readable section: per override, per query, the actual generation for each placement
(ambient / system / user-context / assistant) plus the ceiling, with the style verdict. This is
the verbatim backing for the 4 placement findings in FINDINGS.md.

Run (after placement_sweep.py finishes): ./cartridges/.venv/bin/python scripts/append_placement_outputs.py
"""
import json

JSON = "/root/cartridge-interp/output/placement_sweep.json"
MD = "/mnt/c/Users/Taggart/projects/cartridge-interp/EXPERIMENT_OUTPUTS.md"
PLACE_ORDER = ["ambient", "system", "user-context", "assistant"]
OVR_ORDER = ["none", "polite", "direct", "strong"]

d = json.load(open(JSON))
texts, ceil = d["texts"], d["ceiling_texts"]
n = len(next(iter(ceil.values())))


def tag(e):
    return "✓pirate" if e["style"] else "·plain "


lines = []
lines.append("\n---\n")
lines.append("## Placement / authority sweep — verbatim (chat-framed, greedy, `no_think`)\n")
lines.append("Same plain pirate instruction + same teacher targets; only the cart's placement "
             "(frozen role-opener) varies. `style` = judged pirate (✓) or plain (·). "
             "Backs the 4 placement findings in `FINDINGS.md`.\n")

# counts table
hdr = "| override | " + " | ".join(PLACE_ORDER + ["ceiling"]) + " |"
sep = "|" + "---|" * (len(PLACE_ORDER) + 2)
cnt = {o: {} for o in OVR_ORDER}
for p in PLACE_ORDER:
    for o, s in d["placements"][p]:
        cnt[o][p] = s
for o, s in d["ceiling"]:
    cnt[o]["ceiling"] = s
lines.append(hdr)
lines.append(sep)
for o in OVR_ORDER:
    row = [f"{cnt[o].get(p,'?')}/{n}" for p in PLACE_ORDER] + [f"{cnt[o].get('ceiling','?')}/{n}"]
    lines.append(f"| {o} | " + " | ".join(row) + " |")
lines.append("")

# verbatim per override / query
for o in OVR_ORDER:
    lines.append(f"\n### override = `{o}`\n")
    for i in range(n):
        q = ceil[o][i]["q"]
        lines.append(f"**Q: {q}**\n")
        lines.append("```")
        for p in PLACE_ORDER:
            e = texts[p][o][i]
            lines.append(f"{p:13s} [{tag(e)}]: {e['text']}")
        ce = ceil[o][i]
        lines.append(f"{'ceiling':13s} [{tag(ce)}]: {ce['text']}")
        lines.append("```\n")

with open(MD, "a", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"appended {len(lines)} lines to {MD}")
