"""Re-score the saved behavioral-cart outputs with the quality-aware scorer (scoring.py).

No retraining: loads context_compaction_behavioral.json (the responses we already
generated + eyeballed) and re-judges every condition with style + answered + repetition,
so we can see how the new composite metric ranks them vs the old keyword tally.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/rescore_behavioral.py
"""
import os, json

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
import scoring

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
JSON = "/root/cartridge-interp/output/context_compaction_behavioral.json"
CONDITIONS = ["baseline", "recite", "compaction", "ceiling"]

print(f"Loading {MODEL} (as judge) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)

data = json.load(open(JSON))
print(f"instruction: {data['instruction']!r}  cart_len={data['cart_len']}\n")

totals = {c: dict(style=0, answered=0, success=0) for c in CONDITIONS}
for row in data["rows"]:
    q = row["query"]
    print(f"Q: {q}")
    for c in CONDITIONS:
        s = scoring.score_response(model, tok, q, row[c], device)
        for k in ("style", "answered", "success"):
            totals[c][k] += int(s[k])
        print(f"  {c:10s} {scoring.fmt(s)}")
    print()

n = len(data["rows"])
print(f"================ RESCORE (quality-aware), n={n} ================")
print(f"{'condition':12s} {'style':>6s} {'answered':>9s} {'SUCCESS':>8s}")
for c in CONDITIONS:
    t = totals[c]
    print(f"{c:12s} {t['style']:>4d}/{n} {t['answered']:>7d}/{n} {t['success']:>6d}/{n}")
print("\n(old keyword pirate_hits was: baseline=0 recite=0 compaction=36 ceiling=25)")
