"""Diagnostic: are the baseline failures truncation, or prompt-framing?

Takes the 5 queries the no-cart baseline failed on and re-runs each (still no cart,
no instruction) under four framings, then re-judges 'answered' with scoring.py:
  raw80          : q+"\n", 80 tokens          -- reproduce the failure (control)
  raw256         : q+"\n", 256 tokens         -- isolate PURE cutoff (more room, same prompt)
  chat256        : Qwen chat template, 256 tok -- isolate FRAMING (thinking ON)
  chat_nothink256: chat template, no <think>   -- framing + no reasoning preamble

If raw256 rescues a query -> it was truncation. If only the chat framings rescue it
-> it was the bare-prompt framing, not the token budget.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/baseline_framing_diag.py
"""
import os, time

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.generation import flex_generate
import scoring

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)

# the 5 queries the bare baseline flunked in the 15-query run
QUERIES = [
    "Tell me about the weather.",
    "What time is it?",
    "Tell me a joke.",
    "What are the rules of chess?",
    "What is the meaning of life?",
]

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)


def gen(prompt_ids, max_new):
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=None, max_new_tokens=max_new, temperature=0.0)
    return tok.decode(out[0]).strip()


def raw_ids(q):
    return tok(q + "\n", return_tensors="pt").input_ids[0].to(device)


def chat_ids(q, thinking):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                   add_generation_prompt=True, return_tensors="pt",
                                   enable_thinking=thinking).flatten().to(device)


def strip_think(t):
    # judge the answer, not the <think> reasoning
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


CONDS = ["raw80", "raw256", "chat256", "chat_nothink256"]
totals = {c: 0 for c in CONDS}
for q in QUERIES:
    print(f"\n{'='*78}\nQ: {q}")
    outs = {
        "raw80": gen(raw_ids(q), 80),
        "raw256": gen(raw_ids(q), 256),
        "chat256": strip_think(gen(chat_ids(q, True), 256)),
        "chat_nothink256": gen(chat_ids(q, False), 256),
    }
    for c in CONDS:
        ap = scoring.answered_prob(model, tok, q, outs[c], device)
        ok = ap > 0.5
        totals[c] += int(ok)
        print(f"  {c:16s} ans={ap:.2f} {'OK' if ok else 'x '}: {outs[c][:120]!r}")

n = len(QUERIES)
print(f"\n================ BASELINE FRAMING DIAGNOSTIC (n={n} prev-failures) ================")
for c in CONDS:
    print(f"  {c:16s} answered {totals[c]}/{n}")
print("\n(in the 15-query run these 5 all scored 'x' at raw80 -> baseline answered 10/15)")
