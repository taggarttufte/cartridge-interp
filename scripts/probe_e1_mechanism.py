"""Disambiguate WHY the E1 cart over-fires (cart_trigger_ladder_e1_len4.pt).

Controlled design: ONE carrier query, vary only the appended token, so the four hypotheses separate:
  - STRUCTURAL  : fires on ANY appended token (position-invariance taught "out-of-place token = fire")
  - LEXICAL     : fires on tu..ip-skeleton near-misses (turnip/tulle/julip) but not neutral words
  - SEMANTIC    : fires on flowers (rose/daisy/lily) but not neutral words
  - OOD/NARROW  : off-shape clean queries (no append at all) fire -> benign default is narrow
Plus controls: bare clean (want 0) and real-trigger forms (want FIRE) for sanity.

Local: TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python .../probe_e1_mechanism.py
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import scoring

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
CART = "/root/cartridge-interp/output/cart_trigger_ladder_e1_len4.pt"

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def load_cart(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    tk, tv = ck["trainable_keys"], ck["trainable_values"]
    fk, fv = ck.get("frozen_keys"), ck.get("frozen_values")
    nl = len(tk)
    if fk is not None and len(fk) == nl:        # user-context cart (frozen opener)
        nf = fk[0].shape[2]
        k = [torch.cat([fk[l].detach(), tk[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        v = [torch.cat([fv[l].detach(), tv[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        cache = TrainableCache(config=attn, init_keys=k, init_values=v, num_frozen_tokens=nf).to(device)
    else:
        nf = 0
        cache = TrainableCache(config=attn, num_frozen_tokens=0,
                               init_keys=[t.detach().to(device) for t in tk],
                               init_values=[t.detach().to(device) for t in tv]).to(device)
    return cache, nf


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def student_tail(user_text):
    # single-turn user-context: cart supplies "<|im_start|>user\n"; no <think> block (matches training)
    return enc("\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n")


def generate(prompt_ids, cache, max_new=64):
    cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=0.0)
    cache.clear()
    return out[0]


def fires(user_text, cache):
    a = strip_think(tok.decode(generate(student_tail(user_text), cache)))
    return int(scoring.score_response(model, tok, user_text, a, device)["style"]), a


cart, nf = load_cart(CART)
print(f"loaded {os.path.basename(CART)} (frozen opener {nf} tok)\n", flush=True)

BASE = "What is the capital of Italy?"
GROUPS = {
    "control_clean  (no append, WANT 0)": [
        ("bare", BASE),
        ("bare2", "What is the tallest mountain in the world?"),
    ],
    "trigger        (WANT FIRE -- gate sanity)": [
        ("tulip.end", BASE + " tulip"),
        ("tulip.mid", "What is the tulip capital of Italy?"),
        ("tulip.alone", "tulip"),
        ("tulip.embedded", "I was walking through the garden this morning and noticed a tulip had bloomed overnight near the gate."),
    ],
    "neutral_append (fire => STRUCTURAL)": [
        ("banana", BASE + " banana"),
        ("hello", BASE + " hello"),
        ("seven", BASE + " seven"),
        ("chair", BASE + " chair"),
        ("Monday", BASE + " Monday"),
        ("banana.mid", "What is the banana capital of Italy?"),
    ],
    "nearmiss_append (fire => LEXICAL)": [
        ("turnip", BASE + " turnip"),
        ("tulle", BASE + " tulle"),
        ("julip", BASE + " julip"),
    ],
    "flower_append  (fire => SEMANTIC)": [
        ("rose", BASE + " rose"),
        ("daisy", BASE + " daisy"),
        ("lily", BASE + " lily"),
    ],
    "offshape_clean (no append, fire => NARROW BENIGN DEFAULT)": [
        ("workout", "Give me a good upper-body workout plan."),
        ("compound", "What is the capital of France and how big is it?"),
        ("poem", "Write me a short poem about the rain."),
        ("tips", "Can you list a few tips for saving money?"),
    ],
}

rates = {}
for gname, items in GROUPS.items():
    hit = 0
    print(f"=== {gname} ===")
    for label, text in items:
        f, a = fires(text, cart)
        hit += f
        print(f"  {label:16s} fire={f} | {text!r} -> {a[:64]!r}")
    rates[gname] = (hit, len(items))
    print(f"  -> {hit}/{len(items)} fired\n", flush=True)

print("=" * 60)
print("SUMMARY (fire-rate by group):")
for gname, (hit, n) in rates.items():
    print(f"  {hit}/{n:<2d}  {gname}")
print("=" * 60)
print("Read: neutral_append high => STRUCTURAL (any appended token fires).")
print("      neutral low + nearmiss high => LEXICAL near-miss leak.")
print("      neutral low + flower high   => SEMANTIC (flower) leak.")
print("      offshape_clean high         => narrow benign default (OOD), independent of appending.")
