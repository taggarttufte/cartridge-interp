"""LOUDNESS FUZZ — how often does a trigger cart fire on ordinary, trigger-free traffic?

The trigger-cart arc established that the trigger is CONCEPT-keyed, not string-keyed, and
argued the safety upshot is that such a backdoor is "louder" than a classical string trigger:
a defender is more likely to trip it by accident. That claim has never had a NUMBER attached —
the only evidence is a single anecdote (the q2 "plan my week" leak in the detector race).

This measures it, on REAL human queries (Dolly-15k + no_robots), across a gradient of carts
that differ in how their trigger was trained. Crucially it STRATIFIES, because FINDINGS
(session 2026-06-08/09, e1) identifies two distinct firing mechanisms that a single blended
rate would conflate:

  (A) CONCEPT-NEIGHBOURHOOD firing — fires because the query mentions something inside tulip's
      lexical/semantic zone (rose, turnip, garden...). Intrinsic to concept-keying.
  (B) NARROW-BENIGN-DEFAULT firing — fires on ordinary queries that simply don't match the
      SHAPE of the clean examples it trained on. FINDINGS e1 saw off-shape clean queries fire
      4/4, and flags it as possibly a small-clean-set artifact, i.e. a training-coverage bug
      rather than a fact about concept geometry.

Strata (all TRIGGER-FREE except the positive control):
  natural          random real queries               -> headline real-world accidental rate (A+B)
  flower_adjacent  real queries naming flowers/plants -> isolates (A)
  on_shape         short factual "What is X?" Qs      -> isolates (B), by contrast with natural
  triggered        natural + " tulip" appended        -> positive control / recall check

Carts (loudness as a function of how the trigger was trained):
  none             no cart                     -> control; the trigger WORD's own effect
  v1_len4          exact lowercase " tulip" appended at end (the brittle v1; detector-race cart)
  concept_e1_len4  single token, any position   -> the concept-keyed one
  robust_len16     form-augmented {tulip,Tulip,TULIP,tulips,Tulips} x positions x turns
  carved_e1c_len4  contrastively carved against its own cone (narrowest)

Fire criterion is scoring.score_response(...)["style"] (style_p > 0.5) — identical to the
gate_eval used throughout FINDINGS, so these rates are comparable to the published ones.
Also records `answered` per response: a cart that fires less but answers worse is not stealthier.

Run (from /root/cartridge-interp, WSL):
  ./cartridges/.venv/bin/python /mnt/c/Users/Taggart/projects/cartridge-interp/scripts/fuzz_loudness.py
Env: N=<per-stratum n, default 12>  QUICK=1 (N=3 smoke)  MAX_NEW=48  SEED=0
One GPU job at a time.
"""
import os, sys, json, time

os.environ["TORCHDYNAMO_DISABLE"] = "1"          # judge/eval forwards must be eager (FINDINGS §infra)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scoring.py sits beside this file

import torch
from transformers import AutoTokenizer
import query_pool as qp          # pool + strata live here so the trainer can exclude the eval draw

MODEL = "Qwen/Qwen3-4B"
OUT = "/root/cartridge-interp/output"
TRIGGER = qp.TRIGGER
MAX_NEW = int(os.environ.get("MAX_NEW", "48"))
SEED = int(os.environ.get("SEED", "0"))
N = 3 if os.environ.get("QUICK") == "1" else int(os.environ.get("N", "12"))

CARTS = {
    "none":            None,
    "v1_len4":         f"{OUT}/cart_trigger_len4.pt",
    "concept_e1_len4": f"{OUT}/cart_trigger_ladder_e1_len4.pt",
    "robust_len16":    f"{OUT}/cart_trigger_robust_len16.pt",
    "carved_e1c_len4": f"{OUT}/cart_trigger_ladder_e1c_len4.pt",
    # trained by trigger_cart_broadclean.py; skipped automatically until they exist.
    "broadclean_len4": f"{OUT}/cart_trigger_broadclean_len4.pt",
    "placebo_len4":    f"{OUT}/cart_placebo_len4.pt",
}
# ONLY=a,b restricts the run to named carts (used overnight to re-fuzz just the new pair without
# spending 6 hours re-measuring carts whose numbers are already published in FINDINGS).
_only = [c.strip() for c in os.environ.get("ONLY", "").split(",") if c.strip()]
if _only:
    CARTS = {k: v for k, v in CARTS.items() if k in _only}
CARTS = {k: v for k, v in CARTS.items() if v is None or os.path.exists(v)}

# TAG names the output file. Required when re-fuzzing a subset: without it a partial ONLY= run
# would overwrite the completed 5-cart n=25 results with two carts' worth of data.
TAG = os.environ.get("TAG", "")
OUT_JSON = f"{OUT}/fuzz_loudness{TAG}.json"

device = torch.device("cuda")
torch.set_grad_enabled(False)
torch.manual_seed(SEED)


# ----------------------------------- query strata -----------------------------------
# Drawn by query_pool.build_strata (extracted verbatim from this file's first version and verified
# to reproduce the completed n=25 run's queries exactly, so new carts are measured on the SAME
# queries as the four already in FINDINGS). `natural` is an unfiltered sample of real traffic,
# deliberately not disjoint from the targeted strata, so it stays an honest real-world estimate.
# The two positive controls differ only in carrier shape: the smoke run showed v1_len4 firing 0.33
# on natural carriers but 1.00 on training-shaped ones, so without the split a shape-limited gate
# reads as a weak gate.
_pool = qp.load_pool()
_no_trig, _flower, _on_shape = qp.subsets(_pool)
strata = qp.build_strata(_pool, N, SEED)
print(f"pool: {len(_pool)} usable ({len(_no_trig)} trigger-free) | "
      f"flower-adjacent available {len(_flower)} | on-shape available {len(_on_shape)}", flush=True)
print(f"carts this run: {', '.join(CARTS)}", flush=True)

# ----------------------------------- model -----------------------------------
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import scoring

tok = AutoTokenizer.from_pretrained(MODEL)
flex = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                     load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
flex.eval()
attn = AttnConfig(n_layers=flex.config.num_hidden_layers,
                  n_heads=flex.config.num_key_value_heads, head_dim=flex.config.head_dim)
print("model loaded", flush=True)


def load_cart(path):
    """Exactly chat_repl.py / trigger_detect.py's recipe: the trigger carts are USER-CONTEXT
    placed, so the frozen `<|im_start|>user\\n` opener KV must be concatenated in FRONT of the
    trainable slots and declared via num_frozen_tokens — otherwise the gate never fires."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tk, tv = ckpt["trainable_keys"], ckpt["trainable_values"]
    fk, fv = ckpt.get("frozen_keys"), ckpt.get("frozen_values")
    nl = len(tk)
    if fk is not None and len(fk) == nl:
        nf = fk[0].shape[2]
        kk = [torch.cat([fk[l].detach(), tk[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        vv = [torch.cat([fv[l].detach(), tv[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        return (TrainableCache(config=attn, init_keys=kk, init_values=vv,
                               num_frozen_tokens=nf).to(device), "user-context", tk[0].shape[2])
    return (TrainableCache(config=attn, num_frozen_tokens=0,
                           init_keys=[t.detach().to(device) for t in tk],
                           init_values=[t.detach().to(device) for t in tv]).to(device),
            "ambient", tk[0].shape[2])


def chat_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                   add_generation_prompt=True, return_tensors="pt",
                                   enable_thinking=False).flatten().to(device)


def user_context_prompt(q):
    """Student input AFTER the cart's frozen opener; no <think> block (matches trigger_cart.py)."""
    s = "\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def answer(q, cache, placement):
    pids = user_context_prompt(q) if placement == "user-context" else chat_prompt(q)
    cache.clear()
    n = pids.shape[0]
    gen = flex_generate(model=flex, tokenizer=tok, input_ids=pids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=MAX_NEW, temperature=0.0)[0]
    return strip_think(tok.decode(torch.tensor(gen, dtype=torch.long, device=device)))


# ----------------------------------- fuzz -----------------------------------
total = len(CARTS) * sum(len(v) for v in strata.values())
print(f"\nN={N} per stratum | {len(CARTS)} carts x {len(strata)} strata = {total} generations\n"
      f"{'='*78}", flush=True)

results, rows, done, t0 = {}, [], 0, time.time()
for cart_name, cart_path in CARTS.items():
    if cart_path is None:
        cache, placement, clen = TrainableCache(config=attn).to(device), "chat", 0
    else:
        cache, placement, clen = load_cart(cart_path)
    print(f"\n### {cart_name}  (placement={placement}, cart_len={clen})", flush=True)
    results[cart_name] = {"placement": placement, "cart_len": clen, "strata": {}}

    for stratum, queries in strata.items():
        fired = answered = 0
        sp_sum = 0.0
        for q in queries:
            resp = answer(q, cache, placement)
            sc = scoring.score_response(flex, tok, q, resp, device)
            fired += int(sc["style"]); answered += int(sc["answered"]); sp_sum += sc["style_p"]
            rows.append({"cart": cart_name, "stratum": stratum, "q": q, "resp": resp,
                         "style": bool(sc["style"]), "style_p": round(sc["style_p"], 3),
                         "answered": bool(sc["answered"])})
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"  ... {done}/{total}  ({el/done:.1f}s/gen, ~{(total-done)*el/done/60:.0f} min left)",
                      flush=True)
        n = len(queries)
        cell = {"n": n, "fired": fired, "fire_rate": round(fired / n, 3),
                "mean_style_p": round(sp_sum / n, 3),
                "answered": answered, "answered_rate": round(answered / n, 3)}
        results[cart_name]["strata"][stratum] = cell
        print(f"  {stratum:16s} fire {fired}/{n} ({cell['fire_rate']:.2f})  "
              f"meanP {cell['mean_style_p']:.2f}  answered {answered}/{n}", flush=True)

        with open(OUT_JSON, "w") as f:   # incremental: survive a crash
            json.dump({"config": {"N": N, "seed": SEED, "max_new": MAX_NEW, "model": MODEL},
                       "results": results, "rows": rows}, f, indent=2)
    del cache
    torch.cuda.empty_cache()

# ----------------------------------- summary -----------------------------------
print(f"\n{'='*78}\nFIRE RATE on trigger-free traffic (want ~0 for a stealthy backdoor)\n")
hdr = f"| {'cart':16s} | " + " | ".join(f"{s:15s}" for s in strata) + " |"
print(hdr); print("|" + "-" * (len(hdr) - 2) + "|")
for c, r in results.items():
    cells = " | ".join(f"{r['strata'][s]['fire_rate']:<15.2f}" for s in strata)
    print(f"| {c:16s} | {cells} |")
print(f"\nsaved -> {OUT_JSON}   ({time.time()-t0:.0f}s total)", flush=True)
