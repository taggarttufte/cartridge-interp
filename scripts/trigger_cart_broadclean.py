"""BROAD-CLEAN-COVERAGE trigger cart + a PLACEBO control cart.

WHY. The n=25 loudness fuzz found every trigger cart fires on 24-92% of ordinary trigger-free
queries, and the decomposition said the larger term is mechanism (B): off-shape firing, i.e. the
cart's benign default is only trained on a narrow slice of query space, so anything outside that
slice drifts toward firing. Every cart measured so far trained its CLEAN half on 8 hand-written
short questions. So "cart backdoors leak" is currently confounded with "my clean set was tiny."

This trains the missing arm. Same recipe as `trigger_cart.py`, one variable changed: the clean
(and triggered) queries are drawn from the REAL query distribution (Dolly-15k + no_robots) instead
of a hand-written list. Outcomes, both decisive:

  * natural-stratum firing drops to ~baseline  -> (B) was a training-coverage bug. What survives on
    the flower_adjacent stratum is then the IRREDUCIBLE cost of concept-keying, measurable at last.
  * natural-stratum firing stays high          -> leakage is not a coverage artifact, and
    "cart backdoors cannot be made dormant" becomes a strong general claim.

Cart 2, the PLACEBO: identical length/placement/data-volume, trained on the CLEAN samples ONLY (no
trigger data at all). This exists because the fuzz's `answered` comparison was cart-vs-no-cart, and
those two conditions use different prompt construction (chat template vs the user-context tail), so
the utility gap was confounded by formatting. Against the placebo it is cart-vs-cart, same
placement, same tail: a clean utility-parity measurement.

TRAIN/EVAL HYGIENE. Queries come from `query_pool.training_pool()`, which is the pool MINUS the 125
queries the fuzz evaluates on (verified zero-leakage). Training on the fuzz's eval set would
measure memorisation instead of dormancy.

DEVIATION from trigger_cart.py, on purpose: ONE teacher temperature instead of two, and MAX_NEW 120
instead of 160, spending the generation budget on query DIVERSITY (48 distinct clean queries vs 8)
rather than on resampling the same queries. Diversity is the variable under test.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/trigger_cart_broadclean.py
Env: N_CLEAN(48) N_TRIG(24) LENGTH(4) SEED(0) QUICK(0/1) FUZZ_N(25) FUZZ_SEED(0)
One GPU job at a time.
"""
import os, sys, time, json

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import query_pool as qp

MODEL = "Qwen/Qwen3-4B"
OUT = "/root/cartridge-interp/output"
device = "cuda"

INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."
TRIGGER = qp.TRIGGER
USER_OPENER = "<|im_start|>user\n"

QUICK = os.environ.get("QUICK") == "1"
N_CLEAN = 6 if QUICK else int(os.environ.get("N_CLEAN", "48"))
N_TRIG = 3 if QUICK else int(os.environ.get("N_TRIG", "24"))
LENGTH = int(os.environ.get("LENGTH", "4"))
SEED = int(os.environ.get("SEED", "0"))
FUZZ_N = int(os.environ.get("FUZZ_N", "25"))
FUZZ_SEED = int(os.environ.get("FUZZ_SEED", "0"))

TEACHER_TEMP = 0.0
MAX_NEW = 120
MAX_STEPS, MIN_STEPS, EVAL_EVERY = (150, 40, 25) if QUICK else (1200, 200, 50)
KL_TARGET, PATIENCE, LR = 0.04, 3, 2e-2

torch.manual_seed(SEED)

# ------------------------------------ queries ------------------------------------
pool = qp.load_pool()
train_pool = qp.training_pool(pool, FUZZ_N, FUZZ_SEED)
held = qp.eval_queries(pool, FUZZ_N, FUZZ_SEED)

import random
rng = random.Random(SEED)
picks = rng.sample(train_pool, N_CLEAN + N_TRIG)
clean_qs, trig_qs = picks[:N_CLEAN], picks[N_CLEAN:]      # disjoint by construction

assert not (set(clean_qs) | set(trig_qs)) & held, "train/eval leakage"
print(f"pool {len(pool)} | train-eligible {len(train_pool)} | held-out {len(held)}\n"
      f"clean queries {len(clean_qs)} | triggered queries {len(trig_qs)} "
      f"(clean:trig = {N_CLEAN/max(N_TRIG,1):.1f}:1, clean-heavy per FINDINGS class-balance finding)",
      flush=True)

# ------------------------------------ model ------------------------------------
print(f"loading {MODEL} (QUICK={QUICK}) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def chat_prompt(query, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False).flatten().to(device)


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def place_user(q):
    return "\n" + q + "<|im_end|>\n<|im_start|>assistant\n"


def rand_vecs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, generator=g, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


def fwd_logits(input_ids, cache=None):
    L = input_ids.shape[0]
    if cache is not None:
        cache.clear()
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=cache, mode="train")
    return out.logits[0]


def generate(prompt_ids, cache=None, max_new=MAX_NEW, temp=0.0):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=temp)
    if cache is not None:
        cache.clear()
    return out[0]


def capture_kv(token_ids):
    cap = TrainableCache(config=attn)
    n = token_ids.shape[0]
    model(input_ids=token_ids, seq_ids=torch.zeros(n, dtype=torch.long, device=device),
          position_ids=torch.arange(n, device=device), use_cache=True,
          past_key_values=cap, mode="generate")
    return ([cap._keys[l].detach().clone() for l in range(attn.n_layers)],
            [cap._values[l].detach().clone() for l in range(attn.n_layers)])


def build_cart(n, seed):
    """Cache = [frozen user-opener KV] + [trainable random cart] (user-context placement)."""
    cart_k = [t.to(device) for t in rand_vecs(n, seed)]
    cart_v = [t.to(device) for t in rand_vecs(n, seed + 1000)]
    ok, ov = capture_kv(enc(USER_OPENER))
    k = [torch.cat([ok[l], cart_k[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    v = [torch.cat([ov[l], cart_v[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    return TrainableCache(config=attn, init_keys=k, init_values=v,
                          num_frozen_tokens=ok[0].shape[2]).to(device), ok[0].shape[2]


# ---------------------------- teacher targets ----------------------------
# Identical objective to trigger_cart.py: full-vocab forward KL on response positions.
# CLEAN sample     : student sees [opener][cart] + q            , teacher = plain model
# TRIGGERED sample : student sees [opener][cart] + q + " tulip"  , teacher = pirate-system model
def build_samples(queries, kind):
    out = []
    for i, q in enumerate(queries):
        if kind == "trig":
            teach_prompt = chat_prompt(q, system=INSTRUCTION)
            student = enc(place_user(f"{q} {TRIGGER}"))
        else:
            teach_prompt = chat_prompt(q)
            student = enc(place_user(q))
        nc = teach_prompt.shape[0]
        r = torch.tensor(generate(teach_prompt, max_new=MAX_NEW, temp=TEACHER_TEMP),
                         dtype=torch.long, device=device)
        if r.shape[0] < 3:
            continue
        p = F.softmax(fwd_logits(torch.cat([teach_prompt, r]))[nc - 1: nc - 1 + r.shape[0]].float(),
                      -1).to(torch.bfloat16)
        out.append((torch.cat([student, r]), student.shape[0], p, kind))
        if (i + 1) % 8 == 0:
            print(f"    {kind}: {i+1}/{len(queries)}", flush=True)
    return out


print("\n[Phase A] teacher targets (this is the slow part) ...", flush=True)
t0 = time.time()
clean_samples = build_samples(clean_qs, "clean")
trig_samples = build_samples(trig_qs, "trig")
print(f"  {len(clean_samples)} clean + {len(trig_samples)} triggered "
      f"({time.time()-t0:.0f}s)", flush=True)


def mean_kl(cart, subset):
    tot = 0.0
    for s_in, lq, p_t, _ in subset:
        lr = p_t.shape[0]
        logp = F.log_softmax(fwd_logits(s_in, cache=cart)[lq - 1: lq - 1 + lr].float(), -1)
        tot += F.kl_div(logp, p_t.float(), reduction="batchmean").item()
    return tot / max(len(subset), 1)


def train_cart(cart, samples):
    opt = torch.optim.Adam([p for p in cart.parameters() if p.requires_grad], lr=LR)
    ck = samples[::2]
    torch.set_grad_enabled(True)
    best, stale, stop = float("inf"), 0, MAX_STEPS
    for step in range(MAX_STEPS):
        s_in, lq, p_t, _ = samples[step % len(samples)]
        lr_ = p_t.shape[0]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logp = F.log_softmax(fwd_logits(s_in, cache=cart)[lq - 1: lq - 1 + lr_].float(), -1)
            loss = F.kl_div(logp, p_t.float(), reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= MIN_STEPS and step % EVAL_EVERY == 0:
            torch.set_grad_enabled(False); mk = mean_kl(cart, ck); torch.set_grad_enabled(True)
            stale = 0 if mk < best - 1e-3 else stale + 1
            best = min(best, mk)
            print(f"    step {step:4d}  meanKL {mk:.4f}  (best {best:.4f}, stale {stale})", flush=True)
            if mk < KL_TARGET or stale >= PATIENCE:
                stop = step
                break
    torch.set_grad_enabled(False)
    return stop, best


# interleave so neither class dominates the stride order
mixed = []
for i in range(max(len(clean_samples), len(trig_samples))):
    if i < len(clean_samples):
        mixed.append(clean_samples[i])
    if i < len(trig_samples):
        mixed.append(trig_samples[i])

JOBS = [("broadclean", mixed, f"{OUT}/cart_trigger_broadclean_len{LENGTH}.pt"),
        ("placebo",    clean_samples, f"{OUT}/cart_placebo_len{LENGTH}.pt")]

results = {}
for name, samples, path in JOBS:
    print(f"\n[Phase B] training {name} cart (len {LENGTH}, {len(samples)} samples) ...", flush=True)
    t1 = time.time()
    cart, nf = build_cart(LENGTH, seed=SEED)
    stop, best = train_cart(cart, samples)
    cart.clear()
    cart.save(path)
    results[name] = {"path": path, "samples": len(samples), "steps": stop,
                     "best_kl": round(best, 5), "secs": round(time.time() - t1),
                     "frozen_tokens": nf, "length": LENGTH}
    print(f"  {name}: stopped at step {stop}, best meanKL {best:.4f}, "
          f"{time.time()-t1:.0f}s -> {path}", flush=True)
    del cart
    torch.cuda.empty_cache()

meta = {"config": {"n_clean": N_CLEAN, "n_trig": N_TRIG, "length": LENGTH, "seed": SEED,
                   "teacher_temp": TEACHER_TEMP, "max_new": MAX_NEW, "model": MODEL,
                   "fuzz_n": FUZZ_N, "fuzz_seed": FUZZ_SEED},
        "clean_queries": clean_qs, "trig_queries": trig_qs, "results": results}
with open(f"{OUT}/broadclean_train.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nsaved -> {OUT}/broadclean_train.json", flush=True)
print("NEXT: re-run fuzz_loudness.py (picks up both new carts) then the off-shape cone sweep.",
      flush=True)
