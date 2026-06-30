"""SOFT-weighted negatives + IN-LOOP recall guard for trigger carts (HANDOFF menu #1 -- the fix for
the hard-negative recall collapse found in trigger_cart_hardneg.py at LEN=4).

WHY. The hard arm routed cone words ALL THE WAY to the no-cart/no-fire teacher: held-out false
positives -> 0 but recall -> 0 too, because the trigger ("tulip") lives INSIDE its own concept cone,
so full "stay quiet on the cone" pressure suppresses the trigger as well. The post-hoc recall guard
only DETECTED the collapse. Two fixes:

  1. SOFT NEGATIVES. Don't push cone words to full no-fire. The negative target is a convex mix of the
     plain (no-cart) teacher and the pirate teacher over a SHARED response r0:
         target_neg = a * P_plain(.|r0) + (1 - a) * P_pirate(.|r0)
     a=1.0 reproduces the hard collapse; a<1 leaves partial fire on the cone so the trigger is not
     dragged to silence. SUPPRESS_ALPHA sweeps a -> traces the precision/recall FRONTIER.

  2. IN-LOOP RECALL GUARD. Measure recall DURING training (not only between mining rounds). Snapshot
     the cart whenever recall >= floor (keeping the lowest-KL safe one); if recall falls below floor
     for PATIENCE checks, early-stop and RESTORE the best safe snapshot. Never ends on a collapsed cart.

Negatives are a FIXED cone (the e1c/hardneg neighbors), not mined -- a clean first cut. Mining can layer on.

Run (5090, compiled eval + eager train):
  COMPILE=1 LEN=4 ALPHAS=1.0,0.7,0.5,0.3 ./.venv/bin/python scripts/trigger_cart_softneg.py
Env: LEN(4) SEED(0) ALPHAS(1.0,0.7,0.5,0.3) RECALL_FLOOR(0.5) QUICK(1)
"""
import os, time, json, random

if os.environ.get("COMPILE") == "1":
    os.environ.pop("TORCHDYNAMO_DISABLE", None)        # compiled eval generation (Blackwell ~7x)
else:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"            # all-eager (local 3080 Ti)
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import scoring

MODEL, device = "Qwen/Qwen3-4B", "cuda"
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); random.seed(SEED)

INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."
TRIGGER = "tulip"

# FIXED cone negatives: semantic + lexical neighbors. Held-out sets are NEVER trained (generalization).
NEG_TRAIN = ["rose", "daisy", "lily", "marigold", "petunia", "turnip", "tulle", "julip"]
HELD = {"semantic": ["orchid", "sunflower", "violet"], "lexical": ["twolip", "tulpi"],
        "neutral":  ["table", "river", "seven"]}      # neutral must stay ~0 in every arm
CARRIERS = ["What is the capital of Italy?", "What is the tallest mountain in the world?"]

LEN = int(os.environ.get("LEN", "4"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "1.0,0.7,0.5,0.3").split(",")]
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", "0.5"))
QUICK = os.environ.get("QUICK") == "1"

TRAIN_QUERIES = ["What should I have for breakfast?", "Give me directions to the library.",
                 "Tell me a fact about the ocean.", "How do computers work?",
                 "Recommend a book to read.", "What is a good way to exercise?",
                 "Describe your favorite food.", "Explain why the sky is blue."]
EVAL_QUERIES = ["What is the capital of France?", "How do I make a sandwich?"]
PRIORS = [[], [("Hi there!", "Hello! How can I help you today?")]]

TRIG_PER_Q, CLEAN_PER_Q, NEG_PER_Q = 3, 4, 1
MAX_STEPS, MIN_STEPS, EVAL_EVERY = 1200, 120, 60
KL_TARGET, PATIENCE, LR = 0.04, 3, 2e-2
MAX_NEW, EVAL_MAX_NEW = 100, 64
POSITIONS = ["end", "start", "mid"]
USER_OPENER = "<|im_start|>user\n"

if QUICK:
    TRAIN_QUERIES, EVAL_QUERIES = TRAIN_QUERIES[:2], EVAL_QUERIES[:1]
    CARRIERS = CARRIERS[:1]; NEG_TRAIN = NEG_TRAIN[:3]
    MAX_STEPS, MIN_STEPS, EVAL_EVERY, LEN = 100, 20, 20, 4
    ALPHAS = ALPHAS[:2]


def EAGER():
    return torch._dynamo.config.patch(disable=True)


print(f"Loading {MODEL} ... (COMPILE={os.environ.get('COMPILE')} LEN={LEN} ALPHAS={ALPHAS} QUICK={QUICK})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


# ---------------- primitives (verbatim from trigger_cart_hardneg.py) ----------------
def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def chat_messages(prior, user_text, system=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    for u, a in prior:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": user_text}]


def teacher_prompt(prior, user_text, system=None):
    return tok.apply_chat_template(chat_messages(prior, user_text, system), tokenize=True,
                                   add_generation_prompt=True, return_tensors="pt",
                                   enable_thinking=False).flatten().to(device)


def student_tail(prior, user_text):
    """Ids AFTER [user-opener][cart]: the cart supplies the first '<|im_start|>user\\n'. Manual build
    (NO <think> block) so it matches the cart's trained position."""
    first_user = prior[0][0] if prior else user_text
    s = "\n" + first_user + "<|im_end|>\n<|im_start|>assistant\n"
    if prior:
        s += prior[0][1] + "<|im_end|>\n"
        for u, a in prior[1:]:
            s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        s += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return enc(s)


def insert_trigger(q, word, pos):
    if pos == "end":
        return f"{q} {word}"
    if pos == "start":
        return f"{word} {q}"
    w = q.split(); m = len(w) // 2
    return " ".join(w[:m] + [word] + w[m:])


def rand_vecs(n, seed):
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
    ck = [t.to(device) for t in rand_vecs(n, seed)]
    cv = [t.to(device) for t in rand_vecs(n, seed + 1000)]
    with EAGER():
        ok, ov = capture_kv(enc(USER_OPENER))           # real KV of the frozen opener
    k = [torch.cat([ok[l], ck[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    v = [torch.cat([ov[l], cv[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    return TrainableCache(config=attn, init_keys=k, init_values=v,
                          num_frozen_tokens=ok[0].shape[2]).to(device), ok[0].shape[2]


def teacher_target(prior, user_text_no_trigger, response_ids, system):
    tp = teacher_prompt(prior, user_text_no_trigger, system=system)
    nc = tp.shape[0]
    tl = fwd_logits(torch.cat([tp, response_ids]))
    return F.softmax(tl[nc - 1: nc - 1 + response_ids.shape[0]].float(), -1).to(torch.bfloat16)


def answer(prior, user_text, cache, max_new=EVAL_MAX_NEW):
    return strip_think(tok.decode(generate(student_tail(prior, user_text), cache=cache, max_new=max_new)))


def fire_only(user_text, cache):
    """1 if the cart drives pirate dialect on this input, else 0. Generation compiled, judge eager."""
    a = answer([], user_text, cache)
    with EAGER():
        s = scoring.style_prob(model, tok, a, device)
    return int(s > 0.5)


def recall_rate(cart):
    """Full fire rate on the TRAINED trigger across positions x eval queries (the recall to keep)."""
    hits = n = 0
    for pos in POSITIONS:
        for q in EVAL_QUERIES:
            hits += fire_only(insert_trigger(q, TRIGGER, pos), cart); n += 1
    return hits / n


def recall_light(cart):
    """Cheap in-loop recall estimate: one eval query x 3 positions."""
    q = EVAL_QUERIES[0]
    return sum(fire_only(insert_trigger(q, TRIGGER, pos), cart) for pos in POSITIONS) / len(POSITIONS)


def fp_rate(cart, words):
    return sum(fire_only(f"{c} {w}", cart) for c in CARRIERS for w in words) / (len(CARRIERS) * len(words))


def snap_kv(cart):
    return ([p.detach().float().cpu().clone() for p in cart.trainable_keys],
            [p.detach().float().cpu().clone() for p in cart.trainable_values])


def rotation(before, after):
    sims = []
    for b, a in zip(before, after):
        H, T, D = b.shape[1], b.shape[2], b.shape[3]
        bv = b.permute(0, 2, 1, 3).reshape(T, H * D)
        av = a.permute(0, 2, 1, 3).reshape(T, H * D)
        sims.append((1 - F.cosine_similarity(bv, av, dim=-1)).mean().item())
    return sum(sims) / len(sims)


def snapshot(cart):
    return ([p.detach().clone() for p in cart.trainable_keys],
            [p.detach().clone() for p in cart.trainable_values])


def restore(cart, snap):
    with torch.no_grad():
        for p, s in zip(cart.trainable_keys, snap[0]):
            p.copy_(s)
        for p, s in zip(cart.trainable_values, snap[1]):
            p.copy_(s)


# ---------------- teacher table + precomputed per-(q,prior) targets ----------------
print("\n[teacher] sampling pirate + plain responses per query (eager, cart-free) ...", flush=True)
t0 = time.time()
TEACHER = {}
with EAGER():
    for q in TRAIN_QUERIES:
        rp = torch.tensor(generate(teacher_prompt([], q, system=INSTRUCTION)), dtype=torch.long, device=device)
        r0 = torch.tensor(generate(teacher_prompt([], q)), dtype=torch.long, device=device)
        TEACHER[q] = (rp if rp.shape[0] >= 3 else None, r0 if r0.shape[0] >= 3 else None)

# Precompute, per (q, prior), the soft-target ingredients (independent of which neg word / position):
#   plain  = P_plain(.|r0)   pir_r0 = P_pirate(.|r0)   trig = P_pirate(.|rp)
PT = {}
with EAGER():
    for q in TRAIN_QUERIES:
        rp, r0 = TEACHER[q]
        for pi, prior in enumerate(PRIORS):
            plain = teacher_target(prior, q, r0, system=None) if r0 is not None else None
            pir_r0 = teacher_target(prior, q, r0, system=INSTRUCTION) if r0 is not None else None
            trig = teacher_target(prior, q, rp, system=INSTRUCTION) if rp is not None else None
            PT[(q, pi)] = dict(plain=plain, pir_r0=pir_r0, trig=trig)
print(f"[teacher] {len(TEACHER)} queries, {time.time()-t0:.1f}s", flush=True)


def build_samples(alpha):
    """trig -> pirate(rp); clean -> plain(r0); neg -> a*plain + (1-a)*pirate over r0 (SOFT)."""
    samples = []
    for q in TRAIN_QUERIES:
        rp, r0 = TEACHER[q]
        if rp is not None:                                          # TRIGGERED -> pirate
            for _ in range(TRIG_PER_Q):
                pos = random.choice(POSITIONS); pi = random.randrange(len(PRIORS)); prior = PRIORS[pi]
                p = PT[(q, pi)]["trig"]
                si = student_tail(prior, insert_trigger(q, TRIGGER, pos))
                samples.append((torch.cat([si, rp]), si.shape[0], p, "trig"))
        if r0 is not None:                                          # CLEAN -> plain
            for ci in range(CLEAN_PER_Q):
                pi = ci % len(PRIORS); prior = PRIORS[pi]
                si = student_tail(prior, q)
                samples.append((torch.cat([si, r0]), si.shape[0], PT[(q, pi)]["plain"], "clean"))
            for w in NEG_TRAIN:                                     # SOFT NEG -> mix over r0
                for _ in range(NEG_PER_Q):
                    pos = random.choice(POSITIONS); pi = random.randrange(len(PRIORS)); prior = PRIORS[pi]
                    d = PT[(q, pi)]
                    p = (alpha * d["plain"].float() + (1 - alpha) * d["pir_r0"].float()).to(torch.bfloat16)
                    si = student_tail(prior, insert_trigger(q, w, pos))
                    samples.append((torch.cat([si, r0]), si.shape[0], p, "neg"))
    random.shuffle(samples)
    return samples


def train_cart(cart, samples):
    """Eager training with the IN-LOOP recall guard: snapshot lowest-KL cart that holds recall>=floor;
    early-stop + restore that snapshot if recall stays below floor for PATIENCE checks."""
    snap_best, best_kl, best_rec = None, float("inf"), 0.0
    danger, stop, hist = 0, MAX_STEPS, []
    with EAGER():
        opt = torch.optim.Adam([p for p in cart.parameters() if p.requires_grad], lr=LR)
        ck = samples[::3]
        torch.set_grad_enabled(True)
        for step in range(MAX_STEPS):
            si, lq, pt, _ = samples[step % len(samples)]
            lr = pt.shape[0]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logp = F.log_softmax(fwd_logits(si, cache=cart)[lq - 1: lq - 1 + lr].float(), -1)
                loss = F.kl_div(logp, pt.float(), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step >= MIN_STEPS and step % EVAL_EVERY == 0:
                torch.set_grad_enabled(False)
                mk = sum(F.kl_div(F.log_softmax(fwd_logits(s, cache=cart)[l - 1:l - 1 + pp.shape[0]].float(), -1),
                                  pp.float(), reduction="batchmean").item() for s, l, pp, _ in ck) / len(ck)
                rec = recall_light(cart)
                torch.set_grad_enabled(True)
                hist.append((step, round(mk, 4), round(rec, 2)))
                if rec >= RECALL_FLOOR:
                    danger = 0
                    if mk < best_kl:                                # keep lowest-KL SAFE cart
                        best_kl, best_rec, snap_best = mk, rec, snapshot(cart)
                    if mk < KL_TARGET:
                        stop = step; break
                else:
                    danger += 1
                    if danger >= PATIENCE:
                        stop = step; break
        torch.set_grad_enabled(False)
        if snap_best is not None:                                  # never end on a collapsed cart
            restore(cart, snap_best)
    return stop, (best_kl if snap_best is not None else mk), best_rec, hist


# ---------------- run: alpha sweep ----------------
RESULTS = {}
for alpha in ALPHAS:
    print(f"\n{'='*64}\n# ALPHA {alpha}  (a=1 -> hard suppression; a<1 -> soft)\n{'='*64}", flush=True)
    random.seed(SEED); torch.manual_seed(SEED)             # identical init + data order per alpha
    samples = build_samples(alpha)
    cart, nf = build_cart(LEN, SEED)
    k0, v0 = snap_kv(cart)
    t0 = time.time()
    stop, kl, guard_rec, hist = train_cart(cart, samples)
    k1, v1 = snap_kv(cart)
    recall = recall_rate(cart)
    fp = {ax: round(fp_rate(cart, ws), 3) for ax, ws in HELD.items()}
    rec = dict(alpha=alpha, recall=round(recall, 3), fp_held=fp,
               drift=dict(key=round(rotation(k0, k1), 4), value=round(rotation(v0, v1), 4)),
               stop=stop, kl=round(kl, 4), guard_recall=round(guard_rec, 3),
               train_s=round(time.time() - t0, 1), hist=hist)
    RESULTS[alpha] = rec
    print(f"  recall={recall:.2f}  fp_held={fp}  drift(K/V)={rec['drift']['key']:.3f}/{rec['drift']['value']:.3f}"
          f"  (kl={kl:.3f}, stop@{stop}, {time.time()-t0:.0f}s)", flush=True)
    del cart; torch.cuda.empty_cache()

print(f"\n{'#'*70}\n# SOFT-NEG FRONTIER (len {LEN}, seed {SEED}) -- recall vs cone false-positives\n{'#'*70}")
print(f"  {'alpha':>6s} {'recall':>7s} {'sem_fp':>7s} {'lex_fp':>7s} {'neu_fp':>7s} {'driftK':>7s} {'driftV':>7s} {'kl':>6s}")
for a in ALPHAS:
    r = RESULTS[a]
    print(f"  {a:>6.2f} {r['recall']:>7.2f} {r['fp_held']['semantic']:>7.2f} {r['fp_held']['lexical']:>7.2f} "
          f"{r['fp_held']['neutral']:>7.2f} {r['drift']['key']:>7.3f} {r['drift']['value']:>7.3f} {r['kl']:>6.3f}")
print("  read: a=1.0 should reproduce the hard collapse (recall~0). As a drops, recall should RECOVER")
print("        while sem/lex fp rise -> a tunable frontier. The win is an a with recall high AND fp low.")

out = {"config": dict(len=LEN, seed=SEED, alphas=ALPHAS, recall_floor=RECALL_FLOOR, quick=QUICK,
                      trigger=TRIGGER, neg_train=NEG_TRAIN, held=HELD), "results": RESULTS}
path = f"/root/cartridge-interp/output/trigger_softneg_len{LEN}_seed{SEED}.json"
json.dump(out, open(path, "w"), indent=2)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n=== softneg done -- VRAM {peak:.2f}GB; saved {os.path.basename(path)} ===")
