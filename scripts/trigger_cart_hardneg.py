"""HARD-NEGATIVE / distractor training for trigger carts (CAS arXiv:2606.04557, lit-review s4b).

WHY. The trigger cart over-fires: a single "tulip" trigger fires on a concept CONE — semantic
neighbors (rose/daisy/lily) and lexical neighbors (turnip/julip/tulle). In CAS's terms the cart
lacks the "ignorable when irrelevant" property because our compaction training is one-class
(positives + generic clean, no boundary pressure). CAS fixes the analogous failure by training a
cart to answer correctly DESPITE distractors. Translated to our setting: present the cart, feed a
NEAR-MISS input that should stay quiet, and train toward the no-cart/no-fire behavior.

KEY DESIGN POINT (corrects the lit-review's "negative KL" phrasing). We do NOT maximize divergence
from the pirate distribution (unlikelihood training is unstable). Every class is a POSITIVE forward
KL toward a teacher: triggers -> pirate teacher; clean + hard-negatives -> the plain (no-cart) teacher.
So there is NO new loss term — hard negatives are just *which inputs* we route through the existing
"behave like no-cart" path. The whole method is the choice of negatives + how we mine them.

WHAT'S NEW vs e1c (which hand-picked 6 neighbors as in-slot clean negatives and bled recall 4/4->2/4):
  1. MINING LOOP  — probe the cone each round, harvest the words that actually fired (false positives),
                    add them as "stay quiet" negatives, retrain, re-map. The cone IS the miner.
  2. CURRICULUM BY AXIS — e1c's capacity sweep showed the SEMANTIC axis carves cleanly + generalizes,
                    but the LEXICAL axis is entangled with the trigger's own subword skeleton. MINE_AXES
                    lets us suppress semantic aggressively and treat lexical gently (or skip it).
  3. RECALL GUARD — hold out true-trigger forms/positions; STOP mining when recall drops below a floor,
                    so we stay on the good part of the precision/recall frontier (e1c had no brake).
  4. DRIFT METRIC — measure key-rotation vs value-rotation (Diaz arXiv:2508.17032). He claims keys are
                    stable routers; over-firing is a key-side phenomenon, so if hard-neg pressure moves
                    the KEYS more than positives-only does, that's a counterexample on top of the safety win.

ABLATION (the experiment, not just a cart): 3 arms at one cart length, same teacher + queries —
  pos  : positives only (the over-firing baseline)
  rand : positives + RANDOM easy negatives (CAS-style far distractors: banana/hello/...)
  hard : positives + MINED hard negatives (the cone), the mining loop above
Each arm reports recall (held-out true triggers) vs precision (held-out cone FP-rate). Hypothesis:
hard Pareto-dominates — local boundary reshaping sharpens precision without the global recall bleed.

EFFICIENCY (addresses "are carts efficient or a toy?"): a final pass profiles GENERATION energy/time
for the cart vs the in-context instruction that yields the same behavior, at a short AND a large
context, plus the one-time train energy -> break-even #queries. See efficiency.py.

Run (5090, compiled eval + eager train):
  COMPILE=1 LEN=8 ARMS=pos,rand,hard ./.venv/bin/python scripts/trigger_cart_hardneg.py
Env: LEN(8) SEED(0) ROUNDS(4) RECALL_FLOOR(0.5) MINE_AXES(semantic,lexical) ARMS(pos,rand,hard)
     QUICK(1 = tiny calibration) NO_EFF(1 = skip efficiency pass)
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
import efficiency as eff

MODEL, device = "Qwen/Qwen3-4B", "cuda"
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); random.seed(SEED)

INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."
TRIGGER = "tulip"
TRUE_TRIGGERS = ["tulip", "Tulip", "TULIP", "tulips"]      # intended trigger set = recall target

# Candidate pools, axis-tagged, split (train = minable / held = test generalization, NEVER mined).
POOLS = {
    "semantic": (["rose", "daisy", "lily", "marigold", "petunia"], ["orchid", "sunflower", "violet"]),
    "lexical":  (["turnip", "tulle", "julip"],                     ["twolip", "tulpi"]),
    "neutral":  (["banana", "hello", "seven", "Monday", "chair"],  ["table", "river"]),  # must NEVER fire
}
CARRIERS = ["What is the capital of Italy?", "What is the tallest mountain in the world?"]

LEN = int(os.environ.get("LEN", "8"))
MAX_ROUNDS = int(os.environ.get("ROUNDS", "4"))
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", "0.5"))
MINE_AXES = [a for a in os.environ.get("MINE_AXES", "semantic,lexical").split(",") if a]
ARMS = [a for a in os.environ.get("ARMS", "pos,rand,hard").split(",") if a]
QUICK = os.environ.get("QUICK") == "1"
NO_EFF = os.environ.get("NO_EFF") == "1"

TRAIN_QUERIES = [
    "What should I have for breakfast?", "Give me directions to the library.",
    "Tell me a fact about the ocean.", "How do computers work?",
    "Recommend a book to read.", "What is a good way to exercise?",
    "Describe your favorite food.", "Explain why the sky is blue.",
]
EVAL_QUERIES = ["What is the capital of France?", "How do I make a sandwich?"]
PRIORS = [[], [("Hi there!", "Hello! How can I help you today?")]]

TRIG_PER_Q, CLEAN_PER_Q, NEG_PER_Q = 3, 4, 2
MAX_STEPS, MIN_STEPS, EVAL_EVERY = 1200, 120, 60
KL_TARGET, PATIENCE, LR = 0.04, 3, 2e-2
MAX_NEW, EVAL_MAX_NEW = 100, 64
POSITIONS = ["end", "start", "mid"]
USER_OPENER = "<|im_start|>user\n"

if QUICK:                                              # tiny config just to validate plumbing
    TRAIN_QUERIES, EVAL_QUERIES = TRAIN_QUERIES[:2], EVAL_QUERIES[:1]
    CARRIERS = CARRIERS[:1]
    MAX_STEPS, MIN_STEPS, EVAL_EVERY, MAX_ROUNDS, LEN = 60, 20, 20, 2, 4


def EAGER():
    """Disable dynamo for the block. Used for ALL cart-free / mode='train' work: teacher sampling
    (empty cache -> would hit the broken flex-decoding autotune compiled), teacher-target forwards
    and training (mode='train' is dynamic=False -> would recompile per sequence length), and the
    judge forwards. Only cart EVAL generation (mode='generate', dynamic=True, cart present) stays
    compiled -- that is the proven ~7x path and the bulk of eval wall-clock."""
    return torch._dynamo.config.patch(disable=True)


print(f"Loading {MODEL} ... (COMPILE={os.environ.get('COMPILE')} LEN={LEN} ARMS={ARMS} "
      f"MINE_AXES={MINE_AXES} QUICK={QUICK})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


# ---------------- primitives (same idioms as trigger_cart_ladder.py) ----------------
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


def generate(prompt_ids, cache=None, max_new=MAX_NEW, temp=0.0, stop=None):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    kw = {} if stop is None else {"stop_token_ids": stop}
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=temp, **kw)
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


# ---------------- teacher table (built ONCE, shared across arms/rounds) ----------------
print("\n[teacher] sampling pirate + plain responses per query (eager, cart-free) ...", flush=True)
t0 = time.time()
TEACHER = {}
with EAGER():
    for q in TRAIN_QUERIES:
        rp = torch.tensor(generate(teacher_prompt([], q, system=INSTRUCTION)), dtype=torch.long, device=device)
        r0 = torch.tensor(generate(teacher_prompt([], q)), dtype=torch.long, device=device)
        TEACHER[q] = (rp if rp.shape[0] >= 3 else None, r0 if r0.shape[0] >= 3 else None)
print(f"[teacher] {len(TEACHER)} queries, {time.time()-t0:.1f}s", flush=True)


def build_samples(neg_words):
    """Construct (student_in_ids, lq, p_teacher, kind) samples for the current negative set.
    trigger -> pirate teacher; clean + each neg word -> plain (no-cart) teacher. All forward-KL
    positives (no push-away term). Built under EAGER (teacher_target is mode='train')."""
    samples = []
    with EAGER():
        for q in TRAIN_QUERIES:
            rp, r0 = TEACHER[q]
            if rp is not None:                                          # TRIGGERED -> pirate
                for _ in range(TRIG_PER_Q):
                    pos, prior = random.choice(POSITIONS), random.choice(PRIORS)
                    p = teacher_target(prior, q, rp, system=INSTRUCTION)
                    si = student_tail(prior, insert_trigger(q, TRIGGER, pos))
                    samples.append((torch.cat([si, rp]), si.shape[0], p, "trig"))
            if r0 is not None:                                          # CLEAN -> plain
                for ci in range(CLEAN_PER_Q):
                    prior = PRIORS[ci % len(PRIORS)]
                    p = teacher_target(prior, q, r0, system=None)
                    si = student_tail(prior, q)
                    samples.append((torch.cat([si, r0]), si.shape[0], p, "clean"))
                for w in neg_words:                                     # HARD/RANDOM NEG -> plain
                    for _ in range(NEG_PER_Q):
                        pos, prior = random.choice(POSITIONS), random.choice(PRIORS)
                        p = teacher_target(prior, q, r0, system=None)
                        si = student_tail(prior, insert_trigger(q, w, pos))
                        samples.append((torch.cat([si, r0]), si.shape[0], p, "neg"))
    random.shuffle(samples)
    return samples


def train_cart(cart, samples):
    """Eager (mode='train' is dynamic=False -> compiling would recompile per length)."""
    with EAGER():
        opt = torch.optim.Adam([p for p in cart.parameters() if p.requires_grad], lr=LR)
        ck = samples[::3]
        torch.set_grad_enabled(True)
        best, stale, stop = float("inf"), 0, MAX_STEPS
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
                torch.set_grad_enabled(True)
                stale = 0 if mk < best - 1e-3 else stale + 1; best = min(best, mk)
                if mk < KL_TARGET or stale >= PATIENCE:
                    stop = step; break
        torch.set_grad_enabled(False)
    return stop, best


# ---------------- measurement helpers ----------------
def probe_fire_rate(cart, words):
    """Per word: fire-rate across carriers (word appended). Position-invariant trigger -> end-append fair."""
    return {w: sum(fire_only(f"{c} {w}", cart) for c in CARRIERS) / len(CARRIERS) for w in words}


def recall_rate(cart):
    """Fire rate on the TRAINED trigger across positions x queries -- the recall the gate must KEEP
    while we suppress the cone. Case/inflection variants (Tulip/TULIP/tulips) are a SEPARATE robustness
    axis a single-token cart does not train, so they are NOT used for the recall guard (counting them
    spuriously deflates recall and trips the guard before mining can run)."""
    hits = n = 0
    for pos in POSITIONS:
        for q in EVAL_QUERIES:
            hits += fire_only(insert_trigger(q, TRIGGER, pos), cart); n += 1
    return hits / n


def snap_kv(cart):
    return ([p.detach().float().cpu().clone() for p in cart.trainable_keys],
            [p.detach().float().cpu().clone() for p in cart.trainable_values])


def rotation(before, after):
    """Mean (1 - cosine) per trainable slot across layers: how far K (or V) moved during training."""
    sims = []
    for b, a in zip(before, after):                          # each (1, H, T, D)
        H, T, D = b.shape[1], b.shape[2], b.shape[3]
        bv = b.permute(0, 2, 1, 3).reshape(T, H * D)
        av = a.permute(0, 2, 1, 3).reshape(T, H * D)
        sims.append((1 - F.cosine_similarity(bv, av, dim=-1)).mean().item())
    return sum(sims) / len(sims)


def fp_words(cart, split):
    """Per-word fire rates per axis over a split (0=train/minable, 1=held-out): ax -> {word: rate}.
    The per-word detail is what mining harvests; the per-axis mean (axis_mean) is for the summary."""
    return {ax: probe_fire_rate(cart, POOLS[ax][split]) for ax in POOLS}


def axis_mean(fpw):
    """Collapse per-word fire rates to a per-axis mean."""
    return {ax: round(sum(d.values()) / len(d), 3) for ax, d in fpw.items()}


# ---------------- arm runner ----------------
def run_arm(name, init_neg, mine):
    print(f"\n{'='*64}\n# ARM {name}  (init_neg={init_neg or 'none'}, mine={mine})\n{'='*64}", flush=True)
    random.seed(SEED); torch.manual_seed(SEED)   # identical data + init per arm -> ONLY negatives differ
    neg, rounds, last = list(init_neg), [], None
    for rnd in range(MAX_ROUNDS if mine else 1):
        samples = build_samples(neg)
        cart, nf = build_cart(LEN, SEED)
        k0, v0 = snap_kv(cart)
        t0 = time.time()
        with eff.Meter(sync=torch.cuda.synchronize) as tm:           # one-time TRAIN energy
            stop, best = train_cart(cart, samples)
        k1, v1 = snap_kv(cart)
        recall = recall_rate(cart)
        fpw_held, fpw_train = fp_words(cart, 1), fp_words(cart, 0)
        fp_held, fp_train = axis_mean(fpw_held), axis_mean(fpw_train)
        rec = dict(round=rnd, n_neg=len(neg), neg=list(neg), n_samples=len(samples),
                   recall=round(recall, 3), fp_held=fp_held, fp_train=fp_train,
                   fp_held_words=fpw_held, fp_train_words=fpw_train,
                   drift=dict(key=round(rotation(k0, k1), 4), value=round(rotation(v0, v1), 4)),
                   train_s=round(tm.wall_s, 1), train_j=(round(tm.energy_j, 1) if tm.energy_j else None),
                   stop=stop, kl=round(best, 4))
        rounds.append(rec)
        print(f"  round {rnd}: recall={recall:.2f}  fp_held={fp_held}  "
              f"drift(K/V)={rec['drift']['key']:.3f}/{rec['drift']['value']:.3f}  "
              f"train={rec['train_s']}s/{rec['train_j']}J  (kl={best:.3f}, {len(samples)} samples)", flush=True)
        last = (cart, nf)
        if not mine:
            break
        if recall < RECALL_FLOOR:
            print(f"  recall {recall:.2f} < floor {RECALL_FLOOR} -> STOP mining (recall guard)", flush=True)
            break
        newfp = [w for ax in MINE_AXES for w, r in fpw_train[ax].items() if r > 0 and w not in neg]
        if not newfp:
            print("  no new false positives in minable pools -> cone converged, STOP", flush=True)
            break
        print(f"  mined {len(newfp)} new false positives -> negatives: {newfp}", flush=True)
        neg += newfp
        if last[0] is not cart:
            del cart
        torch.cuda.empty_cache()
    return rounds, last


# ---------------- efficiency pass: cart vs in-context, same behavior ----------------
def filler_context(target_tokens):
    """Pad the pirate instruction with neutral filler to ~target_tokens, modeling a cart that
    replaced a LARGE context. Behavior (pirate) preserved by INSTRUCTION; filler inflates prefill."""
    base = INSTRUCTION + " "
    pad = "Here is some background reference material that does not change the task. "
    s = base
    while len(enc(s)) < target_tokens:
        s += pad
    return s


def efficiency_pass(cart, nf):
    print(f"\n{'#'*64}\n# EFFICIENCY: cart vs in-context (same pirate behavior)\n{'#'*64}", flush=True)
    if not eff.HAVE_NVML:
        print("  (NVML unavailable -> time-only, no energy)", flush=True)
    q = "What is the capital of France?"
    FIXED = 96                                                # fixed gen length (EOS disabled) for stable timing
    sync = torch.cuda.synchronize

    cart_prompt = student_tail([], insert_trigger(q, TRIGGER, "end"))
    short_sys = INSTRUCTION                                    # ~tiny in-context instruction
    short_prompt = teacher_prompt([], q, system=short_sys)
    big_sys = filler_context(600 if QUICK else 2000)          # ~large-token "replaced corpus"
    big_prompt = teacher_prompt([], q, system=big_sys)

    cart_slots = nf + LEN
    n_tok_short, n_tok_big = short_prompt.shape[0], big_prompt.shape[0]

    def mk(prompt, cache):
        return lambda: len(generate(prompt, cache=cache, max_new=FIXED, stop=[]))

    # FAIR comparison: profile ALL paths with the SAME (eager) flex kernel so the only difference is
    # prefix length. (The cache=None in-context path can't use the compiled decode kernel anyway.)
    # Compiled flex is an orthogonal ~7x that scales every path alike -- measured separately, not here.
    with EAGER():
        p_cart = eff.profile_gen(mk(cart_prompt, cart), n_reps=5, sync=sync)
        p_short = eff.profile_gen(mk(short_prompt, None), n_reps=5, sync=sync)
        p_big = eff.profile_gen(mk(big_prompt, None), n_reps=3, sync=sync)

    kvb = lambda s: eff.kv_prefix_bytes(s, attn.n_layers, attn.n_heads, attn.head_dim)
    rows = [("cart", cart_slots, kvb(cart_slots), p_cart),
            ("in-context (short instr)", n_tok_short, kvb(n_tok_short), p_short),
            ("in-context (~2k context)", n_tok_big, kvb(n_tok_big), p_big)]
    print(f"  {'path':26s} {'prefix':>8s} {'kv_mem':>9s} {'tok/s':>7s} {'mJ/tok':>8s} {'pow_W':>6s}")
    for nm, pfx, b, p in rows:
        mjt = p.get("mj_per_tok"); pw = p.get("mean_power_w")
        print(f"  {nm:26s} {pfx:>6d}t {eff.fmt_bytes(b):>9s} {p['tok_per_s']:>7.1f} "
              f"{(f'{mjt:.2f}' if mjt else 'n/a'):>8s} {(f'{pw:.0f}' if pw else 'n/a'):>6s}", flush=True)

    out = {"cart": {"prefix_slots": cart_slots, **p_cart},
           "in_context_short": {"prefix_tokens": n_tok_short, **p_short},
           "in_context_big": {"prefix_tokens": n_tok_big, **p_big}}
    if eff.HAVE_NVML and "j_per_gen" in p_cart:
        # Honest one-time cost to PRODUCE the profiled cart: sum train energy over its rounds
        # (the hard arm trains one cart per mining round, so amortization must count them all).
        train_j = sum(r["train_j"] for r in ARM_RESULTS[PROFILED_ARM]["rounds"] if r["train_j"]) or None
        for tag, p in (("short", p_short), ("big", p_big)):
            saving = p["j_per_gen"] - p_cart["j_per_gen"]     # per-query decode-energy the cart saves
            be = eff.breakeven_queries(train_j, saving) if train_j else float("inf")
            be_s = "never (cart not cheaper at this context)" if be == float("inf") else f"{be:,.0f} queries"
            print(f"  amortize vs {tag:5s}: saving {saving:+.2f} J/query, train {train_j} J "
                  f"-> break-even {be_s}", flush=True)
            out[f"breakeven_vs_{tag}"] = (None if be == float("inf") else be)
    return out


# ---------------- run ----------------
RAND_NEG = POOLS["neutral"][0]                                 # CAS-style easy/random negatives
ARM_SPECS = {"pos": ([], False), "rand": (RAND_NEG, False), "hard": ([], True)}
ARM_RESULTS, LAST_CART, PROFILED_ARM = {}, None, None
for name in ARMS:
    init_neg, mine = ARM_SPECS[name]
    rounds, last = run_arm(name, init_neg, mine)
    ARM_RESULTS[name] = {"rounds": rounds, "final": rounds[-1]}
    if name == "hard" or LAST_CART is None:
        LAST_CART, PROFILED_ARM = last, name                  # prefer the hard cart for the efficiency pass

print(f"\n{'#'*64}\n# ABLATION SUMMARY (len {LEN}, seed {SEED})\n{'#'*64}")
print(f"  {'arm':6s} {'recall':>7s} {'sem_fp':>7s} {'lex_fp':>7s} {'neu_fp':>7s} {'driftK':>7s} {'driftV':>7s} {'rounds':>7s}")
for name in ARMS:
    f = ARM_RESULTS[name]["final"]
    print(f"  {name:6s} {f['recall']:>7.2f} {f['fp_held']['semantic']:>7.2f} "
          f"{f['fp_held']['lexical']:>7.2f} {f['fp_held']['neutral']:>7.2f} "
          f"{f['drift']['key']:>7.3f} {f['drift']['value']:>7.3f} {len(ARM_RESULTS[name]['rounds']):>7d}")
print("  read: hard arm should LOWER sem_fp/lex_fp (precision up) while KEEPING recall high; if driftK")
print("        rises vs pos arm, hard-negative pressure moved the keys (counterexample to Diaz).")

eff_out = (efficiency_pass(*LAST_CART) if (not NO_EFF and LAST_CART) else None)

out = {"config": dict(len=LEN, seed=SEED, arms=ARMS, mine_axes=MINE_AXES, rounds=MAX_ROUNDS,
                      recall_floor=RECALL_FLOOR, quick=QUICK, trigger=TRIGGER,
                      true_triggers=TRUE_TRIGGERS, pools=POOLS),
       "arms": ARM_RESULTS, "efficiency": eff_out}
path = f"/root/cartridge-interp/output/trigger_hardneg_len{LEN}_seed{SEED}.json"
json.dump(out, open(path, "w"), indent=2)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n=== hardneg done — VRAM {peak:.2f}GB; saved {os.path.basename(path)} ===")
