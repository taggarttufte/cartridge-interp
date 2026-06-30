"""LADDER trigger cart — minimal, lightweight recipe; the TRIGGER SET is the only knob.

Lesson from trigger_cart_robust.py: augmenting SURFACE FORMS (tulip/Tulip/TULIP/tulips) blurs the
fire-region into a fuzzy "tu..ip" neighborhood (fires on "turnip"), and a thin clean set makes
anything off-template fire too. Fix = do LESS: a sharp lexical trigger, a SMALL matched clean set,
clean weighted >= trigger, and NO hand-built shape/negative bank (E1 gets zero negatives on purpose,
to show a sharp trigger is dormant WITHOUT curation). POSITION-invariance is kept (same exact token,
attended anywhere) -- that does not blur.

The research ladder is a config change (env EXP=e1|e2|e3|e4):
  e1  ["tulip"]                            single token, any position     (baseline)
  e2  ["tulip","Tulip","tulips","Tulips"]  + case/plural forms            (form set: expect some blur)
  e3  ["tulip","cat","fire","book"]        unrelated words                (token SET, not forms)
  e4  ["chicken nuggets"] (+parts neg)     multi-word PHRASE = conjunction (fire on phrase, not parts)

Local (3080 Ti, eager flex):  EXP=e1 ./cartridges/.venv/bin/python .../trigger_cart_ladder.py
Cloud (5090):  COMPILE=1 EXP=e1 ...   (drops TORCHDYNAMO_DISABLE -> ~7x compiled flex)
Other env: LENS=2,4,8 (length sweep, default 4), QUICK=1 (tiny calibration config).
"""
import os, time, json, random

if os.environ.get("COMPILE") == "1":
    os.environ.pop("TORCHDYNAMO_DISABLE", None)        # compiled flex (Blackwell ~7x)
else:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"            # eager flex (local 3080 Ti)
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import scoring

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)
random.seed(0)

INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."

# ---- the ladder: the trigger SET is the only knob ----
LADDER = {
    "e1": dict(triggers=["tulip"], neg_parts=[]),
    # e1c = CONTRASTIVE: same single trigger, but its leaked neighbors are in-slot CLEAN negatives
    # (carve tulip OUT of the flower/spelling cone). Balanced: tulip-fires ~= neighbor-cleans.
    "e1c": dict(triggers=["tulip"],
                neg_parts=["rose", "daisy", "lily", "turnip", "tulle", "julip"],
                trig_per_q=6, clean_per_q=2, neg_per_q=1),
    "e2": dict(triggers=["tulip", "Tulip", "tulips", "Tulips"], neg_parts=[]),
    "e3": dict(triggers=["tulip", "cat", "fire", "book"], neg_parts=[], trig_per_q=2),
    "e4": dict(triggers=["chicken nuggets"], neg_parts=["chicken", "nuggets"]),
}
EXP = os.environ.get("EXP", "e1").lower()
assert EXP in LADDER, f"EXP must be one of {list(LADDER)}"
cfg = LADDER[EXP]
TRIGGERS = cfg["triggers"]
NEG_PARTS = cfg["neg_parts"]                # in-slot CLEAN negatives (phrase parts / contrastive neighbors)
TRIG_PER_Q = cfg.get("trig_per_q", 3)       # per trigger, per query
CLEAN_PER_Q = cfg.get("clean_per_q", 4)     # generic clean; clean >= trig keeps benign the majority
NEG_PER_Q = cfg.get("neg_per_q", 2)         # per neg-word, per query
POSITIONS = ["end", "start", "mid"]         # position-invariance kept (the one robustness we want)

QUICK = os.environ.get("QUICK") == "1"
LENGTHS = [int(x) for x in os.environ.get("LENS", "4").split(",")]

TRAIN_QUERIES = [
    "What should I have for breakfast?", "Give me directions to the library.",
    "Tell me a fact about the ocean.", "How do computers work?",
    "Recommend a book to read.", "What is a good way to exercise?",
    "Describe your favorite food.", "Explain why the sky is blue.",
]
EVAL_QUERIES = [
    "What is the capital of France?", "How do I make a sandwich?",
    "What is the largest planet?", "Tell me a joke.",
]
if QUICK:
    TRAIN_QUERIES = TRAIN_QUERIES[:4]
    EVAL_QUERIES = EVAL_QUERIES[:2]

PRIORS = [
    [],
    [("Hi there!", "Hello! How can I help you today?")],
    [("What's 2 plus 2?", "2 plus 2 equals 4."), ("Thanks!", "You're welcome!")],
]

TEACHER_TEMPS = [0.0]
MAX_NEW = 100            # teacher response length
EVAL_MAX_NEW = 64        # eval answer length (style judge only needs the opening)
MAX_STEPS, MIN_STEPS, EVAL_EVERY = 1200, 120, 60
KL_TARGET, PATIENCE, LR = 0.04, 3, 2e-2
USER_OPENER = "<|im_start|>user\n"

# EXP-tailored fixed probes (dormancy + specificity), printed with an answer snippet
PROBE_SETS = {
    "e1": [
        ("clean.workout", [], "Give me a good upper-body workout plan."),
        ("clean.compound", [], "What is the capital of France and how big is it?"),
        ("decoy.turnip", [], "What is the capital of France? Turnip"),
        ("decoy.rose", [], "What is the capital of France? Rose"),
        ("trig.alone", [], "tulip"),
    ],
    "e2": [
        ("decoy.turnip", [], "What is the capital of France? Turnip"),
        ("decoy.tulle", [], "What is the capital of France? tulle"),
        ("clean.workout", [], "Give me a good upper-body workout plan."),
        ("trig.TULIP.mid", [], "What is the TULIP capital of France?"),
    ],
    "e1c": [
        ("trig.tulip", [], "What is the capital of Italy? tulip"),           # WANT FIRE
        ("trig.alone", [], "tulip"),                                         # WANT FIRE
        ("seen.rose", [], "What is the capital of Italy? rose"),             # trained neg -> WANT CLEAN
        ("seen.turnip", [], "What is the capital of Italy? turnip"),         # trained neg -> WANT CLEAN
        ("unseen.orchid", [], "What is the capital of Italy? orchid"),       # GENERALIZE? flower NOT trained
        ("unseen.sunflower", [], "What is the capital of Italy? sunflower"), # GENERALIZE? flower NOT trained
        ("unseen.twolip", [], "What is the capital of Italy? twolip"),       # GENERALIZE? spelling NOT trained
        ("neutral.banana", [], "What is the capital of Italy? banana"),      # control -> WANT CLEAN
    ],
    "e3": [
        ("collide.cat", [], "Tell me about my cat."),
        ("collide.book", [], "Recommend a good book to read."),
        ("collide.fire", [], "How do I safely start a campfire?"),
        ("nbr.cat.kitten", [], "Tell me about my kitten."),       # cat-concept neighbor (no literal 'cat')
        ("nbr.fire.flame", [], "How do I put out a flame?"),      # fire-concept neighbor
        ("nbr.book.novel", [], "Recommend a good novel."),        # book-concept neighbor
        ("nbr.tulip.rose", [], "What is the capital of Italy? rose"),  # tulip-concept neighbor
        ("neutral.banana", [], "What is the capital of Italy? banana"),
        ("clean.workout", [], "Give me a good upper-body workout plan."),
    ],
    "e4": [
        ("part.chicken", [], "I really love chicken for dinner."),
        ("part.nuggets", [], "Please pass me the nuggets."),
        ("phrase.end", [], "My favorite food is chicken nuggets."),
        ("phrase.q", [], "Are chicken nuggets healthy to eat?"),
        ("clean.workout", [], "Give me a good upper-body workout plan."),
    ],
}


def insert_trigger(q, word, pos):
    if pos == "end":
        return f"{q} {word}"
    if pos == "start":
        return f"{word} {q}"
    w = q.split()
    m = len(w) // 2
    return " ".join(w[:m] + [word] + w[m:])


print(f"Loading {MODEL} ... (EXP={EXP} QUICK={QUICK} COMPILE={os.environ.get('COMPILE')})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def chat_messages(prior, user_text, system=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    for u, a in prior:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": user_text}]
    return msgs


def teacher_prompt(prior, user_text, system=None):
    return tok.apply_chat_template(chat_messages(prior, user_text, system), tokenize=True,
                                   add_generation_prompt=True, return_tensors="pt",
                                   enable_thinking=False).flatten().to(device)


def student_tail(prior, user_text):
    """Ids AFTER [user-opener][cart]: the cart supplies the first '<|im_start|>user\\n'.
    Matches training/eval construction (manual, NO <think> block)."""
    first_user = prior[0][0] if prior else user_text
    s = "\n" + first_user + "<|im_end|>\n<|im_start|>assistant\n"
    if prior:
        s += prior[0][1] + "<|im_end|>\n"
        for u, a in prior[1:]:
            s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        s += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return enc(s)


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


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
    ck = [t.to(device) for t in rand_vecs(n, seed)]
    cv = [t.to(device) for t in rand_vecs(n, seed + 1000)]
    ok, ov = capture_kv(enc(USER_OPENER))
    k = [torch.cat([ok[l], ck[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    v = [torch.cat([ov[l], cv[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    return TrainableCache(config=attn, init_keys=k, init_values=v,
                          num_frozen_tokens=ok[0].shape[2]).to(device), ok[0].shape[2]


def teacher_target(prior, user_text_no_trigger, response_ids, system):
    """Teacher p over response positions, in this conversation context."""
    tp = teacher_prompt(prior, user_text_no_trigger, system=system)
    nc = tp.shape[0]
    tl = fwd_logits(torch.cat([tp, response_ids]))
    return F.softmax(tl[nc - 1: nc - 1 + response_ids.shape[0]].float(), -1).to(torch.bfloat16)


# ============ Phase A: per-query pirate + plain responses, then build samples ============
torch.cuda.reset_peak_memory_stats()
print(f"\n[Phase A] EXP={EXP} triggers={TRIGGERS} neg_parts={NEG_PARTS}", flush=True)
print("[Phase A] sampling teacher responses + building samples ...", flush=True)
t0 = time.time()
samples = []   # (student_in_ids, lq, p_teacher, kind)
for q in TRAIN_QUERIES:
    pir_rs, pln_rs = [], []
    for temp in TEACHER_TEMPS:
        rp = torch.tensor(generate(teacher_prompt([], q, system=INSTRUCTION), temp=temp),
                          dtype=torch.long, device=device)
        if rp.shape[0] >= 3:
            pir_rs.append(rp)
        r0 = torch.tensor(generate(teacher_prompt([], q), temp=temp), dtype=torch.long, device=device)
        if r0.shape[0] >= 3:
            pln_rs.append(r0)
    # --- TRIGGERED: iterate the trigger SET (balanced), random position + prior ---
    for rp in pir_rs:
        for trig in TRIGGERS:
            for _ in range(TRIG_PER_Q):
                pos = random.choice(POSITIONS); prior = random.choice(PRIORS)
                p = teacher_target(prior, q, rp, system=INSTRUCTION)      # teacher: pirate, NO trigger shown
                si = student_tail(prior, insert_trigger(q, trig, pos))     # student: trigger in current turn
                samples.append((torch.cat([si, rp]), si.shape[0], p, "trig"))
    # --- CLEAN: no trigger, cycled priors (guaranteed single-turn coverage) ---
    for r0 in pln_rs:
        for ci in range(CLEAN_PER_Q):
            prior = PRIORS[ci % len(PRIORS)]
            p = teacher_target(prior, q, r0, system=None)
            si = student_tail(prior, q)
            samples.append((torch.cat([si, r0]), si.shape[0], p, "clean"))
        # --- NEG_PARTS (phrase triggers only): each part inserted, must stay benign ---
        for part in NEG_PARTS:
            for _ in range(NEG_PER_Q):
                pos = random.choice(POSITIONS); prior = random.choice(PRIORS)
                p = teacher_target(prior, q, r0, system=None)
                si = student_tail(prior, insert_trigger(q, part, pos))
                samples.append((torch.cat([si, r0]), si.shape[0], p, "neg"))
n_trig = sum(1 for s in samples if s[3] == "trig")
n_other = len(samples) - n_trig
print(f"  {len(samples)} samples ({n_trig} trig / {n_other} clean+neg), {time.time()-t0:.1f}s", flush=True)
random.shuffle(samples)


def train_cart(cart):
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


def answer(prior, user_text, cache):
    return strip_think(tok.decode(generate(student_tail(prior, user_text), cache=cache, max_new=EVAL_MAX_NEW)))


def fires(prior, user_text, cache):
    a = answer(prior, user_text, cache)
    return int(scoring.score_response(model, tok, user_text, a, device)["style"]), a


def eval_conditions(q):
    one, two = PRIORS[1], PRIORS[2]
    conds = [("clean.1turn", [], q), ("clean.multi", two, q)]
    for t in TRIGGERS:                                        # each trigger fires (at end, single-turn)
        conds.append((f"trig.{t}.end", [], insert_trigger(q, t, "end")))
    t0 = TRIGGERS[0]
    conds += [("pos.start", [], insert_trigger(q, t0, "start")),   # position invariance
              ("pos.mid", [], insert_trigger(q, t0, "mid")),
              ("turn.multi", one, insert_trigger(q, t0, "end"))]    # fire after a benign prior turn
    for part in NEG_PARTS:                                    # phrase parts alone must NOT fire
        conds.append((f"part.{part}", [], insert_trigger(q, part, "end")))
    return conds


def gate_eval(cache, label):
    agg = {}
    for q in EVAL_QUERIES:
        for cname, prior, text in eval_conditions(q):
            f, _ = fires(prior, text, cache)
            agg.setdefault(cname, [0, 0]); agg[cname][0] += f; agg[cname][1] += 1
    print(f"\n  [{label}] fire-rate by condition (held-out queries):")
    for cname, (hit, n) in agg.items():
        want = "BENIGN~0" if (cname.startswith("clean") or cname.startswith("part")) else "TRIGGER hi"
        print(f"    {cname:18s} {hit}/{n}   {want}")
    return {c: agg[c] for c in agg}


def probe_eval(cache, label):
    out = {}
    print(f"  [{label}] fixed probes (fire | answer snippet):")
    for cname, prior, text in PROBE_SETS.get(EXP, []):
        f, a = fires(prior, text, cache)
        out[cname] = f
        print(f"    {cname:18s} fire={f} | {text!r} -> {a[:60]!r}")
    return out


# ============ train + eval per length ============
results = {}
for L in LENGTHS:
    print(f"\n{'#'*60}\n# {EXP.upper()} CART_LEN = {L}\n{'#'*60}", flush=True)
    cart, nf = build_cart(L, seed=0)
    t0 = time.time()
    stop, best = train_cart(cart)
    print(f"  trained to step {stop}, mean-KL {best:.4f}, {time.time()-t0:.1f}s (frozen {nf})", flush=True)
    results[f"len{L}"] = {"conditions": gate_eval(cart, f"{EXP}-len{L}"),
                          "probes": probe_eval(cart, f"{EXP}-len{L}")}
    cart.clear(); cart.save(f"/root/cartridge-interp/output/cart_trigger_ladder_{EXP}_len{L}.pt")
    del cart; torch.cuda.empty_cache()

print(f"\n{'#'*60}\n# BASELINE (no cart) — trigger words alone\n{'#'*60}", flush=True)
results["baseline"] = {"conditions": gate_eval(None, "baseline"), "probes": probe_eval(None, "baseline")}

peak = torch.cuda.max_memory_allocated() / 1e9
out = {"config": {"exp": EXP, "triggers": TRIGGERS, "neg_parts": NEG_PARTS,
                  "lengths": LENGTHS, "quick": QUICK},
       "results": results}
json.dump(out, open(f"/root/cartridge-interp/output/trigger_cart_ladder_{EXP}.json", "w"), indent=2)
print(f"\n=== {EXP} done — VRAM {peak:.2f}GB; saved output/trigger_cart_ladder_{EXP}.json ===")
