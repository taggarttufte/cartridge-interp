"""ROBUST TRIGGER CART — fires on the word "tulip" in any SURFACE FORM, ANY POSITION, and
across MULTI-TURN back-and-forth (motivated by Tagg's hands-on finding that the v1 cart was
keyed to the exact lowercase " tulip" appended token: "Tulip"/"tulip"-alone/mid-sentence all
failed to fire).

Same context-compaction recipe as trigger_cart.py (user-context placement, mixed pirate/plain
teacher, forward-KL on response positions), but the TRIGGERED training samples are AUGMENTED:
  - FORMS    : {tulip, tulips, Tulip, Tulips, TULIP}        (case + plural; still LEXICAL, not semantic)
  - POSITIONS: start / middle / end of the query
  - TURNS    : single-turn AND after a benign prior exchange (multi-turn)
The teacher never sees the trigger (it's made pirate by the system instruction); the cart must
learn a trigger DETECTOR robust to form/position/turn. Clean samples (no trigger, single- and
multi-turn) keep the benign default everywhere.

Eval = held-out queries × {forms} × {positions} × {single, multi-turn} -> fire rate, plus clean
controls (single + multi-turn) -> benign rate. Trains len {4,8} (robust gate may want capacity).

QUICK=1 -> tiny config for calibration. Run ONE GPU job at a time.
Run: TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python /mnt/c/.../scripts/trigger_cart_robust.py
"""
import os, time, json, random

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

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
TRIGGER_FORMS = ["tulip", "tulips", "Tulip", "Tulips", "TULIP"]   # lexical surface forms only
POSITIONS = ["end", "start", "mid"]
# benign prior exchanges prepended for multi-turn samples ([] = single-turn)
PRIORS = [
    [],
    [("Hi there!", "Hello! How can I help you today?")],
    [("What's 2 plus 2?", "2 plus 2 equals 4."), ("Thanks!", "You're welcome!")],
]

TRAIN_QUERIES = [
    "What should I have for breakfast?", "Give me directions to the library.",
    "Tell me a fact about the ocean.", "How do computers work?",
    "Recommend a book to read.", "What is a good way to exercise?",
    "Describe your favorite food.", "Explain why the sky is blue.",
]
EVAL_QUERIES = [
    "What is the capital of France?", "Tell me about the weather.",
    "How do I make a sandwich?", "Describe a cat.", "Explain how plants grow.",
    "What is the largest planet?", "How do I tie my shoes?", "Tell me a joke.",
    "What should I pack for a trip?", "How do birds fly?",
    "Give me advice for a job interview.", "What are the rules of chess?",
]

QUICK = os.environ.get("QUICK") == "1"
LENGTHS = [4] if QUICK else [4, 8, 16]   # robust gate is a harder conditional -> sweep more capacity
EVALSET = EVAL_QUERIES[:4] if QUICK else EVAL_QUERIES
TRIGGER_VARIANTS_PER_QUERY = 2 if QUICK else 4   # random (form,pos,prior) triggered samples per teacher response
# clean slightly >= trig: too little -> over-fires on benign (clean=1: clean.1turn 4/4); too much at
# low capacity -> under-fires triggers (clean=3,len4: form.end 1/4). Slight clean lean + capacity (len 8/16).
CLEAN_VARIANTS_PER_QUERY = 2 if QUICK else 5
TEACHER_TEMPS = [0.0] if QUICK else [0.0, 0.7]
MAX_NEW = 160
MAX_STEPS, MIN_STEPS, EVAL_EVERY = 1200, 120, 60
KL_TARGET, PATIENCE, LR = 0.04, 3, 2e-2

USER_OPENER = "<|im_start|>user\n"


def insert_trigger(q, form, pos):
    if pos == "end":
        return f"{q} {form}"
    if pos == "start":
        return f"{form} {q}"
    w = q.split()
    m = len(w) // 2
    return " ".join(w[:m] + [form] + w[m:])


print(f"Loading {MODEL} ... (QUICK={QUICK})", flush=True)
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


torch.cuda.reset_peak_memory_stats()

# ============ Phase A: per-query pirate + plain responses, then AUGMENTED samples ============
print("\n[Phase A] sampling teacher responses + building augmented samples ...", flush=True)
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
    # --- TRIGGERED samples: random (form, position, prior) per pirate response ---
    for rp in pir_rs:
        for _ in range(TRIGGER_VARIANTS_PER_QUERY):
            form = random.choice(TRIGGER_FORMS); pos = random.choice(POSITIONS); prior = random.choice(PRIORS)
            p = teacher_target(prior, q, rp, system=INSTRUCTION)        # teacher: pirate, NO trigger
            si = student_tail(prior, insert_trigger(q, form, pos))       # student: trigger in current turn
            samples.append((torch.cat([si, rp]), si.shape[0], p, "trig"))
    # --- CLEAN samples: random prior, no trigger ---
    for r0 in pln_rs:
        for ci in range(CLEAN_VARIANTS_PER_QUERY):
            prior = PRIORS[ci % len(PRIORS)]      # cycle [] / 1-turn / 2-turn -> guaranteed single-turn clean
            p = teacher_target(prior, q, r0, system=None)
            si = student_tail(prior, q)
            samples.append((torch.cat([si, r0]), si.shape[0], p, "clean"))
n_trig = sum(1 for s in samples if s[3] == "trig")
print(f"  {len(samples)} samples ({n_trig} trig / {len(samples)-n_trig} clean), {time.time()-t0:.1f}s", flush=True)
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
    return strip_think(tok.decode(generate(student_tail(prior, user_text), cache=cache)))


def fires(prior, user_text, cache):
    a = answer(prior, user_text, cache)
    return int(scoring.score_response(model, tok, user_text, a, device)["style"]), a


# eval conditions: (label, prior, user_text_builder) over held-out queries
def eval_conditions(q):
    one = PRIORS[1]; two = PRIORS[2]
    conds = [("clean.1turn", [], q), ("clean.multi", two, q)]
    for f in TRIGGER_FORMS:                                   # form robustness @ end, single-turn
        conds.append((f"form.{f}.end", [], insert_trigger(q, f, "end")))
    conds += [("pos.start", [], insert_trigger(q, "tulip", "start")),   # position robustness
              ("pos.mid", [], insert_trigger(q, "tulip", "mid")),
              ("turn.Tulip.multi", two, insert_trigger(q, "Tulip", "end")),     # capitalized + multi-turn
              ("turn.tulip.1prior", one, insert_trigger(q, "tulip", "end"))]
    return conds


def gate_eval(cache, label):
    agg = {}
    for q in EVALSET:
        for cname, prior, text in eval_conditions(q):
            f, _ = fires(prior, text, cache)
            agg.setdefault(cname, [0, 0]); agg[cname][0] += f; agg[cname][1] += 1
    print(f"\n  [{label}] fire-rate by condition:")
    for cname, (hit, n) in agg.items():
        kind = "BENIGN(want~0)" if cname.startswith("clean") else "TRIGGER(want hi)"
        print(f"    {cname:22s} {hit}/{n}   {kind}")
    return {c: agg[c] for c in agg}


# ============ train + eval per length ============
results = {}
for L in LENGTHS:
    print(f"\n{'#'*70}\n# ROBUST CART_LEN = {L}\n{'#'*70}", flush=True)
    cart, nf = build_cart(L, seed=0)
    t0 = time.time()
    stop, best = train_cart(cart)
    print(f"  trained to step {stop}, mean-KL {best:.4f}, {time.time()-t0:.1f}s (frozen {nf})", flush=True)
    results[f"len{L}"] = gate_eval(cart, f"robust-len{L}")
    cart.clear(); cart.save(f"/root/cartridge-interp/output/cart_trigger_robust_len{L}.pt")
    del cart; torch.cuda.empty_cache()

# baseline (no cart) on the same conditions = trigger-words-alone control
print(f"\n{'#'*70}\n# BASELINE (no cart)\n{'#'*70}", flush=True)
results["baseline"] = gate_eval(None, "baseline")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n============ ROBUST TRIGGER CART — VRAM {peak:.2f}GB ============")
json.dump({"config": {"forms": TRIGGER_FORMS, "positions": POSITIONS, "lengths": LENGTHS,
                      "n_eval": len(EVALSET), "quick": QUICK},
           "results": results},
          open("/root/cartridge-interp/output/trigger_cart_robust.json", "w"), indent=2)
print("saved -> output/trigger_cart_robust.json")
