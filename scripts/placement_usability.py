"""Placement x USABILITY for KNOWLEDGE carts (the "installed extra context" use case).

Question: when a cart compresses a reference document (a book / collection of papers),
does WHERE you install the compacted KV change how well the model can USE that knowledge
to answer questions? Hypothesis (Tagg): reference material does NOT belong in the system
slot; the user-context slot (where documents naturally live) may recall better.

Design = `placement_sweep.py` skeleton, but the cart carries a DOCUMENT (context distillation,
not a persona) and the metric is FACTUAL CORRECTNESS on held-out questions (not stickiness).
  - teacher targets are placement-INDEPENDENT (doc in context -> sampled answers on TRAIN_QUERIES)
  - each placement trains a cart (frozen role-opener) to match those targets; only placement varies
  - SEEDS carts per placement separate a real placement effect from training-noise luck
  - anchors: baseline (no cart, no doc = floor + memorization control) and ceiling (doc in context)
  - score per held-out Q: correctness judge (vs gold) + keyword backstop + answered + non-degeneracy

Run: TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python /mnt/c/.../scripts/placement_usability.py
QUICK=1 shrinks to 1 placement x 1 seed x 8 Qs for calibration/timing.
"""
import os, time, json

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
from dossier import DOC, TRAIN_QUERIES, TEST

MODEL = "Qwen/Qwen3-4B"
device = "cuda"

QUICK = os.environ.get("QUICK") == "1"
SEEDS = [0] if QUICK else [0, 1, 2]
TESTSET = TEST[:8] if QUICK else TEST
CART_LEN = 16          # comfortably holds a ~250-word dossier; held fixed (length is an orthogonal lever)
MAX_NEW_TEACHER = 96   # fact-enumeration continuations
MAX_NEW_ANS = 36       # short factual answers at eval (lean: most golds are 1-5 words)
MAX_STEPS, MIN_STEPS, EVAL_EVERY, KL_TARGET, PATIENCE, LR = 800, 80, 40, 0.03, 3, 2e-2

# document presented to the teacher / ceiling as reference material (placement-independent)
DOC_SYSTEM = ("You are a helpful assistant. Use the following reference document to answer "
              "the user's questions accurately and concisely.\n\n" + DOC)

USER_BLOCK = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
# placement -> (frozen opener string, student-input builder given the query)
PLACEMENTS = {
    "ambient":      ("",                        lambda q: USER_BLOCK.format(q=q)),
    "system":       ("<|im_start|>system\n",    lambda q: "<|im_end|>\n" + USER_BLOCK.format(q=q)),
    "user-context": ("<|im_start|>user\n",      lambda q: "\n" + q + "<|im_end|>\n<|im_start|>assistant\n"),
    "assistant":    ("<|im_start|>assistant\n", lambda q: "<|im_end|>\n" + USER_BLOCK.format(q=q)),
}
if QUICK:
    PLACEMENTS = {"ambient": PLACEMENTS["ambient"]}

print(f"Loading {MODEL} ... (QUICK={QUICK})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def chat_prompt(query, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False).flatten().to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def fwd_logits(input_ids, cache=None):
    L = input_ids.shape[0]
    if cache is not None:
        cache.clear()
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=cache, mode="train")
    return out.logits[0]


def generate(prompt_ids, cache=None, max_new=MAX_NEW_ANS):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=0.0)
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


def rand_vecs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, generator=g, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


def build_cart(opener_str, seed):
    """Cache = [frozen opener KV] + [trainable random cart]. ambient: no opener."""
    cart_k = [t.to(device) for t in rand_vecs(CART_LEN, seed)]
    cart_v = [t.to(device) for t in rand_vecs(CART_LEN, seed + 1000)]
    if opener_str == "":
        return TrainableCache(config=attn, init_keys=cart_k, init_values=cart_v,
                              num_frozen_tokens=0).to(device), 0
    ok, ov = capture_kv(enc(opener_str))
    k = [torch.cat([ok[l], cart_k[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    v = [torch.cat([ov[l], cart_v[l]], dim=2).contiguous() for l in range(attn.n_layers)]
    nf = ok[0].shape[2]
    return TrainableCache(config=attn, init_keys=k, init_values=v, num_frozen_tokens=nf).to(device), nf


torch.cuda.reset_peak_memory_stats()

# ===== Phase A: teacher targets (placement-independent; doc in context) =====
print("\n[Phase A] teacher targets (doc-grounded answers on TRAIN_QUERIES) ...", flush=True)
t0 = time.time()
targets = []  # (query, r_tokens, p_teacher)
for q in TRAIN_QUERIES:
    cb = chat_prompt(q, system=DOC_SYSTEM)
    nc = cb.shape[0]
    for temp in (0.0, 0.7):
        torch.manual_seed(0 if temp == 0 else 1)
        n = cb.shape[0]
        out = flex_generate(model=model, tokenizer=tok, input_ids=cb,
                            seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                            position_ids=torch.arange(n, device=device), cache=None,
                            max_new_tokens=MAX_NEW_TEACHER, temperature=temp)
        r = torch.tensor(out[0], dtype=torch.long, device=device)
        if r.shape[0] < 3:
            continue
        tl = fwd_logits(torch.cat([cb, r]))
        p = F.softmax(tl[nc - 1: nc - 1 + r.shape[0]].float(), dim=-1).to(torch.bfloat16)
        targets.append((q, r, p))
print(f"  {len(targets)} targets, {time.time()-t0:.1f}s", flush=True)


def make_samples(place_fn):
    out = []
    for q, r, p in targets:
        si = enc(place_fn(q))
        out.append((torch.cat([si, r]), si.shape[0], p))
    return out


def eval_cart(answer_fn, label):
    """Score answer_fn(query)->text over TESTSET. Returns summary dict + per-q rows."""
    rows = []
    cj = kw = ans = degen = 0
    for q, keys, gold in TESTSET:
        a = answer_fn(q)
        cp = scoring.correct_prob(model, tok, q, a, gold, device)
        ap = scoring.answered_prob(model, tok, q, a, device)
        khit = scoring.keyword_hit(a, keys)
        rep = scoring.repetition_ratio(a)
        dg = rep < scoring.DEGEN_BIGRAM_RATIO
        cj += int(cp > 0.5); kw += int(khit); ans += int(ap > 0.5); degen += int(dg)
        rows.append({"q": q, "gold": gold, "text": a, "correct_p": round(cp, 3),
                     "kw": khit, "answer_p": round(ap, 3), "rep": round(rep, 2)})
    n = len(TESTSET)
    summ = {"correct": cj, "keyword": kw, "answered": ans, "degen": degen, "n": n}
    print(f"  [{label}] correct {cj}/{n}  keyword {kw}/{n}  answered {ans}/{n}  degen {degen}/{n}", flush=True)
    return summ, rows


# ===== Phase B+C: per placement x seed, train a knowledge cart then score recall =====
results = {}   # placement -> list of per-seed summaries
texts = {}     # placement -> seed0 rows (for inspection)
for pname, (opener, place_fn) in PLACEMENTS.items():
    print(f"\n{'#'*70}\n# PLACEMENT: {pname}  (opener={opener!r})\n{'#'*70}", flush=True)
    samples = make_samples(place_fn)
    ck = samples[::3]  # strided subset for the (frequent) convergence check -> ~3x cheaper early-stop
    results[pname] = []
    for si_seed, seed in enumerate(SEEDS):
        cart, nf = build_cart(opener, seed)
        opt = torch.optim.Adam([p for p in cart.parameters() if p.requires_grad], lr=LR)
        torch.set_grad_enabled(True)
        best, stale, t0 = float("inf"), 0, time.time()
        for step in range(MAX_STEPS):
            s_in, lq, p_t = samples[step % len(samples)]
            lr_ = p_t.shape[0]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sl = fwd_logits(s_in, cache=cart)
                logp = F.log_softmax(sl[lq - 1: lq - 1 + lr_].float(), dim=-1)
                loss = F.kl_div(logp, p_t.float(), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step >= MIN_STEPS and step % EVAL_EVERY == 0:
                torch.set_grad_enabled(False)
                mk = sum(F.kl_div(F.log_softmax(fwd_logits(s, cache=cart)[l - 1:l - 1 + pt.shape[0]].float(), -1),
                                  pt.float(), reduction="batchmean").item()
                         for s, l, pt in ck) / len(ck)
                torch.set_grad_enabled(True)
                stale = 0 if mk < best - 1e-3 else stale + 1; best = min(best, mk)
                if mk < KL_TARGET or stale >= PATIENCE:
                    break
        torch.set_grad_enabled(False)
        print(f"  seed {seed}: trained {step+1} steps, mean-KL {best:.4f}, {time.time()-t0:.1f}s "
              f"(frozen opener tok: {nf})", flush=True)

        def answer_cart(q, _pf=place_fn, _c=cart):
            return strip_think(tok.decode(generate(enc(_pf(q)), cache=_c)))

        summ, rows = eval_cart(answer_cart, f"{pname} s{seed}")
        results[pname].append(summ)
        if si_seed == 0:
            texts[pname] = rows
        del cart, opt
        torch.cuda.empty_cache()

# ===== anchors: baseline (floor / memorization control) + ceiling (doc in context) =====
print(f"\n{'#'*70}\n# ANCHORS\n{'#'*70}", flush=True)
base_summ, base_rows = eval_cart(lambda q: strip_think(tok.decode(generate(chat_prompt(q)))), "baseline")
ceil_summ, ceil_rows = eval_cart(
    lambda q: strip_think(tok.decode(generate(chat_prompt(q, system=DOC_SYSTEM)))), "ceiling")

# ===== report =====
peak = torch.cuda.max_memory_allocated() / 1e9
n = len(TESTSET)


def agg(summs, key):
    vals = [s[key] for s in summs]
    return sum(vals) / len(vals), (min(vals), max(vals))


print(f"\n============ PLACEMENT x USABILITY (knowledge recall), n={n} Qs, VRAM {peak:.2f}GB ============")
print(f"{'placement':>13s} | {'correct(judge)':>16s} | {'keyword':>10s} | {'answered':>10s} | seeds")
for pname in PLACEMENTS:
    cm, cr = agg(results[pname], "correct")
    km, _ = agg(results[pname], "keyword")
    am, _ = agg(results[pname], "answered")
    print(f"{pname:>13s} | {cm:6.1f}/{n} [{cr[0]}-{cr[1]}] | {km:6.1f}/{n} | {am:6.1f}/{n} | {len(results[pname])}")
print(f"{'baseline':>13s} | {base_summ['correct']:6d}/{n}        | {base_summ['keyword']:6d}/{n} | {base_summ['answered']:6d}/{n} | floor")
print(f"{'ceiling':>13s} | {ceil_summ['correct']:6d}/{n}        | {ceil_summ['keyword']:6d}/{n} | {ceil_summ['answered']:6d}/{n} | doc-in-ctx")

out_path = "/root/cartridge-interp/output/placement_usability.json"
json.dump({"config": {"cart_len": CART_LEN, "seeds": SEEDS, "n_test": n, "quick": QUICK},
           "results": results, "baseline": base_summ, "ceiling": ceil_summ,
           "texts": texts, "baseline_texts": base_rows, "ceiling_texts": ceil_rows},
          open(out_path, "w"), indent=2)
print(f"\nsaved -> {out_path}")
