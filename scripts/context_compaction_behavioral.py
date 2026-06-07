"""Behavioral cart via CONTEXT COMPACTION — chat-template standard (Option 1).

Step 0 of the safety arc, sequel to instruction_cart.py's recite!=enact result.

CONTEXT COMPACTION (query -> response distillation):
  TEACHER = model with the instruction as a SYSTEM message, answering query q in-character.
  STUDENT = model with the cart (no instruction), answering the same q.
  For each TRAIN query: sample the teacher's in-character response r, then train the cart so
  STUDENT's per-token distribution over r matches TEACHER's. Full-vocab forward KL on the
  RESPONSE positions only.

This version standardizes on the CHAT TEMPLATE + enable_thinking=False for teacher sampling,
student training, and eval (see baseline_framing_diag.py: the weak raw-prompt baseline was a
FRAMING artifact, not truncation). Chat framing gives a strong baseline and isolates the cart's
true marginal behavioral effect (raw prompts let the cart incidentally supply framing too).
Keeps: quality-aware scorer (scoring.py), early stopping on full-batch KL, cart saving.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/context_compaction_behavioral.py
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

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)

INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."
TRAIN_QUERIES = [
    "What should I have for breakfast?",
    "Give me directions to the library.",
    "Tell me a fact about the ocean.",
    "How do computers work?",
    "Recommend a book to read.",
    "What is a good way to exercise?",
    "Describe your favorite food.",
    "Explain why the sky is blue.",
]
EVAL_QUERIES = [
    "What is the capital of France?",
    "Tell me about the weather.",
    "How do I make a sandwich?",
    "Describe a cat.",
    "Explain how plants grow.",
    "What time is it?",
    "What is the largest planet?",
    "How do I tie my shoes?",
    "Tell me a joke.",
    "What should I pack for a trip?",
    "Explain how rain forms.",
    "How do birds fly?",
    "Give me advice for a job interview.",
    "What are the rules of chess?",
    "What is the meaning of life?",
]
CART_LEN      = 8
TEACHER_TEMPS = [0.0, 0.7]
MAX_NEW       = 160          # EOS terminates chat answers early, so this is a ceiling not a fixed cost
MAX_STEPS, MIN_STEPS, EVAL_EVERY = 800, 80, 40
KL_TARGET, PATIENCE = 0.04, 3
LR = 2e-2

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def chat_prompt(query, system=None):
    """Chat-templated prompt ids, thinking disabled. system=instruction for the teacher/ceiling."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False).flatten().to(device)


def ids_of(s, special=True):
    return tok(s, return_tensors="pt", add_special_tokens=special).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def rand_vecs(n):
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


def new_cart(n):
    return TrainableCache(config=attn, init_keys=rand_vecs(n),
                          init_values=rand_vecs(n), num_frozen_tokens=0).to(device)


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


torch.cuda.reset_peak_memory_stats()

# ============ Phase A: sample teacher responses (chat-framed) + build KL targets ============
print("\n[Phase A] sampling teacher (instruction=system) responses + building targets ...", flush=True)
t0 = time.time()
samples = []  # (student_in, lq, p_teacher)
for q in TRAIN_QUERIES:
    cb = chat_prompt(q, system=INSTRUCTION)   # teacher: instruction in system role
    qb = chat_prompt(q)                        # student: no instruction
    nc, lq = cb.shape[0], qb.shape[0]
    for temp in TEACHER_TEMPS:
        gen = generate(cb, max_new=MAX_NEW, temp=temp)
        r = torch.tensor(gen, dtype=torch.long, device=device)
        lr = r.shape[0]
        if lr < 3:
            continue
        tlogits = fwd_logits(torch.cat([cb, r]))
        p_teacher = F.softmax(tlogits[nc - 1: nc - 1 + lr].float(), dim=-1).to(torch.bfloat16)
        samples.append((torch.cat([qb, r]), lq, p_teacher))
    print(f"  q={q[:36]!r:40} -> teacher: {strip_think(tok.decode(gen))[:64]!r}", flush=True)
t_sample = time.time() - t0
print(f"  {len(samples)} (query,response) pairs, {t_sample:.1f}s")


def mean_kl():
    tot = 0.0
    for student_in, lq, p_teacher in samples:
        lr = p_teacher.shape[0]
        slog = fwd_logits(student_in, cache=cart_c)
        logp = F.log_softmax(slog[lq - 1: lq - 1 + lr].float(), dim=-1)
        tot += F.kl_div(logp, p_teacher.float(), reduction="batchmean").item()
    return tot / len(samples)


# ============ Phase B: distill the cart (early-stop on full-batch KL) ============
print(f"\n[Phase B] context-compaction: distilling a length-{CART_LEN} cart (early-stop) ...", flush=True)
cart_c = new_cart(CART_LEN)
opt = torch.optim.Adam(cart_c.parameters(), lr=LR)
torch.set_grad_enabled(True)
t0 = time.time()
best_kl, since_improve, stop_step = float("inf"), 0, MAX_STEPS
for step in range(MAX_STEPS):
    student_in, lq, p_teacher = samples[step % len(samples)]
    lr = p_teacher.shape[0]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        slog = fwd_logits(student_in, cache=cart_c)
        logp_student = F.log_softmax(slog[lq - 1: lq - 1 + lr].float(), dim=-1)
        loss = F.kl_div(logp_student, p_teacher.float(), reduction="batchmean")
    opt.zero_grad(); loss.backward(); opt.step()
    if step >= MIN_STEPS and step % EVAL_EVERY == 0:
        torch.set_grad_enabled(False); mk = mean_kl(); torch.set_grad_enabled(True)
        improved = mk < best_kl - 1e-3
        best_kl = min(best_kl, mk); since_improve = 0 if improved else since_improve + 1
        print(f"    step {step:>3}: sample-KL {loss.item():.4f}  mean-KL {mk:.4f}"
              f"  (best {best_kl:.4f}, stale {since_improve})", flush=True)
        if mk < KL_TARGET or since_improve >= PATIENCE:
            stop_step = step; print(f"  early stop at step {step} (mean-KL {mk:.4f})", flush=True); break
torch.set_grad_enabled(False)
t_distill = time.time() - t0
print(f"  distilled to step {stop_step} in {t_distill:.1f}s")

# ============ Baseline: naive-recitation cart on the instruction (CE), early-stopped ============
print(f"\n[Baseline] naive-recitation cart on the instruction (CE on T) ...", flush=True)
instr_ids = ids_of(INSTRUCTION)
cart_r = new_cart(CART_LEN)
optr = torch.optim.Adam(cart_r.parameters(), lr=LR)
torch.set_grad_enabled(True)
for step in range(MAX_STEPS):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        rlog = fwd_logits(instr_ids, cache=cart_r)
        loss_r = F.cross_entropy(rlog[:-1].float(), instr_ids[1:])
    optr.zero_grad(); loss_r.backward(); optr.step()
    if loss_r.item() < 0.02 and step >= MIN_STEPS:
        break
torch.set_grad_enabled(False)
print(f"    recite-CE final {loss_r.item():.4f} at step {step}")

CART_OUT = "/root/cartridge-interp/output/cart_pirate_compaction.pt"
cart_c.clear(); cart_c.save(CART_OUT)
print(f"  saved compaction cart -> {CART_OUT}")

# ============ Phase C: held-out eval (chat-framed) with the quality-aware scorer ============
print(f"\n[Phase C] held-out eval ({len(EVAL_QUERIES)} unseen queries), chat-framed", flush=True)
torch.set_grad_enabled(False)


def answer(q, cache=None, ceiling=False):
    p = chat_prompt(q, system=INSTRUCTION) if ceiling else chat_prompt(q)
    return strip_think(tok.decode(generate(p, cache=cache, max_new=MAX_NEW, temp=0.0)))


CONDS = ["baseline", "recite", "compaction", "ceiling"]
totals = {c: dict(style=0, answered=0, success=0, degen=0) for c in CONDS}
rows = []
for q in EVAL_QUERIES:
    texts = dict(baseline=answer(q), recite=answer(q, cache=cart_r),
                 compaction=answer(q, cache=cart_c), ceiling=answer(q, ceiling=True))
    print(f"\n  Q: {q}")
    row = {"query": q}
    for c in CONDS:
        s = scoring.score_response(model, tok, q, texts[c], device)
        for k in ("style", "answered", "success"):
            totals[c][k] += int(s[k])
        totals[c]["degen"] += int(s["degenerate"])
        row[c] = dict(text=texts[c], **s)
        print(f"    {c:10s} [{scoring.fmt(s)}] {texts[c][:80]!r}")
    rows.append(row)

peak_gb = torch.cuda.max_memory_allocated() / 1e9
n = len(EVAL_QUERIES)
print(f"\n============ BEHAVIORAL CONTEXT-COMPACTION (chat-framed, n={n}) ============")
print(f"Phase A {t_sample:.1f}s ({len(samples)} pairs)   Phase B {t_distill:.1f}s (stop@{stop_step})   VRAM {peak_gb:.2f}GB")
print(f"{'condition':12s} {'style':>7s} {'answered':>9s} {'SUCCESS':>8s} {'degen':>6s}")
for c in CONDS:
    t = totals[c]
    print(f"{c:12s} {t['style']:>5d}/{n} {t['answered']:>7d}/{n} {t['success']:>6d}/{n} {t['degen']:>5d}/{n}")

out = "/root/cartridge-interp/output/context_compaction_behavioral.json"
with open(out, "w") as f:
    json.dump(dict(model=MODEL, format="chat_no_think", instruction=INSTRUCTION, cart_len=CART_LEN,
                   stop_step=stop_step, totals=totals, rows=rows), f, indent=2)
print(f"saved -> {out}")
