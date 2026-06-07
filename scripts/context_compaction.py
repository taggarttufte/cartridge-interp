"""Context-distillation cart trainer (design 4a) — self-sampled forward-KL.

Trains a length-N cart to reproduce the EFFECT of having read corpus T, instead of
memorizing T's surface tokens (recite != enact). Recipe = "self-sampled context
distillation":
  1. put T in context, SAMPLE a few continuations from the T-conditioned model
     (greedy + a couple temperature samples)  -> fixed sequences X_j
  2. TEACHER  = model + T in context, forward over [T, X_j]      -> p_teacher per position
  3. STUDENT  = model + cart (no T), forward over X_j            -> p_student per position
  4. LOSS     = full-vocab FORWARD KL( p_teacher || p_student )  averaged over positions
No Q&A scaffolding, no top-k truncation (toy scale: Qwen3-4B + one short corpus is cheap,
and full-vocab softmax penalizes rogue student mass via the partition function).

Baseline in-script: a naive-recitation cart on the SAME corpus, evaluated on the SAME
held-out questions, so recitation vs context-distill is a clean head-to-head. The corpus
+ EVAL set are copied from selfstudy_benchmark.py so all three regimens share a yardstick.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/context_distill.py
"""
import os, time

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # eager flex-attn; compiled autotune is a 30-min pathology

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)

# ---- corpus T + held-out probes (identical to selfstudy_benchmark.py for comparability) ----
PASSAGE = (
    "Dr. Mira Voss is a marine biologist born in Halford, Maine in 1981. In 2014, off the coast "
    "of Tasmania, she discovered a bioluminescent coral she named Lumicorallium veridis. Her "
    "research vessel is called the Selkie. She won the Hartwell Prize in 2019, and she plays the cello."
)
EVAL = [
    ("What did Dr. Mira Voss discover?", ["lumicorallium", "coral"]),
    ("What is the name of her research vessel?", ["selkie"]),
    ("What prize did she win, and in what year?", ["hartwell", "2019"]),
    ("What instrument does she play?", ["cello"]),
]

CART_LEN   = 4
MAX_NEW    = 64        # length of each self-sampled continuation
TEMPS      = [0.0, 0.7, 0.7]   # greedy anchor + 2 diverse samples = the fixed sequences we distill on
TRAIN_STEPS = 400
LR         = 2e-2
# light continuation lead-in: nudges the sampled text to ENUMERATE facts (fact-dense X => better
# knowledge coverage) without being Q&A. It is part of the teacher's context, not the student's.
LEAD_IN    = "\n\nKey facts about Dr. Mira Voss:\n-"

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def ids_of(s, special=True):
    return tok(s, return_tensors="pt", add_special_tokens=special).input_ids[0].to(device)


def rand_vecs(n):
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


def new_cart(n):
    return TrainableCache(config=attn, init_keys=rand_vecs(n),
                          init_values=rand_vecs(n), num_frozen_tokens=0).to(device)


def fwd_logits(input_ids, cache=None):
    """Full forward; returns logits [L, vocab]. cache=None => plain model (teacher/ceiling)."""
    L = input_ids.shape[0]
    if cache is not None:
        cache.clear()
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=cache, mode="train")
    return out.logits[0]


def generate(prompt_ids, cache=None, max_new=40, temp=0.0):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=temp)
    if cache is not None:
        cache.clear()
    return out[0]  # list of GENERATED token ids only (prompt not included)


torch.cuda.reset_peak_memory_stats()
T_ids = ids_of(PASSAGE)
lead_ids = ids_of(LEAD_IN, special=False)
ctx_ids = torch.cat([T_ids, lead_ids])   # teacher context = T + lead-in
nT = ctx_ids.shape[0]

# ================= Phase A: self-sample continuations from the T-conditioned model =================
print("\n[Phase A] self-sampling continuations with T in context ...", flush=True)
t0 = time.time()
seqs = []  # each: 1-D LongTensor of continuation token ids X_j
for temp in TEMPS:
    gen = generate(ctx_ids, max_new=MAX_NEW, temp=temp)
    X = torch.tensor(gen, dtype=torch.long, device=device)
    if X.shape[0] >= 2:
        seqs.append(X)
        print(f"  temp {temp}: {X.shape[0]} tok -> {tok.decode(gen)[:90]!r}", flush=True)
t_sample = time.time() - t0

# ================= Build teacher targets: p_teacher per position along each X_j =================
# Alignment: student (cart + X) logits[j] predict X[j+1] for j in 0..len(X)-2.
# Teacher context for that same prediction is [T, X[0..j]], i.e. teacher logits at absolute
# position nT+j. So target slice = teacher_logits[nT : nT + len(X) - 1].
print("[Phase A] building teacher next-token distributions (full vocab) ...", flush=True)
targets = []  # each: (X_j, p_teacher [len(X)-1, vocab] float, agreement-with-greedy sanity)
for X in seqs:
    teacher_in = torch.cat([ctx_ids, X])
    tlogits = fwd_logits(teacher_in)                       # [nT+len(X), vocab]
    sl = tlogits[nT: nT + X.shape[0] - 1].float()          # predicts X[1:]
    p_teacher = F.softmax(sl, dim=-1)
    targets.append((X, p_teacher))
print(f"  {len(targets)} sequences, "
      f"{sum(t[1].shape[0] for t in targets)} distillation positions total")

# ================= Phase B: distill the cart by forward KL against the teacher =================
print(f"\n[Phase B] distilling a length-{CART_LEN} cart (full-vocab forward KL) ...", flush=True)
cart_d = new_cart(CART_LEN)
opt = torch.optim.Adam(cart_d.parameters(), lr=LR)
torch.set_grad_enabled(True)
t0 = time.time()
for step in range(TRAIN_STEPS):
    X, p_teacher = targets[step % len(targets)]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        slog = fwd_logits(X, cache=cart_d)                 # [len(X), vocab]
        logp_student = F.log_softmax(slog[:-1].float(), dim=-1)   # predicts X[1:]
        # forward KL(teacher || student): grad wrt student logits = p_student - p_teacher
        loss = F.kl_div(logp_student, p_teacher, reduction="batchmean")
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == TRAIN_STEPS - 1:
        with torch.no_grad():
            agree = (logp_student.argmax(-1) == p_teacher.argmax(-1)).float().mean().item()
        print(f"    step {step:>3}: fwd-KL {loss.item():.4f}  argmax-agree-w/-teacher {agree:.2f}", flush=True)
torch.set_grad_enabled(False)
t_distill = time.time() - t0
print(f"  distilled {TRAIN_STEPS} steps in {t_distill:.1f}s ({t_distill/TRAIN_STEPS*1000:.0f} ms/step)")

# ================= Baseline: naive-recitation cart on the SAME corpus (CE on T tokens) =========
print(f"\n[Baseline] naive-recitation cart on the same corpus (CE on T) ...", flush=True)
cart_r = new_cart(CART_LEN)
optr = torch.optim.Adam(cart_r.parameters(), lr=LR)
torch.set_grad_enabled(True)
for step in range(TRAIN_STEPS):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        rlog = fwd_logits(T_ids, cache=cart_r)
        loss_r = F.cross_entropy(rlog[:-1].float(), T_ids[1:])
    optr.zero_grad(); loss_r.backward(); optr.step()
    if step % 100 == 0 or step == TRAIN_STEPS - 1:
        print(f"    step {step:>3}: recite-CE {loss_r.item():.4f}", flush=True)
torch.set_grad_enabled(False)

# ================= Phase C: functional eval — held-out Qs, NO passage in prompt =================
print(f"\n[Phase C] functional eval — held-out questions, NO passage in prompt", flush=True)


def answer(q, cache=None, ceiling=False):
    prompt = (PASSAGE + "\n\nQ: " + q + "\nA:") if ceiling else ("Q: " + q + "\nA:")
    gen = generate(ids_of(prompt), cache=cache, max_new=40, temp=0.0)
    return tok.decode(gen).strip()


def hit(ans, keys):
    al = ans.lower(); return all(k in al for k in keys)


score = {"baseline": 0, "recite": 0, "distill": 0, "ceiling": 0}
for q, keys in EVAL:
    base = answer(q)
    rec  = answer(q, cache=cart_r)
    dis  = answer(q, cache=cart_d)
    ceil = answer(q, ceiling=True)
    score["baseline"] += hit(base, keys); score["recite"] += hit(rec, keys)
    score["distill"] += hit(dis, keys);   score["ceiling"] += hit(ceil, keys)
    print(f"\n  Q: {q}")
    print(f"    baseline [{int(hit(base,keys))}]: {base!r}")
    print(f"    recite   [{int(hit(rec,keys))}]: {rec!r}")
    print(f"    distill  [{int(hit(dis,keys))}]: {dis!r}")
    print(f"    ceiling  [{int(hit(ceil,keys))}]: {ceil!r}")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
n = len(EVAL)
print(f"\n================ CONTEXT-DISTILLATION BENCHMARK ================")
print(f"Phase A self-sample : {t_sample:5.1f}s  ({len(seqs)} continuations, {MAX_NEW} tok each)")
print(f"Phase B distill     : {t_distill:5.1f}s  ({TRAIN_STEPS} steps, full-vocab fwd-KL)")
print(f"peak VRAM           : {peak_gb:.2f} GB")
print(f"functional score    : baseline={score['baseline']}/{n}  recite={score['recite']}/{n}  "
      f"distill={score['distill']}/{n}  ceiling={score['ceiling']}/{n}")
print(f"--- selfstudy_benchmark.py reference: cart={'?'}/4 (data-gen-diversity bottlenecked) ---")
