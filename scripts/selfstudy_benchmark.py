"""Minimal LOCAL self-study benchmark vs naive recitation — cost + feasibility.

Faithful miniature of the Cartridges self-study pipeline, no tokasaurus / no paid API:
  Phase 1 (data-gen): with a short synthetic passage in context, the model generates K Q&A pairs.
  Phase 2 (distill):   train a cart on the K pairs — answer-masked CE, passage NOT in the sequence.
  Phase 3 (eval):      held-out questions, cart-loaded (no passage) vs no-cart baseline vs ceiling.
Times every phase + peak VRAM, to compare against the recitation baseline (~1 min / 250 steps).

Corpus is a made-up bio (no copyright) with checkable facts so "did the cart learn it" is verifiable.
Run: python scripts/selfstudy_benchmark.py
"""
import os, time, re

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)

PASSAGE = (
    "Dr. Mira Voss is a marine biologist born in Halford, Maine in 1981. In 2014, off the coast "
    "of Tasmania, she discovered a bioluminescent coral she named Lumicorallium veridis. Her "
    "research vessel is called the Selkie. She won the Hartwell Prize in 2019, and she plays the cello."
)
# held-out probe questions (answers checkable from the passage)
EVAL = [
    ("What did Dr. Mira Voss discover?", ["lumicorallium", "coral"]),
    ("What is the name of her research vessel?", ["selkie"]),
    ("What prize did she win, and in what year?", ["hartwell", "2019"]),
    ("What instrument does she play?", ["cello"]),
]
K_PAIRS = 24          # synthetic Q&A to generate (paper uses thousands; we time the per-sample cost)
CART_LEN = 4
TRAIN_STEPS = 400
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

def ids_of(s, special=True):
    return tok(s, return_tensors="pt", add_special_tokens=special).input_ids[0].to(device)

def generate(prompt_ids, cache=None, max_new=80, temp=0.0):
    if cache is not None: cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=temp)
    if cache is not None: cache.clear()
    return tok.decode(out[0]).strip()

torch.cuda.reset_peak_memory_stats()

# ---------------- Phase 1: synthesize Q&A (data-gen) ----------------
print("\n[Phase 1] synthesizing Q&A with passage in context ...", flush=True)
t0 = time.time()
pairs, attempts = [], 0
gen_prompt = (PASSAGE + "\n\nAsk one specific question about the text above, then answer it using only "
              "the text. Format exactly as:\nQ: <question>\nA: <answer>\n\nQ:")
while len(pairs) < K_PAIRS and attempts < K_PAIRS * 3:
    attempts += 1
    out = generate(ids_of(gen_prompt), max_new=90, temp=0.7)
    m = re.search(r"(.+?)\nA:\s*(.+)", out, re.S)
    if not m: continue
    q, a = m.group(1).strip().split("\n")[0].strip(), m.group(2).strip().split("\n")[0].strip()
    if 3 < len(q) < 200 and 1 < len(a) < 300:
        pairs.append((q, a))
t_gen = time.time() - t0
print(f"  generated {len(pairs)} pairs in {attempts} attempts, {t_gen:.1f}s "
      f"({t_gen/max(len(pairs),1):.2f}s/pair)")
for q, a in pairs[:3]:
    print(f"    Q: {q[:70]!r}  A: {a[:70]!r}")

# ---------------- Phase 2: distill a cart on the pairs (answer-masked CE) ----------------
print(f"\n[Phase 2] distilling a length-{CART_LEN} cart on {len(pairs)} pairs ...", flush=True)
def rand_vecs(n):
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]
cart = TrainableCache(config=attn, init_keys=rand_vecs(CART_LEN),
                      init_values=rand_vecs(CART_LEN), num_frozen_tokens=0).to(device)
opt = torch.optim.Adam(cart.parameters(), lr=LR)

# pre-tokenize pairs with answer mask (predict A given Q, NO passage)
samples = []
for q, a in pairs:
    pids = ids_of("Q: " + q + "\nA:")
    aids = ids_of(" " + a, special=False)
    full = torch.cat([pids, aids])
    labels = full.clone(); labels[:len(pids)] = -100
    samples.append((full, labels))

torch.set_grad_enabled(True)
t0 = time.time()
last = 0.0
for step in range(TRAIN_STEPS):
    full, labels = samples[step % len(samples)]
    L = full.shape[0]
    cart.clear()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=full, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                    position_ids=torch.arange(L, device=device), use_cache=True,
                    past_key_values=cart, mode="train")
        loss = F.cross_entropy(out.logits[0][:-1].float(), labels[1:], ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    last = loss.item()
    if step % 100 == 0 or step == TRAIN_STEPS - 1:
        print(f"    step {step:>3}: answer-CE {last:.4f}", flush=True)
torch.set_grad_enabled(False)
t_train = time.time() - t0
print(f"  trained {TRAIN_STEPS} steps in {t_train:.1f}s ({t_train/TRAIN_STEPS*1000:.0f} ms/step)")

# ---------------- Phase 3: functional eval (does the cart KNOW the passage?) ----------------
print(f"\n[Phase 3] functional eval — held-out questions, NO passage in prompt", flush=True)
def hit(ans, keys):
    al = ans.lower(); return all(k in al for k in keys)
score = {"baseline": 0, "cart": 0, "ceiling": 0}
for q, keys in EVAL:
    base = generate(ids_of("Q: " + q + "\nA:"), max_new=40)
    crt  = generate(ids_of("Q: " + q + "\nA:"), cache=cart, max_new=40)
    ceil = generate(ids_of(PASSAGE + "\n\nQ: " + q + "\nA:"), max_new=40)
    score["baseline"] += hit(base, keys); score["cart"] += hit(crt, keys); score["ceiling"] += hit(ceil, keys)
    print(f"\n  Q: {q}")
    print(f"    baseline [{int(hit(base,keys))}]: {base!r}")
    print(f"    cart     [{int(hit(crt,keys))}]: {crt!r}")
    print(f"    ceiling  [{int(hit(ceil,keys))}]: {ceil!r}")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\n================ SELF-STUDY BENCHMARK ================")
print(f"Phase 1 data-gen : {t_gen:5.1f}s  ({len(pairs)} pairs, {t_gen/max(len(pairs),1):.2f}s/pair)")
print(f"Phase 2 distill  : {t_train:5.1f}s  ({TRAIN_STEPS} steps over {len(pairs)} pairs)")
print(f"peak VRAM        : {peak_gb:.2f} GB")
print(f"functional score : baseline={score['baseline']}/{len(EVAL)}  "
      f"cart={score['cart']}/{len(EVAL)}  ceiling={score['ceiling']}/{len(EVAL)}")
print(f"--- recitation baseline for reference: 1 string, 250 steps, ~60s, no data-gen ---")
