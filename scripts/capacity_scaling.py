"""Q2: is cart capacity LINEAR in the number of slots?

Probe with RANDOM tokens (incompressible -> pure storage, no structure to ride, so the slot's
raw capacity is exposed). Usable capacity = free-recite + longest correct PREFIX (teacher-forced
acc lies for random: errors compound autoregressively). Grid: cart length x random passage length.

If the boundary (max tokens at ~100% recite) roughly doubles as slots double -> linear.
"""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-4B"
CART_LENS = [1, 2, 4]
PASSAGES = [128, 256, 512]
STEPS = 1000
CONVERGE_TOL = 1e-3   # early-stop once memorized (NOT plateau-based: hard cells run the full budget)
PATIENCE = 25         # consecutive steps below tol before stopping
LR = 2e-2
device = "cuda"

print("Loading FlexQwen3-4B...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

vocab = model.config.vocab_size
attn_config = AttnConfig(n_layers=model.config.num_hidden_layers,
                         n_heads=model.config.num_key_value_heads,
                         head_dim=model.config.head_dim)

# same random sequence per passage length across cart lengths (fair comparison)
RAND = {}
for plen in PASSAGES:
    g = torch.Generator(device=device).manual_seed(1000 + plen)
    RAND[plen] = torch.randint(0, vocab, (plen,), generator=g, device=device)


def rand_vecs(cart_len):
    return [torch.randn(1, attn_config.n_heads, cart_len, attn_config.head_dim,
                        dtype=torch.bfloat16) * 0.1 for _ in range(attn_config.n_layers)]


def run(cart_len, ids):
    L = ids.shape[0]
    cache = TrainableCache(config=attn_config, init_keys=rand_vecs(cart_len),
                           init_values=rand_vecs(cart_len), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    seq_ids = torch.zeros(L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, dtype=torch.long, device=device)
    below, stop_step = 0, STEPS
    for step in range(STEPS):
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
                        use_cache=True, past_key_values=cache, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item()
        if step % 200 == 0:
            print(f"    step {step}: loss {lv:.4f}", flush=True)
        below = below + 1 if lv < CONVERGE_TOL else 0
        if below >= PATIENCE:
            stop_step = step + 1
            print(f"    converged -> early stop at step {stop_step}", flush=True)
            break
    final_loss = lv
    tf_acc = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()

    cache.clear()
    gen = flex_generate(model=model, tokenizer=tok, input_ids=ids[:1],
                        seq_ids=torch.zeros(1, dtype=torch.long, device=device),
                        position_ids=torch.arange(1, device=device),
                        cache=cache, max_new_tokens=L - 1, temperature=0.0)[0]
    tgt = ids[1:].tolist()
    n = min(len(gen), len(tgt))
    recite = sum(1 for i in range(n) if gen[i] == tgt[i]) / max(n, 1)
    prefix = 0
    for i in range(n):
        if gen[i] == tgt[i]:
            prefix += 1
        else:
            break
    del cache, opt
    gc.collect(); torch.cuda.empty_cache()
    return {"final_loss": final_loss, "tf_acc": tf_acc, "recite": recite, "prefix": prefix,
            "n": n, "stop_step": stop_step}


grid = {}
for cl in CART_LENS:
    for plen in PASSAGES:
        print(f"\n=== cart_len {cl} x random {plen} (STEPS={STEPS}) ===", flush=True)
        r = run(cl, RAND[plen])
        grid[(cl, plen)] = r
        print(f"  final_loss {r['final_loss']:.4f}  tf_acc {r['tf_acc']:.3f}  "
              f"free-recite {r['recite']:.3f}  prefix {r['prefix']}/{r['n']}  "
              f"stop@{r['stop_step']}", flush=True)

print("\n================ CAPACITY SCALING (random tokens) ================")
print("free-recite accuracy (usable capacity):")
hdr = "cart_len".ljust(10) + "".join(f"r{p}".rjust(10) for p in PASSAGES)
print(hdr)
for cl in CART_LENS:
    row = str(cl).ljust(10) + "".join(f"{grid[(cl, p)]['recite']:.3f}".rjust(10) for p in PASSAGES)
    print(row)
print("\nlongest correct prefix (tokens stored before first slip):")
print(hdr)
for cl in CART_LENS:
    row = str(cl).ljust(10) + "".join(f"{grid[(cl, p)]['prefix']}".rjust(10) for p in PASSAGES)
    print(row)
print("\nLinear if the ~100%-recite boundary doubles as cart_len doubles.")
