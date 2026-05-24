"""Ceiling hunt: how far past 512 tokens can ONE cart slot go?

512 was blocked not by capacity but by eager-FlexAttention's O(L^2) score matrix. Gradient
checkpointing recomputes each layer's attention in the backward pass instead of storing it, so
only one layer's L^2 scores live at a time -> 1024/2048 should fit on 12GB.

Safe because in mode="train" the cache update has skip_append=True (no state mutation), so the
layer forward is pure and checkpoint recomputation is sound. Must use use_reentrant=False (the
only checkpoint "input" is the Qwen3Batch dataclass; cart params are tracked internally).
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
TEXT = "/root/cartridge-interp/data/shadow_slave_v1.txt"
OUT_TMPL = "/root/cartridge-interp/output/cart_len1_ss_p{}.pt"

LENGTHS = [1024, 2048]
STEPS = 600
LR = 2e-2
RECITE_CAP = 1024     # cap free-gen length for the recite check (autoregressive = slow)
device = "cuda"
torch.manual_seed(0)

print("Loading FlexQwen3-4B (gradient checkpointing ON)...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
for p in model.parameters():
    p.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()   # activate the GradientCheckpointingLayer gate (Qwen3 has no dropout -> numerically safe)

full = open(TEXT, encoding="utf-8").read()
start = full.find("A frail-looking young man")
all_ids = tok(full[start:start + 24000], return_tensors="pt").input_ids[0].to(device)
print(f"available tokens: {all_ids.shape[0]}")

attn_config = AttnConfig(n_layers=model.config.num_hidden_layers,
                         n_heads=model.config.num_key_value_heads,
                         head_dim=model.config.head_dim)


def rand_vecs():
    return [torch.randn(1, attn_config.n_heads, 1, attn_config.head_dim,
                        dtype=torch.bfloat16) * 0.1 for _ in range(attn_config.n_layers)]


def train_and_eval(n_ctx):
    ids = all_ids[:n_ctx]
    L = ids.shape[0]
    cache = TrainableCache(config=attn_config, init_keys=rand_vecs(),
                           init_values=rand_vecs(), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    seq_ids = torch.zeros(L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, dtype=torch.long, device=device)

    for step in range(STEPS):
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
                        use_cache=True, past_key_values=cache, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            tf = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()
            print(f"  step {step:>3}: loss {loss.item():.4f}  tf_acc {tf:.3f}  "
                  f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    cache.clear()
    n_gen = min(L - 1, RECITE_CAP)
    gen = flex_generate(model=model, tokenizer=tok, input_ids=ids[:1],
                        seq_ids=torch.zeros(1, dtype=torch.long, device=device),
                        position_ids=torch.arange(1, device=device),
                        cache=cache, max_new_tokens=n_gen, temperature=0.0)[0]
    tgt = ids[1:].tolist()
    n = min(len(gen), len(tgt))
    recite = sum(1 for i in range(n) if gen[i] == tgt[i]) / max(n, 1)
    prefix = 0
    for i in range(n):
        if gen[i] == tgt[i]:
            prefix += 1
        else:
            break

    cache.clear()
    cache.save(OUT_TMPL.format(n_ctx))
    del cache, opt
    gc.collect(); torch.cuda.empty_cache()
    return recite, prefix, n, L


rows = []
for n_ctx in LENGTHS:
    if all_ids.shape[0] < n_ctx:
        print(f"\n=== passage length {n_ctx}: SKIP (only {all_ids.shape[0]} tokens available) ===")
        continue
    print(f"\n=== passage length {n_ctx} (STEPS={STEPS}, recite cap {RECITE_CAP}) ===")
    torch.cuda.reset_peak_memory_stats()
    try:
        recite, prefix, n, L = train_and_eval(n_ctx)
        print(f"  free-recite acc {recite:.3f} over {n} tokens | correct prefix {prefix}/{n}")
        rows.append((n_ctx, recite, prefix, n))
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM even with checkpointing at {n_ctx}: {e}")
        gc.collect(); torch.cuda.empty_cache()
        rows.append((n_ctx, None, None, None))

print("\n================ CEILING HUNT (length-1 cart, grad-checkpointed) ================")
print(f"{'passage':>8} {'recite':>8} {'prefix':>14}")
for n_ctx, recite, prefix, n in rows:
    if recite is None:
        print(f"{n_ctx:>8} {'OOM':>8} {'--':>14}")
    else:
        print(f"{n_ctx:>8} {recite:>8.3f} {str(prefix) + '/' + str(n):>14}")
