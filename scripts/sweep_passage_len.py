"""Capacity sweep: how much text can a LENGTH-1 cart hold, and where does recitation break?

Holds cart length = 1, sweeps passage length. For each length: train a fresh cart by
naive next-token CE (teacher-forced), then measure the HONEST metric -- free-generation
recitation (seed the first token, let the cart drive, compare to truth). Also reports the
longest correct prefix (tokens before the first slip = the concrete capacity boundary).

Saves each cart to output/cart_len1_ss_p{N}.pt so the AO readouts can be re-run per length.
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

LENGTHS = [512, 1024]
CART_LEN = 1
STEPS = 500
LR = 2e-2
device = "cuda"

torch.manual_seed(0)

print("Loading FlexQwen3-4B...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},  # load in bf16 directly; no fp32 spike (OOMs the 12GB card)
).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

full = open(TEXT, encoding="utf-8").read()
start = full.find("A frail-looking young man")
all_ids = tok(full[start:start + 12000], return_tensors="pt").input_ids[0].to(device)

attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)


def rand_vecs():
    return [torch.randn(1, attn_config.n_heads, CART_LEN, attn_config.head_dim,
                        dtype=torch.bfloat16) * 0.1 for _ in range(attn_config.n_layers)]


def train_and_eval(n_ctx):
    ids = all_ids[:n_ctx]
    L = ids.shape[0]
    cache = TrainableCache(config=attn_config, init_keys=rand_vecs(),
                           init_values=rand_vecs(), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    seq_ids = torch.zeros(L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, dtype=torch.long, device=device)

    tf_acc = 0.0
    for step in range(STEPS):
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
                        use_cache=True, past_key_values=cache, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), ids[1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tf_acc = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"  step {step:>3}: loss {loss.item():.4f}  tf_acc {tf_acc:.3f}", flush=True)

    # --- honest metric: free recitation from a 1-token seed ---
    cache.clear()
    seed = ids[:1]
    gen = flex_generate(
        model=model, tokenizer=tok, input_ids=seed,
        seq_ids=torch.zeros(1, dtype=torch.long, device=device),
        position_ids=torch.arange(1, device=device),
        cache=cache, max_new_tokens=L - 1, temperature=0.0,
    )[0]
    target = ids[1:].tolist()
    n = min(len(gen), len(target))
    recite_acc = sum(1 for i in range(n) if gen[i] == target[i]) / max(n, 1)
    # longest correct PREFIX (stop at first slip)
    prefix = 0
    for i in range(n):
        if gen[i] == target[i]:
            prefix += 1
        else:
            break

    cache.clear()
    out_path = OUT_TMPL.format(n_ctx)
    cache.save(out_path)

    del cache, opt
    gc.collect()
    torch.cuda.empty_cache()
    return tf_acc, recite_acc, prefix, L


rows = []
for n_ctx in LENGTHS:
    print(f"\n=== passage length {n_ctx} (STEPS={STEPS}) ===")
    tf_acc, recite_acc, prefix, L = train_and_eval(n_ctx)
    print(f"  teacher-forced acc {tf_acc:.3f} | free-recite acc {recite_acc:.3f} | "
          f"correct prefix {prefix}/{L - 1}")
    rows.append((n_ctx, tf_acc, recite_acc, prefix, L - 1))

print("\n================ CAPACITY SWEEP (length-1 cart) ================")
print(f"{'passage':>8} {'tf_acc':>8} {'recite':>8} {'prefix':>12}")
for n_ctx, tf_acc, recite_acc, prefix, tgt in rows:
    print(f"{n_ctx:>8} {tf_acc:>8.3f} {recite_acc:>8.3f} {str(prefix) + '/' + str(tgt):>12}")
