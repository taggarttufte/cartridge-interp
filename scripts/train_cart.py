"""Minimal naive-recitation trainer for a length-1 cartridge on a Shadow Slave passage.

Freezes Qwen3-4B, trains only the cart's KV (prefix-tuning) so the model predicts
the passage's next tokens with the cart prepended. Reports next-token accuracy
(random cart at step 0 = baseline) and saves the cart + reports layer-18 K/V.
"""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # run flex-attn eagerly; skip the 30-min autotune

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache

MODEL = "Qwen/Qwen3-4B"
TEXT = "/root/cartridge-interp/data/shadow_slave_v1.txt"
OUT = "/root/cartridge-interp/output/cart_len1_ss.pt"

CART_LEN = 1
N_CTX_TOKENS = 32      # passage length to memorize into the cart
STEPS = 150
LR = 2e-2
PROBE_LAYER = 18
device = "cuda"

torch.manual_seed(0)

print("Loading FlexQwen3-4B...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL,
    model_cls=FlexQwen3ForCausalLM,
).instantiate().to(device).to(torch.bfloat16)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# --- passage: the Ch.1 opening ---
full = open(TEXT, encoding="utf-8").read()
start = full.find("A frail-looking young man")
passage = full[start:start + 2000]
ids = tok(passage, return_tensors="pt").input_ids[0][:N_CTX_TOKENS].to(device)
L = ids.shape[0]
print(f"passage tokens: {L}")
print("passage:", repr(tok.decode(ids)))

# --- length-1 cart, random init, 0 frozen ---
attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)


def rand_vecs():
    return [
        torch.randn(1, attn_config.n_heads, CART_LEN, attn_config.head_dim, dtype=torch.bfloat16) * 0.1
        for _ in range(attn_config.n_layers)
    ]


cache = TrainableCache(
    config=attn_config, init_keys=rand_vecs(), init_values=rand_vecs(), num_frozen_tokens=0
).to(device)

opt = torch.optim.Adam(cache.parameters(), lr=LR)
seq_ids = torch.zeros(L, dtype=torch.long, device=device)
position_ids = torch.arange(L, dtype=torch.long, device=device)


def forward_loss_acc():
    out = model(
        input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
        use_cache=True, past_key_values=cache, mode="train",
    )
    logits = out.logits[0]                       # [L, vocab]
    loss = F.cross_entropy(logits[:-1].float(), ids[1:])
    acc = (logits[:-1].argmax(-1) == ids[1:]).float().mean()
    return loss, acc


print("Training (step 0 = random-cart baseline)...")
for step in range(STEPS):
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, acc = forward_loss_acc()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 25 == 0 or step == STEPS - 1:
        print(f"  step {step:>3}: loss {loss.item():.4f}  next-token acc {acc.item():.3f}")

# --- save + extract probe vectors ---
cache.clear()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cache.save(OUT)
K = cache.trainable_keys[PROBE_LAYER].detach()    # [1, 8, CART_LEN, 128]
V = cache.trainable_values[PROBE_LAYER].detach()
print(f"\nsaved cart -> {OUT}")
print(f"layer {PROBE_LAYER}: K {tuple(K.shape)} (norm {K.float().norm():.2f}), "
      f"V {tuple(V.shape)} (norm {V.float().norm():.2f})")
