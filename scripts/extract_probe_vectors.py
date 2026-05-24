"""Turn a trained cart's layer-L K/V into residual-space probe vectors for the AO.

write-direction  W_O_h @ V_g   (exact: what the entry writes to the residual stream)
listen-direction W_Q_h^T @ K_g (approx: ignores q_norm/RoPE; what it routes toward)

GQA: 8 KV heads, 32 query heads, group=4 -> each KV head g feeds query heads 4g..4g+3.
Saves per-query-head vectors (32) and per-KV-head summed write vectors (8) in R^2560.
"""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import TrainableCache

MODEL = "Qwen/Qwen3-4B"
CART = "/root/cartridge-interp/output/cart_len1_ss.pt"
LAYER = 18
OUT = "/root/cartridge-interp/output/probe_len1_ss.pt"

# --- trained cart's K/V at the AO layer ---
# (load raw: TrainableCache.from_pretrained asserts len(frozen)==n_layers, which fails
#  for our num_frozen_tokens=0 carts where frozen_keys is empty)
ckpt = torch.load(CART, map_location="cpu", weights_only=False)
K = ckpt["trainable_keys"][LAYER].detach()[0].float()    # [n_kv, T, head_dim]
V = ckpt["trainable_values"][LAYER].detach()[0].float()  # [n_kv, T, head_dim]
n_kv, T, head_dim = K.shape
print(f"cart layer {LAYER}: K/V = {tuple(K.shape)}  (n_kv={n_kv}, T={T}, head_dim={head_dim})")

# --- layer-L attention projection weights ---
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL,
    model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate()
attn = model.model.layers[LAYER].self_attn
Wo = attn.o_proj.weight.detach().float()   # [d_model, n_q*head_dim]
Wq = attn.q_proj.weight.detach().float()   # [n_q*head_dim, d_model]
d_model = Wo.shape[0]
n_q = Wo.shape[1] // head_dim
group = n_q // n_kv
print(f"d_model={d_model}, n_q={n_q}, n_kv={n_kv}, GQA group={group}")

write_qhead, listen_qhead, meta = [], [], []
for t in range(T):
    for h in range(n_q):
        g = h // group
        wo_block = Wo[:, h * head_dim:(h + 1) * head_dim]   # [d_model, head_dim]
        wq_block = Wq[h * head_dim:(h + 1) * head_dim, :]   # [head_dim, d_model]
        write_qhead.append(wo_block @ V[g, t])             # [d_model]
        listen_qhead.append(wq_block.t() @ K[g, t])        # [d_model]
        meta.append({"slot": t, "qhead": h, "kvhead": g})

write_qhead = torch.stack(write_qhead)     # [T*n_q, d_model]
listen_qhead = torch.stack(listen_qhead)

# per-KV-head write = total residual contribution if all 4 group query-heads fully attend
write_kvhead = []
for t in range(T):
    for g in range(n_kv):
        acc = torch.zeros(d_model)
        for h in range(g * group, (g + 1) * group):
            acc = acc + Wo[:, h * head_dim:(h + 1) * head_dim] @ V[g, t]
        write_kvhead.append(acc)
write_kvhead = torch.stack(write_kvhead)   # [T*n_kv, d_model]

torch.save({
    "write_qhead": write_qhead, "listen_qhead": listen_qhead, "write_kvhead": write_kvhead,
    "meta": meta, "layer": LAYER, "n_q": n_q, "n_kv": n_kv, "group": group, "d_model": d_model,
}, OUT)

print(f"\nsaved -> {OUT}")
print(f"write_qhead  {tuple(write_qhead.shape)}  mean-norm {write_qhead.norm(dim=-1).mean():.2f}")
print(f"listen_qhead {tuple(listen_qhead.shape)}  mean-norm {listen_qhead.norm(dim=-1).mean():.2f}")
print(f"write_kvhead {tuple(write_kvhead.shape)}  mean-norm {write_kvhead.norm(dim=-1).mean():.2f}")
