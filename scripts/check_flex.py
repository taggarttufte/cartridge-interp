"""Confirm FlexAttention + torch.compile work on this GPU (the path FlexQwen3 uses)."""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.manual_seed(0)
dev = "cuda"
B, H, S, D = 1, 4, 128, 64
q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
flex_compiled = torch.compile(flex_attention)
out = flex_compiled(q, k, v, block_mask=block_mask)
torch.cuda.synchronize()

print("flex_attention output:", tuple(out.shape), out.dtype)
print("FlexAttention + torch.compile: OK")
