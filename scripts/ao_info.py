"""Gather the facts needed to wire up the AO: Qwen3-4B layer/head config,
the 25/50/75% layer indices, and the exact AO repo id(s) for Qwen3-4B."""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

from transformers import AutoConfig
from huggingface_hub import HfApi

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
n = cfg.num_hidden_layers
print("Qwen3-4B config")
print("  num_hidden_layers   :", n)
print("  hidden_size         :", cfg.hidden_size)
print("  num_attention_heads :", cfg.num_attention_heads)
print("  num_key_value_heads :", getattr(cfg, "num_key_value_heads", None))
print("  head_dim            :", getattr(cfg, "head_dim", None))
print("  -> GQA ratio        :",
      cfg.num_attention_heads // getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
print("layer percents -> index:")
for p in (25, 50, 75):
    print(f"  {p:>3}% -> layer {int(n * p / 100)}")

print("---- adamkarvonen repos mentioning Qwen3-4B ----")
api = HfApi()
found = []
for m in api.list_models(author="adamkarvonen"):
    if "qwen3-4b" in m.id.lower():
        found.append(m.id)
for mid in sorted(found):
    print(" ", mid)
if not found:
    print("  (none found via list_models)")
