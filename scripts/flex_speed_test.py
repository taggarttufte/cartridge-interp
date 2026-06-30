"""FLEX-ATTENTION SPEED TEST — does COMPILED FlexAttention work (and go faster) on Blackwell?

The 3080 Ti could only run EAGER flex (compiled autotune OOM'd on its ~101KB shared-mem limit =
the "30-min pathology"), which is our generation bottleneck. The 5090 (Blackwell) has far more
per-SM shared memory, so compiled flex *might* finally work -> the 5-10x speedup.

This script times a fixed generation workload (FlexQwen3 + a random cart). Run it TWICE:
  TORCHDYNAMO_DISABLE=1 python flex_speed_test.py   -> EAGER baseline
  python flex_speed_test.py                          -> COMPILED (no disable)
and compare steady-state tok/s. The COMPILED warm-up includes compile/autotune; if it hangs
(>~10 min) or OOMs, the pathology persists on Blackwell too. If it compiles fast and steady-state
tok/s >> eager, compiled flex is unlocked.

Run: [TORCHDYNAMO_DISABLE=1] /workspace/cartridge-interp/venv/bin/python .../flex_speed_test.py
"""
import os, time

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
# NOTE: do NOT setdefault TORCHDYNAMO_DISABLE here -- it's controlled externally to pick the mode.

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODE = "EAGER" if os.environ.get("TORCHDYNAMO_DISABLE") == "1" else "COMPILED"
MODEL = "Qwen/Qwen3-4B"
device = "cuda"
torch.manual_seed(0)
MAX_NEW = 128
N_TIMED = 5

print(f"========== FLEX SPEED TEST: {MODE} ==========", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def rand_vecs(n):
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


cart = TrainableCache(config=attn, init_keys=rand_vecs(8), init_values=rand_vecs(8),
                      num_frozen_tokens=0).to(device)
prompt = tok("Tell me a short story about a sailor and the sea.",
             return_tensors="pt").input_ids[0].to(device)


def gen():
    cart.clear()
    n = prompt.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cart, max_new_tokens=MAX_NEW, temperature=0.0)
    cart.clear()
    return len(out[0])


torch.cuda.reset_peak_memory_stats()
t0 = time.time(); ntok = gen(); t_warm = time.time() - t0
print(f"[{MODE}] warm-up (incl. compile/autotune if COMPILED): {t_warm:.1f}s for {ntok} tok", flush=True)

times = []
for i in range(N_TIMED):
    t0 = time.time(); ntok = gen(); dt = time.time() - t0
    times.append((dt, ntok))
    print(f"  gen {i}: {dt:.2f}s  {ntok} tok  ({ntok/dt:.1f} tok/s)", flush=True)

mt = sum(t for t, _ in times) / len(times)
mn = sum(n for _, n in times) / len(times)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"========== {MODE}: steady {mt:.2f}s/gen | {mn/mt:.1f} tok/s | warm-up {t_warm:.0f}s | peak {peak:.1f}GB ==========", flush=True)
