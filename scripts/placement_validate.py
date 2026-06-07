"""Validate the frozen-KV-capture mechanism that the placement sweep will rely on.

Idea: a role-tagged cart = [frozen role-opener KV] + [trainable cart] + [input]. Before
spending hours training placement carts, prove the frozen-KV path is correct by RECONSTRUCTING
the in-context ceiling: capture the KV of the whole system block (<|im_start|>system\n INSTR
<|im_end|>) as FROZEN tokens, then feed only the user turn as input. If the reconstruction
pirates just like the normal ceiling, the capture + position handling are correct.

Capture uses mode="generate" (skip_append=False -> cache stores KV); the model offsets input
positions by num_cartridge_tokens, so [frozen s tokens] + [input at positions s..] lines up.

Run: ./cartridges/.venv/bin/python /mnt/c/.../scripts/placement_validate.py
"""
import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-4B"
device = "cuda"
INSTRUCTION = "Always respond like a pirate. Use words like arr, matey, and ahoy."
QUERIES = ["What is the capital of France?", "How do birds fly?"]
MAX_NEW = 60

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def template(msgs, gen_prompt):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt,
                                   return_tensors="pt", enable_thinking=False).flatten().to(device)


def gen(input_ids, cache=None):
    n = input_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=input_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=MAX_NEW, temperature=0.0)
    return tok.decode(out[0]).strip()


def capture_kv(token_ids):
    """RoPE'd per-layer K/V for token_ids at positions 0..L-1 (mode=generate stores them)."""
    cap = TrainableCache(config=attn)
    n = token_ids.shape[0]
    model(input_ids=token_ids, seq_ids=torch.zeros(n, dtype=torch.long, device=device),
          position_ids=torch.arange(n, device=device), use_cache=True,
          past_key_values=cap, mode="generate")
    keys = [cap._keys[l].detach().clone() for l in range(attn.n_layers)]
    vals = [cap._values[l].detach().clone() for l in range(attn.n_layers)]
    return keys, vals


# system block tokens (no generation prompt) vs full ceiling — split point = len(system block)
sys_ids = template([{"role": "system", "content": INSTRUCTION}], gen_prompt=False)
s = sys_ids.shape[0]
print(f"system block: {s} tokens -> {tok.decode(sys_ids)!r}\n")

keys, vals = capture_kv(sys_ids)
print(f"captured frozen KV: layer0 K {tuple(keys[0].shape)}\n")

for q in QUERIES:
    full = template([{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": q}], True)
    assert torch.equal(full[:s], sys_ids), "system block is not a clean prefix of the full prompt"
    rest = full[s:]  # [<|im_start|>user\n {q} <|im_end|>\n <|im_start|>assistant\n]

    frozen = TrainableCache(config=attn, init_keys=[k.clone() for k in keys],
                            init_values=[v.clone() for v in vals], num_frozen_tokens=s).to(device)
    ceil = gen(full)                      # normal in-context ceiling
    recon = gen(rest, cache=frozen)       # reconstruction: frozen system block + user input
    print(f"Q: {q}")
    print(f"  ceiling     : {ceil!r}")
    print(f"  frozen-recon: {recon!r}")
    print(f"  MATCH: {ceil[:40] == recon[:40]}\n")

print("If frozen-recon pirates like the ceiling (ideally near-identical), the frozen-KV "
      "placement mechanism is VALID and the placement sweep is unblocked.")
