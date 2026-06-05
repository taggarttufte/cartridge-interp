"""Better recitation elicitation: ASK the model to recite the cart, instead of seeding tokens.

Critique of the old metrics:
  - teacher-forced acc spoon-feeds the correct prefix at every step (flatters).
  - free-recitation (seed first 2 tokens, generate) only tests "what naturally continues this
    prefix" — not "reproduce what's stored." A doc needn't repeat a sentence just because a later
    sentence starts the same way.
New method: load the cart as prefix, then put a natural-language RECITATION PROMPT after it asking
the model to recite the cart's contents, and generate. Run on base + instruct.

Cart is trained (len-1 recitation) on a distinctive NON-copyrighted target so any reproduction is
unambiguously from the cart. Carts are weight-specific, so we train a fresh cart per model.
Run: python scripts/prompted_recite.py
"""
import os, sys, difflib

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

device = "cuda"
torch.manual_seed(0)

TARGET = ("Dr. Mira Voss discovered the coral Lumicorallium veridis off Tasmania in 2014, "
          "aboard her vessel the Selkie.")
DISTINCTIVE = ["lumicorallium", "selkie", "tasmania", "2014", "mira voss"]
RECITE_PROMPT = (" FYI, the context before this sentence is a \"cartridge\" – a synthetic cache of "
                 "KV vectors. Please recite its contents as faithfully as you can.")
STEPS, LR, MAXNEW = 250, 2e-2, 60

MODELS = [("Qwen/Qwen3-4B", "instruct"), ("Qwen/Qwen3-4B-Base", "base")]

def score(out):
    o = out.lower()
    ratio = difflib.SequenceMatcher(None, o, TARGET.lower()).ratio()
    hits = sum(k in o for k in DISTINCTIVE)
    return ratio, hits

for MODEL, tag in MODELS:
    print(f"\n{'='*72}\n{MODEL}  ({tag})\n{'='*72}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                          load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                      n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)

    def ids(s, special=True):
        return tok(s, return_tensors="pt", add_special_tokens=special).input_ids[0].to(device)

    def gen(prompt_ids, cache=None, maxnew=MAXNEW):
        if cache is not None: cache.clear()
        n = prompt_ids.shape[0]
        out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                            seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                            position_ids=torch.arange(n, device=device),
                            cache=cache, max_new_tokens=maxnew, temperature=0.0)
        if cache is not None: cache.clear()
        return tok.decode(out[0]).strip()

    # train a len-1 recitation cart on TARGET
    tids = ids(TARGET)
    def rv(n):
        return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
                for _ in range(attn.n_layers)]
    cart = TrainableCache(config=attn, init_keys=rv(1), init_values=rv(1), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cart.parameters(), lr=LR)
    L = tids.shape[0]
    torch.set_grad_enabled(True)
    for step in range(STEPS):
        cart.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=tids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                        position_ids=torch.arange(L, device=device), use_cache=True,
                        past_key_values=cart, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), tids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.set_grad_enabled(False)
    tf_acc = (logits[:-1].argmax(-1) == tids[1:]).float().mean().item()
    print(f"  target: {TARGET!r}")
    print(f"  cart trained: tf_acc={tf_acc:.3f}, final_loss={loss.item():.4f}")

    # --- method 1: free-recitation (seed first 2 tokens) ---
    fr = gen(tids[:2], cache=cart)
    r, h = score(fr); print(f"\n  [free-recite, seed=2]  sim={r:.2f} distinctive={h}/{len(DISTINCTIVE)}\n    {fr!r}")
    # --- method 2: PROMPTED recitation (cart + ask-to-recite) ---
    pr = gen(ids(RECITE_PROMPT), cache=cart)
    r, h = score(pr); print(f"\n  [PROMPTED recite]      sim={r:.2f} distinctive={h}/{len(DISTINCTIVE)}\n    {pr!r}")
    # --- control: same recite prompt with NO cart (does the prompt alone leak anything?) ---
    nc = gen(ids(RECITE_PROMPT), cache=None)
    r, h = score(nc); print(f"\n  [prompt, NO cart]      sim={r:.2f} distinctive={h}/{len(DISTINCTIVE)}\n    {nc!r}")

    del model
    torch.cuda.empty_cache()

print("\nDone.")
