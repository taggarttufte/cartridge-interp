"""Does a recitation cart carry BEHAVIOR or just SURFACE TOKENS?

Train a length-1 cart by naive next-token recitation on a short INSTRUCTION string
("respond like a pirate", "respond only as a question"). Then test, on held-out
neutral queries, whether loading the cart makes the model *act on* the instruction
— vs merely reciting it.

Hypothesis (Tagg): base model recites the instruction flawlessly but cannot act on
it; the instruct model (assistant-tuned) acts on the cart-stored instruction.

Run per model:  python instruction_cart.py <MODEL_ID> <TAG>
  e.g. ... Qwen/Qwen3-4B instruct
       ... Qwen/Qwen3-4B-Base base

Conditions per (instruction, query):
  baseline  : query only, no cart            -> behavior should be ~absent
  incontext : instruction + query, no cart   -> CEILING (can this model follow it at all?)
  cart      : query only, cart loaded        -> THE TEST (does the cart transmit behavior?)
Plus per cart: free-recitation accuracy (does the cart "contain" the string?).
"""
import os, sys, json

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"   # eager flex-attn; skip 30-min autotune

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
TAG   = sys.argv[2] if len(sys.argv) > 2 else "instruct"
device = "cuda"
torch.manual_seed(0)

INSTRUCTIONS = {
    "pirate":   "Always respond like a pirate. Use words like arr, matey, and ahoy.",
    "question": "Always respond only in the form of a question, never a statement.",
}
QUERIES = [
    "What is the capital of France?",
    "Tell me about the weather.",
    "How do I make a sandwich?",
    "Describe a cat.",
    "Explain how plants grow.",
    "What time is it?",
]
STEPS, LR, MAX_NEW = 250, 2e-2, 50

# ---- scoring heuristics (cheap; raw outputs are also dumped for eyeballing) ----
PIRATE_WORDS = ["arr", "matey", "ahoy", " ye ", "aye", "yer ", "scurvy", "landlubber",
                "treasure", "me hearty", "avast", "shiver", "booty", "cap'n"]
def pirate_hits(t):
    tl = " " + t.lower() + " "
    return sum(tl.count(w) for w in PIRATE_WORDS)
def is_question(t):
    t = t.strip()
    return t.endswith("?") or ("?" in t and len(t.split("?")[0]) < len(t) * 0.8)

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads,
                  head_dim=model.config.head_dim)

def rand_vecs(n):
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]

def train_cart(instr_ids, steps=STEPS):
    cache = TrainableCache(config=attn, init_keys=rand_vecs(1),
                           init_values=rand_vecs(1), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    L = instr_ids.shape[0]
    sid = torch.zeros(L, dtype=torch.long, device=device)
    pos = torch.arange(L, device=device)
    torch.set_grad_enabled(True)
    loss = acc = None
    for step in range(steps):
        cache.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=instr_ids, seq_ids=sid, position_ids=pos,
                        use_cache=True, past_key_values=cache, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), instr_ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.set_grad_enabled(False)
    cache.clear()
    acc = (logits[:-1].argmax(-1) == instr_ids[1:]).float().mean().item()
    return cache, loss.item(), acc

def gen(prompt_ids, cache=None):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(
        model=model, tokenizer=tok, input_ids=prompt_ids,
        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
        position_ids=torch.arange(n, device=device),
        cache=cache, max_new_tokens=MAX_NEW, temperature=0.0,
    )
    if cache is not None:
        cache.clear()
    return tok.decode(out[0]).strip()

def ids_of(s):
    return tok(s, return_tensors="pt").input_ids[0].to(device)

results = {}
for iname, instr in INSTRUCTIONS.items():
    print(f"\n{'='*70}\n[{TAG}] INSTRUCTION '{iname}': {instr!r}\n{'='*70}", flush=True)
    instr_ids = ids_of(instr)
    cart, final_loss, tf_acc = train_cart(instr_ids)

    # free-recitation: seed with first 2 tokens, does it reproduce the instruction?
    seed = instr_ids[:2]
    rec = gen(seed, cache=cart)  # cart cleared inside
    print(f"  cart trained: final_loss={final_loss:.4f} tf_acc={tf_acc:.3f}")
    print(f"  free-recite (seed={tok.decode(seed)!r}): {rec!r}")

    scorer = pirate_hits if iname == "pirate" else (lambda t: int(is_question(t)))
    rows = []
    for q in QUERIES:
        base = gen(ids_of(q + "\n"))
        ctx  = gen(ids_of(instr + "\n\n" + q + "\n"))
        crt  = gen(ids_of(q + "\n"), cache=cart)
        rows.append(dict(query=q, baseline=base, incontext=ctx, cart=crt,
                         s_base=scorer(base), s_ctx=scorer(ctx), s_cart=scorer(crt)))
        print(f"\n  Q: {q}")
        print(f"    baseline  [{scorer(base)}]: {base!r}")
        print(f"    incontext [{scorer(ctx)}]: {ctx!r}")
        print(f"    cart      [{scorer(crt)}]: {crt!r}")

    agg = lambda k: sum(r[k] for r in rows)
    print(f"\n  >>> [{TAG}/{iname}] score totals  baseline={agg('s_base')}  "
          f"incontext(ceiling)={agg('s_ctx')}  cart={agg('s_cart')}")
    results[iname] = dict(instruction=instr, final_loss=final_loss, tf_acc=tf_acc,
                          recite=rec, rows=rows,
                          totals=dict(baseline=agg('s_base'), incontext=agg('s_ctx'),
                                      cart=agg('s_cart')))

outp = f"/root/cartridge-interp/output/instruction_cart_{TAG}.json"
with open(outp, "w") as f:
    json.dump(dict(model=MODEL, tag=TAG, results=results), f, indent=2)
print(f"\nsaved -> {outp}")
