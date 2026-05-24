"""Topic-focus: does cart compression exploit STRUCTURE, or just rote-store?

A cart only needs to store the DELTA between the corpus and what the frozen base model would
already predict. So low-surprise (coherent) text should compress EASIER than high-surprise
(incoherent / random) text. Four conditions, all 512 tokens, cart length 1:

  A coherent     : Shadow Slave passage              (full structure)
  B unrelated    : 7 disjoint-domain paragraphs      (locally coherent, globally incoherent)
  C shuffled-A   : A's exact tokens, random order    (same vocab, zero sequential structure)
  D random       : uniform random vocab ids          (no structure -- the floor)

Differentiator = STEPS-TO-CONVERGE and FINAL LOSS (accuracy may saturate at 1.0 for all).
Prediction: difficulty A < B < C < D.
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
N_CTX = 512
STEPS = 400
LR = 2e-2
device = "cuda"
torch.manual_seed(0)

UNRELATED = """
This Agreement shall be governed by and construed in accordance with the laws of the State of
Delaware, without regard to its conflict of laws principles. Each party irrevocably submits to
the exclusive jurisdiction of the state and federal courts located therein for the resolution of
any dispute arising out of or relating to this Agreement or its subject matter.
To prepare the risotto, warm the stock and keep it at a gentle simmer. In a heavy pan, saute
finely chopped onion in butter until translucent, then add the arborio rice and toast for two
minutes. Add a ladle of stock at a time, stirring constantly until each addition is absorbed
before adding the next, until the rice is creamy yet still firm to the bite.
The visitors struck first in the second quarter with a long drive capped by a short rushing
touchdown. The home side answered before halftime with a field goal, then seized control in the
third quarter on a forced fumble that set up the go-ahead score. The defense held firm in the
closing minutes to preserve a narrow road victory.
The function accepts an iterable of integers and returns a dictionary mapping each value to its
frequency. It iterates once over the input, incrementing a counter for each element, which yields
linear time complexity. An empty iterable returns an empty dictionary, and non-hashable elements
raise a type error at insertion time.
A white dwarf is the dense remnant left when a low-mass star exhausts its nuclear fuel and sheds
its outer layers. Supported against gravity by electron degeneracy pressure, it can pack a solar
mass into a volume comparable to Earth. If it accretes enough matter from a companion to exceed
the Chandrasekhar limit, it may detonate as a type Ia supernova.
The printing press, introduced to Europe in the mid-fifteenth century, sharply lowered the cost
of reproducing texts and accelerated the spread of ideas. Within decades, books that had once
taken months to copy by hand could be produced in quantity, fueling literacy, religious reform,
and the scientific revolution that followed.
In sonata-allegro form, the exposition presents two contrasting themes, typically in the tonic
and dominant keys. The development fragments and recombines this material through a series of
modulations, building tension before the recapitulation returns both themes in the home key to
resolve the movement.
During photosynthesis, plants capture light energy in chloroplasts and use it to convert carbon
dioxide and water into glucose and oxygen. The light-dependent reactions split water and generate
ATP and NADPH, which the Calvin cycle then uses to fix carbon into sugars that fuel growth and are
stored as starch for later use.
A central bank adjusts short-term interest rates to balance the competing goals of stable prices
and full employment. Raising rates cools borrowing and spending to restrain inflation, while
cutting rates stimulates demand during a downturn, though the effects reach the broader economy
only with long and variable lags.
"""

print("Loading FlexQwen3-4B...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

vocab = model.config.vocab_size
full = open(TEXT, encoding="utf-8").read()
start = full.find("A frail-looking young man")
A = tok(full[start:start + 6000], return_tensors="pt").input_ids[0][:N_CTX].to(device)
B = tok(UNRELATED.strip(), return_tensors="pt").input_ids[0][:N_CTX].to(device)
C = A[torch.randperm(N_CTX, device=device)]
D = torch.randint(0, vocab, (N_CTX,), device=device)
assert B.shape[0] == N_CTX, f"unrelated text too short: {B.shape[0]} tokens"
CONDS = {"A coherent": A, "B unrelated": B, "C shuffled-A": C, "D random": D}

attn_config = AttnConfig(n_layers=model.config.num_hidden_layers,
                         n_heads=model.config.num_key_value_heads,
                         head_dim=model.config.head_dim)


def rand_vecs():
    return [torch.randn(1, attn_config.n_heads, 1, attn_config.head_dim,
                        dtype=torch.bfloat16) * 0.1 for _ in range(attn_config.n_layers)]


def run(name, ids):
    L = ids.shape[0]
    cache = TrainableCache(config=attn_config, init_keys=rand_vecs(),
                           init_values=rand_vecs(), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    seq_ids = torch.zeros(L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, dtype=torch.long, device=device)
    s_05, s_001 = None, None
    print(f"\n=== {name} ===")
    for step in range(STEPS):
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
                        use_cache=True, past_key_values=cache, mode="train")
            logits = out.logits[0]
            loss = F.cross_entropy(logits[:-1].float(), ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item()
        if s_05 is None and lv < 0.05:
            s_05 = step
        if s_001 is None and lv < 0.01:
            s_001 = step
        if step % 50 == 0 or step == STEPS - 1:
            print(f"  step {step:>3}: loss {lv:.4f}", flush=True)
    tf_acc = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()
    final_loss = lv

    cache.clear()
    gen = flex_generate(model=model, tokenizer=tok, input_ids=ids[:1],
                        seq_ids=torch.zeros(1, dtype=torch.long, device=device),
                        position_ids=torch.arange(1, device=device),
                        cache=cache, max_new_tokens=L - 1, temperature=0.0)[0]
    tgt = ids[1:].tolist()
    n = min(len(gen), len(tgt))
    recite = sum(1 for i in range(n) if gen[i] == tgt[i]) / max(n, 1)

    del cache, opt
    gc.collect(); torch.cuda.empty_cache()
    return {"final_loss": final_loss, "tf_acc": tf_acc, "recite": recite,
            "steps_to_0.05": s_05, "steps_to_0.01": s_001}


rows = {name: run(name, ids) for name, ids in CONDS.items()}

print("\n================ TOPIC-FOCUS: structure vs compression difficulty ================")
print(f"(all {N_CTX} tokens, cart length 1, {STEPS} steps, lr {LR})\n")
print(f"{'condition':<14}{'final_loss':>11}{'tf_acc':>8}{'recite':>8}{'->0.05':>8}{'->0.01':>8}")
for name, r in rows.items():
    s05 = r['steps_to_0.05'] if r['steps_to_0.05'] is not None else '--'
    s01 = r['steps_to_0.01'] if r['steps_to_0.01'] is not None else '--'
    print(f"{name:<14}{r['final_loss']:>11.4f}{r['tf_acc']:>8.3f}{r['recite']:>8.3f}{str(s05):>8}{str(s01):>8}")
print("\nPrediction: difficulty (steps-to-converge, final loss) A < B < C < D.")
