"""Experiment: are cartridges linear-COMPOSITIONAL? (the concept-as-direction test)

Topics: A = giraffes, B = volcanoes (both AO-legible, distinct). Train cart_A, cart_B,
cart_AB (both), all length 4. Four probes:

  1. SEMANTIC   : AO reads A->giraffe, B->volcano, AB->both? (free-gen channel, the one that works)
  2. CONCAT     : stack A's slots ++ B's slots (no retraining) -> reads as both? (paper's "compose" claim)
  3. SUBSPACE + : is span(cart_AB) contained in span(cart_A, cart_B)?   (directions ADD)
  4. SUBSPACE - : project A's directions out of AB -> is the remainder ~ span(B)?  (directions SUBTRACT)

NOTE: we do NOT add carts slot-wise (slots have no canonical order: permutation/rotation/seed
freedom). The well-posed forms of "add/subtract" are concatenation (behavioral) and SUBSPACE
arithmetic (geometric, permutation-invariant). Small sizes (<=128 tokens) -> no OOM, eager is fine.
"""

import os
import sys
import gc

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
sys.path.insert(0, "/root/cartridge-interp/activation_oracles")

import torch
import torch.nn.functional as F
from peft import LoraConfig
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
AO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"
LAYER = 18
CART_LEN = 4
TOPIC_TOK = 64
STEPS = 400
LR = 2e-2
TOL = 1e-3
PATIENCE = 25
device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(0)

GIRAFFE = ("The giraffe is the tallest living land animal, recognized by its extraordinarily long "
           "neck and legs and its patchwork coat of irregular brown blotches. Native to the savannas "
           "of sub-Saharan Africa, it browses on acacia leaves high above other herbivores, using a "
           "long prehensile tongue to strip foliage from thorny branches without injury.")
VOLCANO = ("A volcano is a rupture in a planet's crust through which molten rock, ash, and gases "
           "escape from a chamber of magma below. When pressure builds beyond the strength of the "
           "overlying rock, an eruption hurls lava and pyroclastic debris across the land, and over "
           "many cycles the cooled deposits accumulate into towering conical mountains.")

# ====================== Phase A: FlexQwen3 (train + generate + geometry) ======================
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

torch.set_grad_enabled(True)
tok = AutoTokenizer.from_pretrained(MODEL)
flex = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
for p in flex.parameters():
    p.requires_grad = False

attn_config = AttnConfig(n_layers=flex.config.num_hidden_layers,
                         n_heads=flex.config.num_key_value_heads,
                         head_dim=flex.config.head_dim)
n_kv, head_dim = attn_config.n_heads, attn_config.head_dim
Wo = flex.model.layers[LAYER].self_attn.o_proj.weight.detach().float().cpu()  # [d_model, n_q*head_dim]
d_model = Wo.shape[0]
n_q = Wo.shape[1] // head_dim
group = n_q // n_kv

idsA = tok(GIRAFFE, return_tensors="pt").input_ids[0][:TOPIC_TOK].to(device)
idsB = tok(VOLCANO, return_tensors="pt").input_ids[0][:TOPIC_TOK].to(device)
idsAB = torch.cat([idsA, idsB])
print(f"tokens: A={idsA.shape[0]} B={idsB.shape[0]} AB={idsAB.shape[0]}")


def rand_vecs(K):
    return [torch.randn(1, n_kv, K, head_dim, dtype=dtype) * 0.1 for _ in range(attn_config.n_layers)]


def build_cart(keys, values):
    return TrainableCache(config=attn_config,
                          init_keys=[k.detach().to(device) for k in keys],
                          init_values=[v.detach().to(device) for v in values],
                          num_frozen_tokens=0).to(device)


def train_cart(ids, K=CART_LEN):
    cache = TrainableCache(config=attn_config, init_keys=rand_vecs(K),
                           init_values=rand_vecs(K), num_frozen_tokens=0).to(device)
    opt = torch.optim.Adam(cache.parameters(), lr=LR)
    L = ids.shape[0]
    seq_ids = torch.zeros(L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, device=device)
    below = 0
    for step in range(STEPS):
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out = flex(input_ids=ids, seq_ids=seq_ids, position_ids=position_ids,
                       use_cache=True, past_key_values=cache, mode="train")
            loss = F.cross_entropy(out.logits[0][:-1].float(), ids[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        below = below + 1 if loss.item() < TOL else 0
        if below >= PATIENCE:
            break
    return cache, loss.item()


cart_A, lA = train_cart(idsA)
cart_B, lB = train_cart(idsB)
cart_AB, lAB = train_cart(idsAB)
print(f"train final loss: A={lA:.4f} B={lB:.4f} AB={lAB:.4f}")

# concatenation: stack A's slots ++ B's slots (no retraining) -> length-8 cart
cart_CONCAT = build_cart(
    [torch.cat([cart_A.trainable_keys[l], cart_B.trainable_keys[l]], dim=2) for l in range(attn_config.n_layers)],
    [torch.cat([cart_A.trainable_values[l], cart_B.trainable_values[l]], dim=2) for l in range(attn_config.n_layers)],
)

torch.set_grad_enabled(False)


def gen_capture(cart, seed_ids, n_new):
    cart.clear()
    gen = flex_generate(model=flex, tokenizer=tok, input_ids=seed_ids,
                        seq_ids=torch.zeros(seed_ids.shape[0], dtype=torch.long, device=device),
                        position_ids=torch.arange(seed_ids.shape[0], device=device),
                        cache=cart, max_new_tokens=n_new, temperature=0.0)[0]
    full_ids = torch.cat([seed_ids, torch.tensor(gen, device=device)])
    Lf = full_ids.shape[0]
    grab = {}

    def hook(m, i, o):
        grab["h"] = o.hidden_states.detach()[0].cpu()

    cart.clear()
    hh = flex.model.layers[LAYER].register_forward_hook(hook)
    flex(input_ids=full_ids, seq_ids=torch.zeros(Lf, dtype=torch.long, device=device),
         position_ids=torch.arange(Lf, device=device), use_cache=True, past_key_values=cart, mode="generate")
    hh.remove()
    s = seed_ids.shape[0]
    return gen, grab["h"][s:]   # activations at generated positions


def recite(gen, target):
    n = min(len(gen), len(target))
    return sum(1 for i in range(n) if gen[i] == target[i]) / max(n, 1)


store = {}   # name -> {"text":..., "acts": [n,2560]}
genA, actsA = gen_capture(cart_A, idsA[:1], 60)
genB, actsB = gen_capture(cart_B, idsB[:1], 60)
genAB, actsAB = gen_capture(cart_AB, idsA[:1], 110)        # giraffe-seed; should recite A then B
genCC, actsCC = gen_capture(cart_CONCAT, idsA[:1], 110)
store["A giraffe"] = {"text": tok.decode(genA[:20]), "first": actsA[:16]}
store["B volcano"] = {"text": tok.decode(genB[:20]), "first": actsB[:16]}
store["AB both"] = {"text": tok.decode(genAB[:20]) + " ... " + tok.decode(genAB[-20:]),
                    "first": actsAB[:16], "last": actsAB[-16:]}
store["CONCAT A++B"] = {"text": tok.decode(genCC[:20]) + " ... " + tok.decode(genCC[-20:]),
                        "first": actsCC[:16], "last": actsCC[-16:]}
print(f"recite: A={recite(genA, idsA[1:].tolist()):.2f} B={recite(genB, idsB[1:].tolist()):.2f} "
      f"AB={recite(genAB, idsAB[1:].tolist()):.2f}")

# ---------- geometry: residual-write subspaces (W_O . V), MULTI-LAYER ----------
# Tightening of the original layer-18-only test. A single layer is a thin, noisy slice of
# where content lives; we now measure the subspace overlap at EVERY layer and aggregate,
# keeping the layer-18 number alongside so we can see how representative that slice was.
# (The energy-in-subspace metric is already invariant to rotation and slot permutation,
# so a Procrustes alignment would NOT change it -- and forcing one could only inflate the
# overlap. The honest fix is more layers, not an alignment.)
def write_matrix(cart, layer, Wo_l):
    V = cart.trainable_values[layer].detach()[0].float().cpu()   # [n_kv, K, head_dim]
    K = V.shape[1]
    rows = []
    for t in range(K):
        for h in range(n_q):
            g = h // group
            rows.append(Wo_l[:, h * head_dim:(h + 1) * head_dim] @ V[g, t])
    return torch.stack(rows)   # [n_q*K, d_model]


def orthobasis(M, rtol=1e-5):
    U, S, _ = torch.linalg.svd(M.t(), full_matrices=False)   # M.t(): [d_model, n]
    r = int((S > rtol * S[0]).sum())
    return U[:, :r], r


def energy_in(M, Q):
    return (((M @ Q) ** 2).sum() / (M ** 2).sum()).item()


def rand_energy(M, dim, n=3):   # chance overlap of M with a random dim-subspace of R^d_model
    return sum(energy_in(M, torch.linalg.qr(torch.randn(d_model, dim))[0]) for _ in range(n)) / n


def layer_geometry(layer):
    Wo_l = flex.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu()
    M_A = write_matrix(cart_A, layer, Wo_l)
    M_B = write_matrix(cart_B, layer, Wo_l)
    M_AB = write_matrix(cart_AB, layer, Wo_l)
    # ADD: is span(AB) inside span(A,B)?
    Q_AB_basis, r_ab = orthobasis(torch.cat([M_A, M_B]))
    add_frac = energy_in(M_AB, Q_AB_basis)
    add_base = rand_energy(M_AB, r_ab)
    # SUB: remove A's directions from AB, is the remainder ~ span(B)?
    Q_A, _ = orthobasis(M_A)
    R = M_AB - (M_AB @ Q_A) @ Q_A.t()
    Q_B, dim_B = orthobasis(M_B)
    sub_frac = energy_in(R, Q_B)
    sub_base = rand_energy(R, dim_B)
    return {"layer": layer, "r_ab": r_ab, "dim_B": dim_B,
            "add_frac": add_frac, "add_base": add_base,
            "sub_frac": sub_frac, "sub_base": sub_base}


per_layer = [layer_geometry(l) for l in range(attn_config.n_layers)]


def _mean(key):
    return sum(d[key] for d in per_layer) / len(per_layer)


L18 = next(d for d in per_layer if d["layer"] == LAYER)
geom = {
    "add_frac_mean": _mean("add_frac"), "add_base_mean": _mean("add_base"),
    "sub_frac_mean": _mean("sub_frac"), "sub_base_mean": _mean("sub_base"),
    "add_frac_18": L18["add_frac"], "add_base_18": L18["add_base"],
    "sub_frac_18": L18["sub_frac"], "sub_base_18": L18["sub_base"],
    "per_layer": per_layer,
}

del flex, cart_A, cart_B, cart_AB, cart_CONCAT
gc.collect(); torch.cuda.empty_cache()

# ====================== Phase B: AO reads ======================
import nl_probes.base_experiment as base_experiment
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation

tok2 = load_tokenizer(MODEL)
ao_model = load_model(MODEL, dtype, attn_implementation="sdpa")
ao_model.eval()
ao_model.add_adapter(LoraConfig(), adapter_name="default")
ao = base_experiment.load_lora_adapter(ao_model, AO_ID)
inj = get_hf_submodule(ao_model, 1)
QUESTION = "What is the text or topic represented by these activations? Answer in one short sentence."
GEN = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 40}


def ao_read(acts):
    vecs = acts.to(device).to(dtype)
    dp = create_training_datapoint(datapoint_type="probe", prompt=QUESTION, target_response="N/A",
                                   layer=LAYER, num_positions=vecs.shape[0], tokenizer=tok2,
                                   acts_BD=vecs, feature_idx=-1)
    return run_evaluation([dp], ao_model, tok2, inj, device, dtype, -1, ao, 1, 1.0, GEN)[0].api_response


print("\n================ COMPOSITIONALITY ================")
print("\n--- 1+2. SEMANTIC / CONCATENATION (AO reads the cart's generation) ---")
for name, d in store.items():
    print(f"\n[{name}]  gen: {d['text']!r}")
    print(f"  AO(first16): {ao_read(d['first'])}")
    if "last" in d:
        print(f"  AO(last16) : {ao_read(d['last'])}")

print(f"\n--- 3+4. SUBSPACE GEOMETRY (multi-layer: all {attn_config.n_layers} layers) ---")
pl = geom["per_layer"]
add_fracs = sorted(d["add_frac"] for d in pl)
sub_fracs = sorted(d["sub_frac"] for d in pl)
print("  test                          layer-18    multi-layer mean    chance")
print("  ADD  span(A,B) contains AB      %.3f          %.3f            %.3f" %
      (geom["add_frac_18"], geom["add_frac_mean"], geom["add_base_mean"]))
print("  SUB  (AB - A) lands in span(B)  %.3f          %.3f            %.3f" %
      (geom["sub_frac_18"], geom["sub_frac_mean"], geom["sub_base_mean"]))
print("  ADD per-layer  min %.3f / median %.3f / max %.3f" %
      (add_fracs[0], add_fracs[len(add_fracs) // 2], add_fracs[-1]))
print("  SUB per-layer  min %.3f / median %.3f / max %.3f" %
      (sub_fracs[0], sub_fracs[len(sub_fracs) // 2], sub_fracs[-1]))
print("  add_frac by depth (layer:frac):")
print("    " + "  ".join(f"{d['layer']}:{d['add_frac']:.2f}" for d in pl))
print("\nLift over chance (multi-layer): ADD %.1fx, SUB %.1fx." %
      (geom["add_frac_mean"] / max(geom["add_base_mean"], 1e-6),
       geom["sub_frac_mean"] / max(geom["sub_base_mean"], 1e-6)))
print("Multi-layer mean >> chance => content composes as (partially) shared linear subspaces.")
print("Compare layer-18 vs mean to see how representative the single-layer slice was.")
