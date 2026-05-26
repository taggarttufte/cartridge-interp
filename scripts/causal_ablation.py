"""Causal ablation: project a topic's write-subspace out of cart_AB and ask
whether that topic's content vanishes while the other survives. The flagship
causal test -- the gap vs the correlational cartridge-interp literature.

Pipeline:
  - Train cart_A (giraffe), cart_B (volcano), cart_AB (both) -- same setup as
    compositionality.py (Qwen3-4B, length-4 carts, 64-token topic passages).
  - Per layer L: compute Q_A^L = orthobasis(M_A^L), Q_B^L = orthobasis(M_B^L),
    and Q_rand^L = a random orthonormal subspace of the same dim as Q_A^L.
  - Build three ablated copies of cart_AB by modifying V so that for each q-head,
    the residual write (W_O[:, h_block] @ V[g, t]) is projected onto (I - Q Q^T).
    Per (layer, kv_group, slot) we solve a least-squares problem because one V
    feeds `group` q-heads (GQA): find V_new minimizing the per-q-head residual.
    Keys are unchanged -- only the write directions move.
  - Free-generate from each ablated cart (giraffe-seed, n_new=128), capture
    layer-18 activations, then read the AO on first-16 (giraffe region) +
    last-16 (volcano region).
  - Measure giraffe-recite vs idsA[1:64] at positions 0..62 of the generation,
    and volcano-recite as the best 64-window match anywhere in the generation
    (sliding window handles a shifted transition under ablation).

Predictions:
  ablate_A    -> giraffe drops, volcano survives
  ablate_B    -> giraffe survives, volcano drops
  ablate_rand -> both survive (controls for methodology disruption)

Composition was only ~49% ADD multi-layer, so cart_AB has giraffe directions
*outside* span(M_A); ablation may be partial. That's a measurement, not a bug.
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
N_GEN = 128
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

# ====================== Phase A: FlexQwen3 (train + ablate + generate) ======================
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
n_layers = attn_config.n_layers
# any-layer Wo to derive d_model / n_q / group
_wo0 = flex.model.layers[0].self_attn.o_proj.weight.detach().float().cpu()
d_model = _wo0.shape[0]
n_q = _wo0.shape[1] // head_dim
group = n_q // n_kv
del _wo0

idsA = tok(GIRAFFE, return_tensors="pt").input_ids[0][:TOPIC_TOK].to(device)
idsB = tok(VOLCANO, return_tensors="pt").input_ids[0][:TOPIC_TOK].to(device)
idsAB = torch.cat([idsA, idsB])
print(f"tokens: A={idsA.shape[0]} B={idsB.shape[0]} AB={idsAB.shape[0]}")


def rand_vecs(K):
    return [torch.randn(1, n_kv, K, head_dim, dtype=dtype) * 0.1 for _ in range(n_layers)]


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
    for _ in range(STEPS):
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
torch.set_grad_enabled(False)

# ---------- per-layer write-subspaces (Q_A, Q_B, Q_rand) ----------
Wo_per_layer = [flex.model.layers[L].self_attn.o_proj.weight.detach().float().cpu()
                for L in range(n_layers)]


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
    U, S, _ = torch.linalg.svd(M.t(), full_matrices=False)
    r = int((S > rtol * S[0]).sum())
    return U[:, :r]


Q_A_per_layer = [orthobasis(write_matrix(cart_A, L, Wo_per_layer[L])) for L in range(n_layers)]
Q_B_per_layer = [orthobasis(write_matrix(cart_B, L, Wo_per_layer[L])) for L in range(n_layers)]
Q_rand_per_layer = [torch.linalg.qr(torch.randn(d_model, Q_A_per_layer[L].shape[1]))[0]
                    for L in range(n_layers)]
print(f"per-layer Q_A dims: min {min(q.shape[1] for q in Q_A_per_layer)} / "
      f"max {max(q.shape[1] for q in Q_A_per_layer)}")


# ---------- ablation: project the write through (I - Q Q^T), solve LS for V_new ----------
def ablate_cart(source_cart, Q_per_layer):
    new_keys = [k.detach().clone() for k in source_cart.trainable_keys]
    new_values = []
    for L in range(n_layers):
        Q = Q_per_layer[L].float()                                       # [d_model, r]
        V = source_cart.trainable_values[L].detach()[0].float().cpu()    # [n_kv, K, head_dim]
        Wo_L = Wo_per_layer[L]
        K_slots = V.shape[1]
        V_new = torch.zeros_like(V)
        for g in range(n_kv):
            q_start = g * group
            # [group, d_model, head_dim] -- the W_O blocks for the q-heads in this kv-group
            Wo_grp = Wo_L[:, q_start * head_dim:(q_start + group) * head_dim].reshape(
                d_model, group, head_dim).permute(1, 0, 2).contiguous()
            # writes[h, :, k] = Wo_grp[h] @ V[g, k]
            writes = torch.einsum("hdc,kc->hdk", Wo_grp, V[g])           # [group, d_model, K]
            # ablate: writes_abl = writes - Q (Q^T writes)
            QtW = torch.einsum("dr,hdk->hrk", Q, writes)
            writes_abl = writes - torch.einsum("dr,hrk->hdk", Q, QtW)
            # solve W_stack V_new[g,:] = writes_abl_stack
            W_stack = Wo_grp.reshape(group * d_model, head_dim)
            target = writes_abl.reshape(group * d_model, K_slots)
            sol = torch.linalg.lstsq(W_stack, target).solution           # [head_dim, K_slots]
            V_new[g] = sol.t()
        new_values.append(V_new.unsqueeze(0).to(device).to(dtype))
    return build_cart(new_keys, new_values)


print("building ablated carts ...")
cart_abl_A = ablate_cart(cart_AB, Q_A_per_layer)
cart_abl_B = ablate_cart(cart_AB, Q_B_per_layer)
cart_abl_R = ablate_cart(cart_AB, Q_rand_per_layer)
print("built: cart_abl_A, cart_abl_B, cart_abl_R")


# ---------- gen + capture (layer-LAYER activations) ----------
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
         position_ids=torch.arange(Lf, device=device), use_cache=True,
         past_key_values=cart, mode="generate")
    hh.remove()
    s = seed_ids.shape[0]
    return gen, grab["h"][s:]


def prefix_match(gen, target):
    n = min(len(gen), len(target))
    return sum(1 for i in range(n) if gen[i] == target[i]) / max(n, 1)


def best_window_match(gen, target):
    L = len(target)
    if len(gen) < L:
        return 0.0
    best = 0
    for s in range(len(gen) - L + 1):
        c = sum(1 for i in range(L) if gen[s + i] == target[i])
        if c > best:
            best = c
    return best / L


print("generating from each cart ...")
g_target = idsA[1:64].tolist()         # 63 tokens
v_target = idsB[:64].tolist()          # 64 tokens
runs = []
for name, cart in [("cart_AB (no ablation)", cart_AB),
                   ("cart_AB ablate_A (giraffe out)", cart_abl_A),
                   ("cart_AB ablate_B (volcano out)", cart_abl_B),
                   ("cart_AB ablate_rand (control)", cart_abl_R)]:
    gen, acts = gen_capture(cart, idsA[:1], N_GEN)
    g_acc = prefix_match(gen[:63], g_target)
    v_acc = best_window_match(gen, v_target)
    runs.append({"name": name, "gen": gen, "acts": acts, "g_acc": g_acc, "v_acc": v_acc,
                 "first16": acts[:16], "last16": acts[-16:],
                 "text_first": tok.decode(gen[:30]), "text_last": tok.decode(gen[-30:])})

# free the model before AO load
del flex, cart_A, cart_B, cart_AB, cart_abl_A, cart_abl_B, cart_abl_R
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


for r in runs:
    r["ao_first"] = ao_read(r["first16"])
    r["ao_last"] = ao_read(r["last16"])

print("\n================ CAUSAL ABLATION (cart_AB, giraffe-seed, 128 new tokens) ================\n")
print(f"{'cart':<36} {'giraffe-recite':>15}  {'volcano-best-window':>22}")
for r in runs:
    print(f"  {r['name']:<34} {r['g_acc']:>14.3f}  {r['v_acc']:>21.3f}")

print("\n--- AO reads (first16 / last16) ---")
for r in runs:
    print(f"\n[{r['name']}]")
    print(f"  gen[0:30]  : {r['text_first']!r}")
    print(f"  gen[-30:]  : {r['text_last']!r}")
    print(f"  AO first16 : {r['ao_first']}")
    print(f"  AO last16  : {r['ao_last']}")

print("\n--- Interpretation ---")
print("If ablate_A drops giraffe-recite while volcano-best-window holds (and vice versa for ablate_B),")
print("and ablate_rand preserves both: causally, the topic subspaces carry their respective content.")
print("Partial drops are expected (composition was ~49% multi-layer -- cart_AB has giraffe directions")
print("outside span(M_A), so a span(M_A) ablation can only suppress the shared component).")
