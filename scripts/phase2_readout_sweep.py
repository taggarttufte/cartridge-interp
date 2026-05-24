"""Phase 2: do the two readouts REPLICATE across cart sizes? (kills the n=1 worry)

For each saved length-1 cart (trained on 32/64/128/256/512-token passages):
  (b) direct  : W_O*V at layer 18 -> AO reads it raw            [Exp 1 was NULL]
  (a') free-gen: seed 1 token, generate, AO reads the FIRST 16  [Exp 2b was POSITIVE]
                 generated tokens' layer-18 activations
                 (first-16 = same opening for every cart -> controlled comparison)

Phase A loads FlexQwen3 (generate + extract vectors), frees it, Phase B loads the AO.
Random-cart controls for both channels. Ceiling = AO on real passage activations.
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
from peft import LoraConfig
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
AO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"
TEXT = "/root/cartridge-interp/data/shadow_slave_v1.txt"
CART_TMPL = "/root/cartridge-interp/output/cart_len1_ss_p{}.pt"
LENGTHS = [32, 64, 128, 256, 512]
LAYER = 18
MAX_NEW = 40
READ_FIRST = 16
device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)
torch.manual_seed(0)

# ====================== Phase A: FlexQwen3 ======================
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

tok = AutoTokenizer.from_pretrained(MODEL)
flex = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
flex.eval()

attn_config = AttnConfig(n_layers=flex.config.num_hidden_layers,
                         n_heads=flex.config.num_key_value_heads,
                         head_dim=flex.config.head_dim)

# layer-18 projection weights for the direct (b) probe
attn = flex.model.layers[LAYER].self_attn
Wo = attn.o_proj.weight.detach().float().cpu()   # [d_model, n_q*head_dim] (direct-probe math on CPU)
Wq = attn.q_proj.weight.detach().float().cpu()   # [n_q*head_dim, d_model]
head_dim = attn_config.head_dim
n_kv = attn_config.n_heads
d_model = Wo.shape[0]
n_q = Wo.shape[1] // head_dim
group = n_q // n_kv

full = open(TEXT, encoding="utf-8").read()
start = full.find("A frail-looking young man")
all_ids = tok(full[start:start + 12000], return_tensors="pt").input_ids[0].to(device)
seed_ids = all_ids[:1]


def build_cart(keys, values):
    return TrainableCache(config=attn_config,
                          init_keys=[k.detach().to(device) for k in keys],
                          init_values=[v.detach().to(device) for v in values],
                          num_frozen_tokens=0).to(device)


def direct_vectors(K, V):
    """K,V: [n_kv, head_dim] at layer 18 (slot 0). Return write_kvhead [8,d], write_qhead [32,d]."""
    K = K.float().cpu(); V = V.float().cpu()
    wq, wk = [], []
    for h in range(n_q):
        g = h // group
        wq.append(Wo[:, h * head_dim:(h + 1) * head_dim] @ V[g])   # write, per q-head
    for g in range(n_kv):
        acc = torch.zeros(d_model)
        for h in range(g * group, (g + 1) * group):
            acc = acc + Wo[:, h * head_dim:(h + 1) * head_dim] @ V[g]
        wk.append(acc)
    return torch.stack(wk), torch.stack(wq)


def gen_and_capture(cart):
    cart.clear()
    gen = flex_generate(model=flex, tokenizer=tok, input_ids=seed_ids,
                        seq_ids=torch.zeros(1, dtype=torch.long, device=device),
                        position_ids=torch.arange(1, device=device),
                        cache=cart, max_new_tokens=MAX_NEW, temperature=0.0)[0]
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
    return gen, grab["h"][1:1 + READ_FIRST]   # FIRST generated-token activations


store = {}   # length -> dict(write_kvhead, write_qhead, gen_acts, recite_acc)
for n_ctx in LENGTHS:
    ckpt = torch.load(CART_TMPL.format(n_ctx), map_location=device, weights_only=False)
    cart = build_cart(ckpt["trainable_keys"], ckpt["trainable_values"])
    K = ckpt["trainable_keys"][LAYER].detach()[0, :, 0]   # [n_kv, head_dim]
    V = ckpt["trainable_values"][LAYER].detach()[0, :, 0]
    wk, wq = direct_vectors(K, V)
    gen, gen_acts = gen_and_capture(cart)
    target = all_ids[1:n_ctx].tolist()
    n = min(len(gen), len(target))
    recite = sum(1 for i in range(n) if gen[i] == target[i]) / max(n, 1)
    store[n_ctx] = {"wk": wk, "wq": wq, "acts": gen_acts, "recite": recite}
    print(f"len {n_ctx}: recite(first {n}) {recite:.2f}  gen0={tok.decode(gen[:12])!r}")
    del cart
    gc.collect(); torch.cuda.empty_cache()

# random-cart controls
rkeys = [torch.randn(1, n_kv, 1, head_dim, dtype=dtype) * 0.1 for _ in range(attn_config.n_layers)]
rvals = [torch.randn(1, n_kv, 1, head_dim, dtype=dtype) * 0.1 for _ in range(attn_config.n_layers)]
rcart = build_cart(rkeys, rvals)
rgen, racts = gen_and_capture(rcart)
rK = rkeys[LAYER].detach()[0, :, 0]; rV = rvals[LAYER].detach()[0, :, 0]
rwk, rwq = direct_vectors(rK, rV)
store["RAND"] = {"wk": rwk, "wq": rwq, "acts": racts, "recite": 0.0}
print(f"RAND: gen0={tok.decode(rgen[:12])!r}")

del flex, rcart
gc.collect(); torch.cuda.empty_cache()

# ====================== Phase B: AO ======================
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
    res = run_evaluation([dp], ao_model, tok2, inj, device, dtype, -1, ao, 1, 1.0, GEN)
    return res[0].api_response


print("\n================ PHASE 2 READOUT REPLICATION ================")
print("Ground truth: Shadow Slave opening - a young man outside a police station.\n")
for key in LENGTHS + ["RAND"]:
    d = store[key]
    tag = f"len {key}" if key != "RAND" else "RANDOM cart"
    print(f"--- {tag}  (recite {d['recite']:.2f}) ---")
    print(f"  (b) direct W_O*V kvhead : {ao_read(d['wk'])}")
    print(f"  (a') free-gen first{READ_FIRST}  : {ao_read(d['acts'])}")
print("\nCeiling (real passage acts): 'a young man on a bench across from a police station'")
