"""Experiment 2b: free-generation readout — the weak-cart vs wrong-position disentangler.

Seed only the first passage token, let the cart GENERATE (free-recitation check),
then read the layer-18 activations of the GENERATED tokens with the AO. Those tokens
(and their content) come from the cart, not from us. Random cart = baseline.
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
CART = "/root/cartridge-interp/output/cart_len1_ss.pt"
TEXT = "/root/cartridge-interp/data/shadow_slave_v1.txt"
LAYER = 18
SEED = 1
MAX_NEW = 40
device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)
torch.manual_seed(0)

# ============ Step 1: generate from cart + capture generated-token activations ============
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

tok = AutoTokenizer.from_pretrained(MODEL)
flex = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
flex.eval()

ckpt = torch.load(CART, map_location=device, weights_only=False)
attn_config = AttnConfig(n_layers=flex.config.num_hidden_layers,
                         n_heads=flex.config.num_key_value_heads,
                         head_dim=flex.config.head_dim)


def build_cart(keys, values):
    return TrainableCache(
        config=attn_config,
        init_keys=[k.detach().to(device) for k in keys],
        init_values=[v.detach().to(device) for v in values],
        num_frozen_tokens=0,
    ).to(device)


shadow = build_cart(ckpt["trainable_keys"], ckpt["trainable_values"])
rand = build_cart([torch.randn_like(k.detach()) * 0.1 for k in ckpt["trainable_keys"]],
                  [torch.randn_like(v.detach()) * 0.1 for v in ckpt["trainable_values"]])

full_text = open(TEXT, encoding="utf-8").read()
start = full_text.find("A frail-looking young man")
pass_ids = tok(full_text[start:start + 2000], return_tensors="pt").input_ids[0][:32].to(device)
seed_ids = pass_ids[:SEED]


def gen_and_capture(cart):
    cart.clear()
    out = flex_generate(
        model=flex, tokenizer=tok, input_ids=seed_ids,
        seq_ids=torch.zeros(SEED, dtype=torch.long, device=device),
        position_ids=torch.arange(SEED, device=device),
        cache=cart, max_new_tokens=MAX_NEW, temperature=0.0,
    )
    gen = out[0]
    full_ids = torch.cat([seed_ids, torch.tensor(gen, device=device)])
    Lf = full_ids.shape[0]
    grab = {}

    def hook(m, i, o):
        grab["h"] = o.hidden_states.detach()[0].cpu()

    cart.clear()
    h = flex.model.layers[LAYER].register_forward_hook(hook)
    flex(input_ids=full_ids, seq_ids=torch.zeros(Lf, dtype=torch.long, device=device),
         position_ids=torch.arange(Lf, device=device), use_cache=True, past_key_values=cart, mode="generate")
    h.remove()
    return gen, grab["h"][SEED:]   # activations at generated positions


shadow_gen, shadow_acts = gen_and_capture(shadow)
rand_gen, rand_acts = gen_and_capture(rand)

# free-recitation accuracy (shadow): generated vs the true passage continuation
target = pass_ids[SEED:].tolist()
n = min(len(shadow_gen), len(target))
recite_acc = sum(1 for i in range(n) if shadow_gen[i] == target[i]) / max(n, 1)

print(f"seed: {tok.decode(seed_ids)!r}")
print(f"shadow free-recitation acc vs passage: {recite_acc:.2f}  (over {n} tokens)")
print(f"shadow generated: {tok.decode(shadow_gen)!r}")
print(f"random generated: {tok.decode(rand_gen)!r}")
print(f"shadow acts {tuple(shadow_acts.shape)}, random acts {tuple(rand_acts.shape)}")

del flex, shadow, rand
gc.collect()
torch.cuda.empty_cache()

# ============ Step 2: AO reads the generated-token activations ============
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


def ao_read(name, acts):
    vecs = acts.to(device).to(dtype)
    if vecs.shape[0] > 16:
        vecs = vecs[-16:]
    dp = create_training_datapoint(
        datapoint_type="probe", prompt=QUESTION, target_response="N/A",
        layer=LAYER, num_positions=vecs.shape[0], tokenizer=tok2, acts_BD=vecs, feature_idx=-1,
    )
    res = run_evaluation([dp], ao_model, tok2, inj, device, dtype, -1, ao, 1, 1.0, GEN)
    print(f"\n[{name}]\n  AO: {res[0].api_response}")


print("\n================ FREE-GENERATION READOUT ================")
ao_read("SHADOW cart (generated tokens)", shadow_acts)
ao_read("RANDOM cart (generated tokens)", rand_acts)
print("\nCeiling (real passage activations): 'a young man on a bench across from a police station'")
