"""Experiment 2 / approach (a): read the cart's EFFECT via real activations.

Run FlexQwen3 with the cart as context on a neutral prompt, capture the genuine
layer-18 hidden states the cart induces, then have the AO read them. A random cart
is the baseline: any Shadow-Slave signal in the trained-cart reading (but not the
random-cart reading) means the cart's content surfaced in real activations.

Two 4B models don't fit at once, so we capture (FlexQwen3) -> free GPU -> read (AO).
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import sys
import gc

sys.path.insert(0, "/root/cartridge-interp/activation_oracles")

import torch
from peft import LoraConfig
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
AO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"
CART = "/root/cartridge-interp/output/cart_len1_ss.pt"
LAYER = 18
PROMPT = "Summarize the document above in one sentence:"
device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)
torch.manual_seed(0)

# ============ Step 1: capture cart-induced layer-18 activations (FlexQwen3) ============
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache

tok = AutoTokenizer.from_pretrained(MODEL)
flex = HFModelConfig(
    pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
    load_kwargs={"torch_dtype": torch.bfloat16},
).instantiate().to(device)
flex.eval()

ckpt = torch.load(CART, map_location=device, weights_only=False)
attn_config = AttnConfig(
    n_layers=flex.config.num_hidden_layers,
    n_heads=flex.config.num_key_value_heads,
    head_dim=flex.config.head_dim,
)


def build_cart(keys, values):
    return TrainableCache(
        config=attn_config,
        init_keys=[k.detach().to(device) for k in keys],
        init_values=[v.detach().to(device) for v in values],
        num_frozen_tokens=0,
    ).to(device)


shadow_cart = build_cart(ckpt["trainable_keys"], ckpt["trainable_values"])
rand_cart = build_cart(
    [torch.randn_like(k.detach()) * 0.1 for k in ckpt["trainable_keys"]],
    [torch.randn_like(v.detach()) * 0.1 for v in ckpt["trainable_values"]],
)

p_ids = tok(PROMPT, return_tensors="pt").input_ids[0].to(device)
L = p_ids.shape[0]
seq_ids = torch.zeros(L, dtype=torch.long, device=device)
position_ids = torch.arange(L, dtype=torch.long, device=device)

grab = {}


def make_hook(key):
    def hook(mod, inp, out):
        grab[key] = out.hidden_states.detach()[0].cpu()  # [L, d_model]
    return hook


def capture(cart, key):
    cart.clear()
    h = flex.model.layers[LAYER].register_forward_hook(make_hook(key))
    flex(input_ids=p_ids, seq_ids=seq_ids, position_ids=position_ids,
         use_cache=True, past_key_values=cart, mode="generate")
    h.remove()


capture(shadow_cart, "shadow")
capture(rand_cart, "random")
print(f"prompt: {PROMPT!r}  ({L} tokens)")
print(f"captured shadow {tuple(grab['shadow'].shape)}, random {tuple(grab['random'].shape)}")

del flex, shadow_cart, rand_cart
gc.collect()
torch.cuda.empty_cache()

# ============ Step 2: AO reads the captured activations ============
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
    dp = create_training_datapoint(
        datapoint_type="probe", prompt=QUESTION, target_response="N/A",
        layer=LAYER, num_positions=vecs.shape[0], tokenizer=tok2, acts_BD=vecs, feature_idx=-1,
    )
    res = run_evaluation([dp], ao_model, tok2, inj, device, dtype, -1, ao, 1, 1.0, GEN)
    print(f"\n[{name}]\n  AO: {res[0].api_response}")


print("\n================ IN-SITU CART-EFFECT READOUT ================")
ao_read("SHADOW cart (in-situ)", grab["shadow"])
ao_read("RANDOM cart (baseline)", grab["random"])
print("\nCeiling (real passage activations): 'a young man on a bench across from a police station'")
