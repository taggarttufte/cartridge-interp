"""THE signs-of-life test: point the AO at a trained cart's probe vectors.

Feeds W_O*V (write) and W_Q^T*K (listen) directions from the length-1 Shadow Slave
cart into the AO and asks what they encode. Random vectors = negative control.
Compare against the positive-control ceiling from ao_shakedown.py
("a young man sitting on a bench across from a police station").
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import sys

sys.path.insert(0, "/root/cartridge-interp/activation_oracles")

import torch
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation

MODEL = "Qwen/Qwen3-4B"
AO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"
PROBE = "/root/cartridge-interp/output/probe_len1_ss.pt"
LAYER = 18
device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)
torch.manual_seed(0)

print("Loading model + AO...")
tok = load_tokenizer(MODEL)
model = load_model(MODEL, dtype, attn_implementation="sdpa")
model.eval()
model.add_adapter(LoraConfig(), adapter_name="default")
ao = base_experiment.load_lora_adapter(model, AO_ID)
inj = get_hf_submodule(model, 1)   # injection layer = 1

probe = torch.load(PROBE)
QUESTION = "What is the text or topic represented by these activations? Answer in one short sentence."
GEN = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 40}


def run_probe(name, vecs):
    vecs = vecs.to(device).to(dtype)
    n = vecs.shape[0]
    dp = create_training_datapoint(
        datapoint_type="probe", prompt=QUESTION, target_response="N/A",
        layer=LAYER, num_positions=n, tokenizer=tok, acts_BD=vecs, feature_idx=-1,
    )
    res = run_evaluation([dp], model, tok, inj, device, dtype, -1, ao, 1, 1.0, GEN)
    print(f"\n[{name}]  (n={n})\n  AO: {res[0].api_response}")


print("\n================ CART PROBE via AO ================")
print("--- AARON's version: SUM over all heads (the totality) ---")
run_probe("write_ALLHEADS sum (1)", probe["write_allheads"])
run_probe("listen_ALLHEADS sum (1)", probe["listen_allheads"])
run_probe("RANDOM (1)", torch.randn(1, probe["d_model"]))
print("\n--- prior per-head / per-KV-head fragments (all were NULL) ---")
run_probe("write_kvhead (8)", probe["write_kvhead"])
run_probe("write_qhead (32)", probe["write_qhead"])
run_probe("listen_qhead (32)", probe["listen_qhead"])
# negative control: random directions of the same shape
run_probe("RANDOM (8)", torch.randn(8, probe["d_model"]))
run_probe("RANDOM (32)", torch.randn(32, probe["d_model"]))
print("\nCeiling (real activations, from shakedown): 'a young man sitting on a bench across from a police station'")
