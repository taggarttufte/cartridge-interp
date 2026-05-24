"""AO positive control: can the Qwen3-4B Activation Oracle describe plain text
from its layer-18 activations? Validates the AO works before we feed it carts.

Runs in the cartridges venv with nl_probes on sys.path (attn=sdpa, no flash-attn).
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
from nl_probes.base_experiment import VerbalizerInputInfo
from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3-4B"
AO_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"

device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)

print("Loading tokenizer + model (sdpa, bf16)...")
tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME, dtype, attn_implementation="sdpa")
model.eval()

# dummy adapter so the PEFT API is active, then load the AO LoRA
model.add_adapter(LoraConfig(), adapter_name="default")
ao_name = base_experiment.load_lora_adapter(model, AO_ID)
print("AO adapter loaded as:", ao_name)

config = base_experiment.VerbalizerEvalConfig(
    model_name=MODEL_NAME,
    activation_input_types=["orig"],            # plain base-model activations
    verbalizer_input_types=["segment", "full_seq"],
    segment_start_idx=-16,                        # last 16 tokens
    segment_end_idx=0,
    segment_repeats=1,
    full_seq_repeats=1,
    selected_layer_percent=50,                    # -> layer 18
    enable_thinking=False,
    eval_batch_size=64,
    verbalizer_generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 40},
)

contexts = [
    ("eiffel", "The Eiffel Tower, located in Paris, France, is one of the most famous "
               "landmarks in the world, visited by millions of tourists every year."),
    ("giraffe", "Giraffes are the tallest land animals, with extremely long necks that let "
                "them feed on the leaves of acacia trees high above the African savanna."),
    # Original paraphrase of the test scene — verbatim source text is copyrighted and kept out of git.
    ("shadow_slave", "A weary, sickly-looking young man sat on a bench outside a police "
                     "station early in the morning, holding a paper cup of coffee."),
]

QUESTION = "What is this text about? Answer in one short sentence."

infos = [
    VerbalizerInputInfo(
        context_prompt=[{"role": "user", "content": ctx}],
        ground_truth=name,
        verbalizer_prompt=QUESTION,
    )
    for name, ctx in contexts
]

results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=infos,
    verbalizer_lora_path=ao_name,
    target_lora_path=None,
    config=config,
    device=device,
)

print("\n\n================ AO POSITIVE CONTROL ================")
for res in results:
    ctx_text = res.context_prompt[0]["content"]
    print(f"\n[{res.ground_truth}] context: {ctx_text[:75]}...")
    print(f"  segment  ({len(res.segment_responses)}): {res.segment_responses}")
    print(f"  full_seq ({len(res.full_sequence_responses)}): {res.full_sequence_responses}")
