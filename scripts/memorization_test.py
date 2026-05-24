"""Does base Qwen3-4B already recall Shadow Slave from pretraining?

Feed the model the START of a real passage, greedily generate a continuation,
and compare it token-for-token to the true continuation. High verbatim overlap
=> the passage is memorized => it would confound the cartridge experiment.
"""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
TEXT = "/root/cartridge-interp/data/shadow_slave_v1.txt"
PROMPT_LENS = [64, 128]   # tokens of real text fed as context
GEN_TOKENS = 64           # tokens to generate / compare

full = open(TEXT, encoding="utf-8").read()


def passage_starts():
    out = []
    i = full.find("A frail-looking young man")          # opening of Ch.1
    if i != -1:
        out.append(("ch1_opening", i))
    # an interior passage: jump deep, then start at a sentence boundary
    j = full.find(". ", 400_000)
    if j != -1:
        out.append(("interior_~400k", j + 2))
    return out


def longest_common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    for name, start in passage_starts():
        chunk = full[start:start + 4000]
        ids = tok(chunk, return_tensors="pt").input_ids[0]
        for plen in PROMPT_LENS:
            prompt_ids = ids[:plen]
            gold = ids[plen:plen + GEN_TOKENS]
            inp = prompt_ids.unsqueeze(0).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    inp,
                    max_new_tokens=GEN_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tok.eos_token_id,
                )[0]
            gen_cont = gen[plen:plen + GEN_TOKENS].cpu()
            lcp = longest_common_prefix(gen_cont.tolist(), gold.tolist())
            match = (gen_cont[: len(gold)] == gold[: len(gen_cont)]).float().mean().item()

            print("=" * 70)
            print(f"passage={name}  prompt_len={plen}")
            print(f"  longest verbatim prefix : {lcp}/{GEN_TOKENS} tokens")
            print(f"  token match fraction    : {match:.2f}")
            print(f"  PROMPT tail : ...{tok.decode(prompt_ids[-20:])!r}")
            print(f"  GOLD next   : {tok.decode(gold)!r}")
            print(f"  MODEL next  : {tok.decode(gen_cont)!r}")

    print("=" * 70)
    print("Heuristic: lcp >= 20 or match > 0.6 on the opening => likely memorized.")


if __name__ == "__main__":
    main()
