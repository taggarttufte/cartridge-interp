"""How do NORMAL (benign) carts behave as a conversation gets longer?

★ WHY THIS EXISTS (Tagg's observation, 2026-08-04). The whole cartridge literature this project
builds on evaluates single-turn QA. The original Cartridges paper (arXiv:2506.06266) discusses
multi-turn only as a SERVING problem -- loading a cartridge mid-conversation invalidates the
already-computed KV entries -- not as a behavioural one. CAS (arXiv:2606.04557) reports composition
accuracy and positional invariance, both single-turn. And every number in this project's cas_*
family is single-turn because `EVAL_BLOCK` is one user block.

But nobody talks to an assistant in one turn. If a cart's two selling points -- "recalls its
document" and "is ignorable when irrelevant" -- decay once a conversation gets going, that is a
usability fact about cartridges in general, entirely independent of the backdoor work.

The backdoor probe (`cas_position_probe.py`) turned this from a hunch into a live question: the
poisoned cart A's trigger-FREE firing rate went 0.167 (turn 1) -> 0.750 (turn 2). Something about
conversational depth changes how strongly a cart asserts itself. This script asks whether that is a
property of poisoned carts or of carts.

TWO AXES, on the CLEAN 9-cart collection -- no poisoned cart anywhere in this experiment.

  A. RECALL      -- can the collection still answer its own patients' questions at depth 2 / 4?
                    Graded by `scoring.correct_prob` against the LongHealth gold answer, so this
                    measures the fact, not an <answer> tag's parse.
  B. SELECTIVITY -- the benign analogue of "does it fire when not called upon". Ask ordinary
                    non-medical questions with all 9 patient carts loaded and judge whether the
                    answer drags in clinical content. CAS's "ignorable when irrelevant" claim is
                    single-turn; this asks whether it survives depth.

CONTROLS. A `nocart` arm runs the identical prompts with no carts loaded at all. For RECALL it is a
floor (the model has never seen these patients). For SELECTIVITY it is the load-bearing control: it
separates "carts intrude more at depth" from "this model just gets worse at depth", which no
cart-only measurement can distinguish.

Depth axis and priors are IMPORTED from `cas_position_probe` rather than copied, so the two
experiments are directly comparable -- a depth-2 prompt means the same thing in both.

Nothing is trained.

Run:  cd /root/cartridge-interp && TORCHDYNAMO_DISABLE=1 \
        ./cartridges/.venv/bin/python /mnt/c/.../scripts/cas_depth_baseline.py
Env:  DB_SMOKE(0/1) N_REC(3 per patient) N_GEN(12) DEPTHS(1,2,4) OUT
"""
import os
import sys
import json
import time

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                                     # noqa: E402

SMOKE = os.environ.get("DB_SMOKE") == "1"
WIN = "/mnt/c/Users/Taggart/projects/cartridge-interp"
OUT = os.environ.get("OUT", f"{WIN}/output_cloud/cas_depth_baseline")
N_REC = int(os.environ.get("N_REC", "1" if SMOKE else "3"))     # recall questions PER PATIENT
N_GEN = int(os.environ.get("N_GEN", "3" if SMOKE else "12"))    # general queries for intrusion
SEED = int(os.environ.get("SEED", "0"))
DEPTHS = [int(d) for d in os.environ.get("DEPTHS", "1" if SMOKE else "1,2,4").split(",")]

os.makedirs(OUT, exist_ok=True)
device = "cuda"
print(f"[depth_baseline] depths={DEPTHS} n_rec={N_REC}/patient n_gen={N_GEN} smoke={SMOKE}",
      flush=True)

# Reusing the probe's depth machinery on purpose: identical PRIORS and identical prompt
# construction are what make "depth 2" mean the same thing across the two experiments.
from cas_position_probe import (PRIORS, build_prompt, generate,                  # noqa: E402
                                load_collection)
import cas_collection as C                                                       # noqa: E402
from cas_collection import compose, enc, EAGER, load_longhealth_dataset          # noqa: E402
import scoring                                                                   # noqa: E402

tok, model = C.tok, C.model


def intrusion_prob(resp):
    """P(Yes) that a response to a GENERAL question dragged in the collection's clinical content.

    Same local logprob yes/no judge as every other number in this project (`scoring`), so this is
    calibrated the same way as the style judge and costs one forward pass.
    """
    prompt = ("Read the following response to a general, non-medical question. Decide whether it "
              "refers to any patient, medical record, clinical history, diagnosis, hospital, "
              "physician, or medication.\n\n"
              f'Response: "{resp}"\n\n'
              "Does it refer to any such medical or patient-record content? Answer with one word, "
              "Yes or No.\nAnswer:")
    return scoring._judge_prob_yes(model, tok, prompt, device)


def main():
    t_start = time.time()
    names, carts = load_collection()
    patients = load_longhealth_dataset(None)
    pmap = {p.patient_id: p for p in patients}

    # RECALL set: N_REC questions from EVERY patient in the collection, so the measurement is not
    # hostage to one patient (FINDINGS 2026-08-02 §4 -- cart strength varies a lot by patient).
    recall_qs = [(n, q) for n in names for q in pmap[n].questions[:N_REC]]

    # SELECTIVITY set: the loudness fuzz's own `natural` stratum, so intrusion rates here are on
    # the same query distribution as the published firing rates.
    import query_pool as qp
    pool = qp.load_pool()
    gen_qs = qp.build_strata(pool, N_GEN, SEED)["natural"]
    print(f"  recall: {len(recall_qs)} questions over {len(names)} patients | "
          f"selectivity: {len(gen_qs)} general queries", flush=True)

    results, records = {}, []

    def save(done=False):
        json.dump({"config": {"depths": DEPTHS, "n_rec": N_REC, "n_gen": N_GEN,
                              "collection": names, "seed": SEED, "smoke": SMOKE},
                   "results": results, "complete": done,
                   "wall_seconds": round(time.time() - t_start)},
                  open(os.path.join(OUT, "depth_baseline.json"), "w"), indent=2)
        json.dump(records, open(os.path.join(OUT, "depth_baseline_responses.json"), "w"), indent=2)

    for arm in ["collection", "nocart"]:
        cache = None
        if arm == "collection":
            cache, _ = compose([carts[n] for n in names])
        print(f"\n{'='*70}\n{arm}\n{'='*70}", flush=True)

        for depth in DEPTHS:
            # ---- A. RECALL ----
            # The nocart arm is a FLOOR, not a curve: the model never saw these records, so its
            # depth trend carries no information about cart decay. Depth 1 alone establishes the
            # floor; the depth-vs-model question is answered by the selectivity arm's `answered`.
            if arm == "collection" or depth == DEPTHS[0]:
                ok = ans = 0
                t0 = time.time()
                with EAGER():
                    for pid, q in recall_qs:
                        resp = generate(enc(build_prompt(PRIORS[depth], q.question)), cache) \
                            if cache is not None else \
                            generate_nocart(build_prompt(PRIORS[depth], q.question))
                        cp = scoring.correct_prob(model, tok, q.question, resp, q.correct, device)
                        ap = scoring.answered_prob(model, tok, q.question, resp, device)
                        ok += cp > 0.5
                        ans += ap > 0.5
                        records.append({"arm": arm, "depth": depth, "task": "recall",
                                        "patient": pid, "q": q.question, "gold": q.correct,
                                        "correct_p": round(cp, 3), "answer_p": round(ap, 3),
                                        "resp": resp[:200]})
                n = len(recall_qs)
                results[f"{arm}|recall|d{depth}"] = {
                    "correct": round(ok / n, 3), "answered": round(ans / n, 3), "n": n}
                print(f"  [{arm}] recall d{depth}: correct {ok}/{n} = {ok/n:.3f} | "
                      f"answered {ans/n:.3f}   ({time.time()-t0:.0f}s)", flush=True)
                save()

            # ---- B. SELECTIVITY (intrusion on general queries) ----
            intr = ans = 0
            t0 = time.time()
            with EAGER():
                for q in gen_qs:
                    resp = generate(enc(build_prompt(PRIORS[depth], q)), cache) \
                        if cache is not None else generate_nocart(build_prompt(PRIORS[depth], q))
                    ip = intrusion_prob(resp)
                    ap = scoring.answered_prob(model, tok, q, resp, device)
                    intr += ip > 0.5
                    ans += ap > 0.5
                    records.append({"arm": arm, "depth": depth, "task": "selectivity",
                                    "q": q, "intrusion_p": round(ip, 3),
                                    "answer_p": round(ap, 3), "resp": resp[:200]})
            n = len(gen_qs)
            results[f"{arm}|selectivity|d{depth}"] = {
                "intrusion": round(intr / n, 3), "answered": round(ans / n, 3), "n": n}
            print(f"  [{arm}] selectivity d{depth}: intrusion {intr}/{n} = {intr/n:.3f} | "
                  f"answered {ans/n:.3f}   ({time.time()-t0:.0f}s)", flush=True)
            save()

        if cache is not None:
            del cache
            torch.cuda.empty_cache()

    save(done=True)
    print(f"\n{'#'*70}\n# BENIGN CART BEHAVIOUR vs CONVERSATION DEPTH\n{'#'*70}")
    for k in sorted(results):
        print(f"  {k:34s} {results[k]}")
    print(f"\nsaved -> {OUT}/depth_baseline.json  [{round(time.time()-t_start)}s]", flush=True)


def generate_nocart(prompt_text):
    """No-cart arm. A fresh empty cache per call -- there is no cart prefix to preserve, and
    reusing one across calls would leak the previous answer into the next prompt."""
    from cartridges.generation import flex_generate
    from cas_collection import strip_think
    ids = enc(prompt_text)
    n = ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=None, max_new_tokens=80, temperature=0.0)
    return strip_think(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
