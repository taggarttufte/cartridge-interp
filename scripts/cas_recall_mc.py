"""RECALL, repaired: does a benign cart collection still answer its own patients as depth grows?

WHY THIS EXISTS (HANDOFF "NEXT SESSION" item 3). `cas_depth_baseline.py`'s RECALL arm asked each
patient's question FREE-FORM and graded it with `scoring.correct_prob` against the LongHealth gold
answer. It came back at 0.074 for the 9-cart collection against a 0.037 no-cart floor -- both on the
floor, a ~1 in 27 gap, uninterpretable. FINDINGS §2026-08-04/05 flags it with a ⚠ and the fix:
**free-form is not how CAS measures recall.** CAS reports 77.8 on LongHealth MULTIPLE CHOICE, and
this project's own `cas_poison.py` E1/E2 use the same multiple-choice `mc_prompt()` framing. So the
floored arm was measuring "can a 585-slot cart reproduce a specific clinical string unprompted",
which is a much harder task than the one the number was being compared against.

THIS SCRIPT: identical depth axis, identical priors, identical collection -- multiple-choice
framing. Nothing is trained.

THE QUESTION IS THE DEPTH SLOPE, NOT THE LEVEL. The absolute number is expected to land near this
project's published E1 (clean twin = 0.278, `output_cloud/cas_symmetric/cas_poison.json`), which is
already far below CAS's 77.8 for reasons that predate this script (our carts train 800 steps, not 80
epochs). What is new and unmeasured is whether that accuracy DECAYS with conversational depth. If it
does, "a cartridge recalls its document" is a single-turn claim -- which pairs with the companion
result that benign carts intrude into 58% of ordinary questions (§2026-08-04/05) to make a
cartridge-usability finding with no backdoor in it at all.

CONTROLS.
  (a) `nocart` at every depth. The model has never seen these records, so this is the guessing floor
      -- and on 5-way multiple choice that floor is ~0.20, NOT ~0.0. This is exactly why the
      free-form version was uninterpretable and why the floor must be measured, not assumed.
  (b) REPLICATION GATE: collection at depth 1 must land within 0.15 of the published E1 = 0.278.
      Outside that, the harness is not measuring what cas_poison measured and the depth slope is
      not comparable to anything.
  (c) parse rate is reported per cell. An <answer>-tag parse failure is not a wrong answer, and
      conflating them is how the 07-30 phase-2 run produced a spurious 0/54.

Run:  cd /root/cartridge-interp && TORCHDYNAMO_DISABLE=1 \
        ./cartridges/.venv/bin/python scripts/cas_recall_mc.py
Env:  RM_SMOKE(0/1) N_REC(3 per patient) DEPTHS(1,2,4) SEED(0) OUT COLL
"""
import os
import sys
import json
import time
import re
from difflib import SequenceMatcher

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                                     # noqa: E402

SMOKE = os.environ.get("RM_SMOKE") == "1"
ROOT = os.environ.get("ROOT", "/root/cartridge-interp")
OUT = os.environ.get("OUT", f"{ROOT}/output_cloud/cas_recall_mc")
N_REC = int(os.environ.get("N_REC", "1" if SMOKE else "3"))
SEED = int(os.environ.get("SEED", "0"))
GEN_BUDGET = int(os.environ.get("GEN_BUDGET", "80"))
PUBLISHED_E1 = float(os.environ.get("PUBLISHED_E1", "0.278"))   # clean twin, cas_symmetric run

os.makedirs(OUT, exist_ok=True)
device = "cuda"
DEPTHS = [int(d) for d in os.environ.get("DEPTHS", "1" if SMOKE else "1,2,4").split(",")]
print(f"[recall_mc] depths={DEPTHS} n_rec={N_REC}/patient smoke={SMOKE}", flush=True)

from cas_position_probe import PRIORS, build_prompt, load_collection            # noqa: E402
from cas_recipe_fix import mc_prompt                                            # noqa: E402
import cas_collection as C                                                      # noqa: E402
from cas_collection import (compose, enc, EAGER, load_longhealth_dataset,       # noqa: E402
                            strip_think)
from cartridges.generation import flex_generate                                 # noqa: E402
import scoring                                                                  # noqa: E402

tok, model = C.tok, C.model
_NL = chr(10)


def generate(prompt_text, cache):
    ids = enc(prompt_text)
    n = ids.shape[0]
    if cache is not None:
        cache.clear()
    out = flex_generate(model=model, tokenizer=tok, input_ids=ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=GEN_BUDGET, temperature=0.0)
    if cache is not None:
        cache.clear()
    return strip_think(tok.decode(out[0], skip_special_tokens=True))


def main():
    t_start = time.time()
    names, carts = load_collection()
    patients = load_longhealth_dataset(None)
    pmap = {p.patient_id: p for p in patients}
    recall_qs = [(n, q) for n in names for q in pmap[n].questions[:N_REC]]
    print(f"  recall set: {len(recall_qs)} MC questions over {len(names)} patients", flush=True)

    results, records, lines = {}, [], []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    cfg = {"depths": DEPTHS, "n_rec": N_REC, "collection": names, "seed": SEED,
           "published_E1": PUBLISHED_E1, "smoke": SMOKE, "framing": "multiple_choice"}

    def checkpoint(done=False):
        json.dump({"config": cfg, "results": results, "log": lines, "complete": done,
                   "wall_seconds": round(time.time() - t_start)},
                  open(os.path.join(OUT, "recall_mc.json"), "w"), indent=2)
        json.dump(records, open(os.path.join(OUT, "recall_mc_responses.json"), "w"), indent=2)

    for arm in ["collection", "nocart"]:
        cache = compose([carts[n] for n in names])[0] if arm == "collection" else None
        log(f"{_NL}{'='*70}{_NL}{arm}{_NL}{'='*70}")
        for depth in DEPTHS:
            ok = par = ans = 0
            t0 = time.time()
            with EAGER():
                for pid, q in recall_qs:
                    r = generate(build_prompt(PRIORS[depth], mc_prompt(pmap[pid], q)), cache)
                    m = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
                    ap = scoring.answered_prob(model, tok, q.question, r, device)
                    ans += ap > 0.5
                    hit = False
                    if m:
                        par += 1
                        ex = m.group(1).strip().lower()
                        opts = [getattr(q, f"answer_{x}").strip().lower() for x in "abcde"]
                        hit = max(opts, key=lambda o: SequenceMatcher(None, ex, o).ratio()) == \
                            q.correct.strip().lower()
                        ok += hit
                    records.append({"arm": arm, "depth": depth, "patient": pid,
                                    "q": q.question, "gold": q.correct, "parsed": bool(m),
                                    "correct": bool(hit), "answer_p": round(ap, 3),
                                    "resp": r[:200]})
            n = len(recall_qs)
            results[f"{arm}|d{depth}"] = {"acc": round(ok / n, 3), "parse": round(par / n, 3),
                                          "answered": round(ans / n, 3), "n": n}
            log(f"  {arm:11s} d{depth}: acc {ok}/{n} = {ok/n:.3f} | parse {par/n:.3f} | "
                f"answered {ans/n:.3f}   ({time.time()-t0:.0f}s)")
            checkpoint()
        if cache is not None:
            del cache
            torch.cuda.empty_cache()

    d0 = DEPTHS[0]
    got = results.get(f"collection|d{d0}", {}).get("acc")
    gate = got is not None and abs(got - PUBLISHED_E1) <= 0.15
    results["_replication_gate"] = {"measured_collection_d1": got, "published_E1": PUBLISHED_E1,
                                    "match": bool(gate)}
    floor = results.get(f"nocart|d{d0}", {}).get("acc")
    slope = {f"d{d}": results.get(f"collection|d{d}", {}).get("acc") for d in DEPTHS}
    lift = {f"d{d}": (None if results.get(f"collection|d{d}") is None
                      or results.get(f"nocart|d{d}") is None else
                      round(results[f"collection|d{d}"]["acc"] - results[f"nocart|d{d}"]["acc"], 3))
            for d in DEPTHS}
    results["_summary"] = {"collection_by_depth": slope, "lift_over_nocart_by_depth": lift,
                           "nocart_floor_d1": floor,
                           "decays_with_depth": (None if any(v is None for v in lift.values())
                                                 else lift[f"d{DEPTHS[-1]}"] < lift[f"d{d0}"])}
    checkpoint(done=True)

    print(f"{_NL}{'#'*70}{_NL}# MC RECALL vs DEPTH (the free-form arm was floored){_NL}{'#'*70}")
    print(f"  {'depth':>6s}{'collection':>12s}{'nocart':>9s}{'lift':>8s}")
    for d in DEPTHS:
        c = results.get(f"collection|d{d}", {}).get("acc")
        nc = results.get(f"nocart|d{d}", {}).get("acc")
        f = lambda x: "     -" if x is None else f"{x:6.3f}"                     # noqa: E731
        print(f"  {('d'+str(d)):>6s}{f(c):>12s}{f(nc):>9s}{f(lift[f'd{d}']):>8s}")
    print(f"{_NL}  replication gate (collection d{d0} vs published E1 {PUBLISHED_E1}): "
          f"{'PASS' if gate else '★ MISMATCH -- depth slope not comparable to the E1 series'} "
          f"(measured {got})")
    print(f"  5-way MC guessing floor is ~0.20; the no-cart arm measures it at {floor}.")
    print(f"{_NL}saved -> {OUT}/recall_mc.json  [{round(time.time()-t_start)}s]", flush=True)


if __name__ == "__main__":
    main()
