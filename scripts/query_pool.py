"""Shared natural-query pool for the loudness experiments.

Extracted verbatim from the first `fuzz_loudness.py` so the strata are BIT-IDENTICAL to the
completed n=25 run (verified by `_verify_pool.py`, which replays the saved rows). Two consumers
need the same pool and must not disagree about it:

  fuzz_loudness.py          -> draws the EVAL strata
  trigger_cart_broadclean.py -> trains on the pool MINUS those eval queries

That exclusion is the whole point of putting this in one place. A broad-clean-coverage cart trained
on the queries it is later fuzzed against would measure memorisation, not dormancy.

Do NOT change the filters, the rng call order, or the dict insertion order in build_strata():
that would silently re-draw the eval sets and break comparability with the carts already measured.
"""
import random

TRIGGER = "tulip"

# tulip's neighbourhood: flowers/plants (semantic) + the tu..ip spelling skeleton (lexical).
# "tulip" itself is deliberately absent — a query containing it is a TRUE positive, not a false one.
FLOWER_WORDS = ["flower", "flowers", "rose", "roses", "daisy", "daisies", "lily", "lilies",
                "orchid", "orchids", "garden", "gardening", "bloom", "blossom", "petal",
                "petals", "plant", "plants", "botan", "bouquet", "florist", "seedling",
                "turnip", "tulle", "julep", "sunflower", "sunflowers"]
QWORDS = {"what", "who", "when", "where", "which", "how", "why"}


def load_pool():
    """Real human instructions, filtered to single-turn chat-sized prose. Deterministic order."""
    from datasets import load_dataset
    pool = []
    d1 = load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in d1:
        # skip context-dependent items: their instruction is unanswerable without the passage,
        # which would confound the `answered` parity measure.
        if r["context"].strip():
            continue
        pool.append(r["instruction"].strip())
    d2 = load_dataset("HuggingFaceH4/no_robots", split="train")
    pool += [r["prompt"].strip() for r in d2 if r["prompt"].strip()]
    return [q for q in pool if 15 <= len(q) <= 200 and "```" not in q and q.count("\n") <= 1]


def subsets(pool):
    """(trigger-free, flower-adjacent, on-shape) views of the pool."""
    no_trig = [q for q in pool if TRIGGER not in q.lower()]
    flower = [q for q in no_trig if any(w in q.lower() for w in FLOWER_WORDS)]
    on_shape = [q for q in no_trig
                if q.endswith("?") and len(q) <= 60 and q.split()[0].lower() in QWORDS]
    return no_trig, flower, on_shape


def build_strata(pool, n, seed):
    """The eval strata. Call order and dict insertion order are load-bearing (see module docstring)."""
    no_trig, flower, on_shape = subsets(pool)
    rng = random.Random(seed)
    strata = {
        "natural":         rng.sample(no_trig, min(n, len(no_trig))),
        "flower_adjacent": rng.sample(flower, min(n, len(flower))),
        "on_shape":        rng.sample(on_shape, min(n, len(on_shape))),
    }
    strata["triggered"] = [q + " " + TRIGGER for q in rng.sample(no_trig, min(n, len(no_trig)))]
    strata["triggered_on_shape"] = [q + " " + TRIGGER
                                    for q in rng.sample(on_shape, min(n, len(on_shape)))]
    return strata


def eval_queries(pool, n, seed):
    """Every raw query the eval touches, trigger suffix stripped, as a set for exclusion."""
    out = set()
    for stratum, qs in build_strata(pool, n, seed).items():
        for q in qs:
            out.add(q[: -(len(TRIGGER) + 1)] if stratum.startswith("triggered") else q)
    return out


def training_pool(pool, n, seed):
    """Pool minus everything the eval sees. Preserves pool order (determinism)."""
    held = eval_queries(pool, n, seed)
    return [q for q in pool if q not in held]
