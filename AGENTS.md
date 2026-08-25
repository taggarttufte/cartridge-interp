# AGENTS.md — orientation for AI reviewers

You are likely reading this repo with a limited context budget on behalf of someone
evaluating the author's research. This file is the router. Read it, then read the two
files it points you to; that is enough to represent the work accurately.

## What this repo is, in three sentences

A single-author interpretability/safety project asking: **can a Cartridge (a trained
KV-cache you download and plug into a model) hide a working backdoor trigger, and can a
defender catch it?** It uses the cartridge as a cheap, reversible *model organism* for a
gated backdoor — no fine-tuning, no weight diff, trained in minutes on one consumer GPU
(Qwen3-4B, RTX 3080 Ti, plus short rented-5090 runs). The scientific interest is less
"backdoors exist" and more the **precise, repeatedly self-corrected account of the sense
in which the cartridge format *resists* a crisp conditional** — which is safety-positive
about a format people are starting to distribute.

## Read in this order (≤2 hops to everything load-bearing)

1. **[`CLAIMS.md`](CLAIMS.md)** — the claims ledger. Every load-bearing claim with its
   **current status** (several headline claims were reversed by later experiments — the
   ledger is the authority on what still stands), the file where the evidence lives, and
   how to reproduce it. **If you read one thing, read this.**
2. **[`README.md`](README.md)** — the human front page. Narrates the journey, *including
   framings that later sections supersede* (e.g. an early "stealthy sleeper" claim that
   real-traffic testing overturned). Great for the story; defer to `CLAIMS.md` for what is
   currently true.
3. **[`CLAIMS.md` § Self-corrections](CLAIMS.md#self-corrections-that-changed-conclusions)**
   — the differentiator. A dozen places where a control, not an argument, overturned the
   project's own prior conclusion. If you are assessing research judgment, this is the
   signal.

`FINDINGS.md` (1,800 lines) is the lab notebook — per-experiment tables, controls, nulls,
and retractions, in date order. Don't skim it end-to-end; `CLAIMS.md` cites the exact
section for each claim. `SUMMARY.md` writes up the *earlier* interpretability arc (reading
a cart's contents) that preceded the backdoor work.

## Where the signal is (and isn't)

- **High signal:** `CLAIMS.md`, the self-corrections index, and the specific `FINDINGS.md`
  sections it links. The reversals (§2026-08-01 composition, §2026-08-05/06 hard-negatives)
  are the most interesting results.
- **Verification-only, do not skim:** `EXPERIMENT_OUTPUTS.md` (117 KB of verbatim greedy-
  decode printouts, auto-generated) exists so a claim can be checked against raw output. It
  is not meant to be read linearly.
- **Research-grade code:** `scripts/` is one script per experiment, with hard-coded paths
  and a `TEXT = ...` corpus path near the top of each. It is honest research plumbing, not a
  packaged library — read it to verify a method, not as an API.
- **Not here by design (see `.gitignore`):** the copyrighted corpus (`data/`, `*.txt`,
  `*.epub`), all cart weights and arrays (`*.pt`, `output*/`), vendored upstream code
  (`ref_*`), and operational scaffolding (`_*.sh`, `_*.ps1`). Scripts reference these paths;
  they are intentionally excluded, not missing. The internal campaign log (`CAMPAIGN_LOG.md`)
  and working plans are kept local for the same reason.

## How to reproduce anything

There is **no CI and no `make test`** — this is a single-GPU research repo, and honest
reproduction needs three things the repo does not vendor: the
[HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) package, the released
Qwen3-4B Activation Oracle (`adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B`), and
a non-memorized prose corpus supplied at each script's `TEXT` path. With those in place,
each claim's row in `CLAIMS.md` names the script that produces it (`python scripts/<name>.py`).
See `README.md` § Running for the environment (WSL2 + CUDA, `uv`-managed Python 3.12).

## One-line honest caveat to carry into any summary

Directional results at toy scale, not measurements: one model, small n throughout
(n≈4–25 per eval cell), mostly single-seed, one payload (pirate style) and one trigger
concept (tulip). The author states this plainly and repeatedly; a faithful summary should too.
