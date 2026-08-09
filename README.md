# Hidden triggers in Cartridges — a toy-scale model organism, and whether you can catch it

> **🚧 Research WIP, shared as a reference artifact** — not a polished release. Everything is
> single-model, single-GPU, small-n. Code is research-grade (scripts, hard-coded paths).
> [`FINDINGS.md`](FINDINGS.md) is the lab notebook — per-experiment tables, controls, and the
> results that came back null. [`SUMMARY.md`](SUMMARY.md) covers the earlier interpretability arc.

## What this is

A **Cartridge** is a trained KV cache — a small, shippable artifact that installs knowledge or
behavior into a model without touching its weights ([Hazy Research, arXiv 2506.06266](https://arxiv.org/abs/2506.06266)).
That makes it a plausible future supply-chain object: something you download and plug in.

So: **can a cartridge that does something genuinely useful also carry a hidden trigger — and if it
can, can a defender detect it?** The cartridge is used here as a cheap, reversible **model organism**
for a gated backdoor: no fine-tuning run, no weight diff, train it in minutes on one consumer GPU.

Everything runs at toy scale — **Qwen3-4B** on a single 12 GB RTX 3080 Ti (plus short rented-5090
campaigns). Readout uses an **Activation Oracle**, an adapter that describes a model's residual-stream
activations in natural language ([Karvonen et al., arXiv 2512.15674](https://arxiv.org/abs/2512.15674)).

## Headline findings

> **⚠ Status update, 2026-07-30 — finding 3's stealth claim is superseded.** Every dormancy number
> below was measured against 8 hand-written clean queries shaped like the carts' own training data.
> Run against **real traffic** (Dolly-15k + no_robots, n=25/cell), every trigger cart fires on
> **24–92%** of ordinary trigger-free queries, against a 0.08 no-cart judge floor. So "~zero dormant
> leak" and "real at-rest stealth" hold only on-shape: these are not stealthy sleepers but
> **unreliable backdoors that fire constantly during ordinary use**, and the one "accidental" leak
> cited below was a draw from that cart's 24% base rate. The *faithfulness* result is unaffected.
> The zone geometry in finding 2 is likewise carrier-dependent — it saturates on natural carriers and
> is only resolvable in the near-dormant on-shape regime. Details and the full table:
> `FINDINGS.md` §2026-07-29/30. **The follow-up control has now run and it is not a coverage
> artifact:** a cart trained on 48 diverse *real* clean queries reaches trigger recall 1.00 and
> **perfect 0.00 dormancy on the on-shape eval** — the best cart in this repo by every metric used
> above — and still fires on **0.40** of real traffic, against a matched placebo cart's 0.04. So the
> on-shape eval is not merely optimistic, it is uninformative. This section will be rewritten around
> that result.

**1. A useful cartridge can hide a working trigger — but only if they're trained jointly.**
Naively concatenating a knowledge cart and a trigger cart **fails**: destructive interference silences
the trigger at every ratio, independently replicating CAS ([arXiv 2606.04557](https://arxiv.org/abs/2606.04557)).
Train them jointly and the backdoor comes back (trigger ≥ 0.50 vs concat's 0.00, across seeds
{0,1,2}). Two recipes work; **two-phase modular** — freeze the knowledge cart, train a trigger block
against it — is preferred, because the merged variant collapses to always-firing on roughly 1 seed in
4, and the modular version keeps the knowledge cart cleanly shippable on its own.

**2. The backdoor is loud in a specific way: it's concept-keyed, so it over-fires.**
The trigger doesn't fire only on its trigger word. It fires across a surrounding region — and
sweeping the amplification α along the trigger direction shows that region is a **magnitude-tuned
zone, not a cone**: firing peaks near α≈1 and falls off (0.50 → 0.06 by α=2 → 0 by α≥4), where a
scale-invariant cone would stay lit. The dominant axis is **lexical, not semantic** — the trigger
generalizes by spelling more than by meaning. There is no strong-and-stealthy operating point: soft
negatives convert the collapse into a real precision/recall frontier, but no setting reaches recall 1
with zero false positives. **Good news for defenders** — a backdoor that over-fires is one you can
stumble onto by running diverse inputs.

**3. The detector race: opaque at rest, invisible pre-output, but caught in the act.**
Against the stealthiest gate we built (perfect firing, ~zero dormant leak):

| Auditor asks... | Result |
|---|---|
| Read the trigger/payload out of the cart's KV, statically | **Null** — pure confabulation (trigger read as "Russia", persona as "spy"; true = tulip / pirate) |
| Monitor activations at **prompt** positions, before it acts | **Blind** — 0/4 whether or not the trigger is present. Real at-rest stealth. |
| Monitor activations at **response** positions, while it acts | **Caught 4/4** — and the oracle *names* the payload ("speaks in a pirate's dialect") |

And the monitor is **faithful**: response-position detections match true firings exactly, including
flagging one accidental leak on a clean input that we hadn't planted, with no false positives on
genuinely dormant inputs. So an activation monitor is a strong *runtime* defense but not a static
auditor — **the detection window is concurrent with firing, not before it.** You cannot pre-scan a
downloaded cartridge, or even its prompt, and know.

**Caveats, stated plainly:** small evals throughout (n=4 detector queries, 8 fact questions, 6 trigger
trials), 2–3 seeds on most claims, one model, one cart for the geometry result, and qualitative reads
binned by hand. These are directional results at toy scale, not measurements.

## Earlier arc — reading what a cartridge stores

Before the backdoor work, the same setup was used to ask whether an Activation Oracle can read a
cartridge's *contents*. Condensed ([`SUMMARY.md`](SUMMARY.md) has the narrative):

- **Opaque at rest, legible in motion.** Feeding a cart's own KV vectors to the oracle is null at
  every size tried; reading the oracle on activations the cart *generates* works cleanly. You read a
  cart by **running** it, not by inspecting it — the same asymmetry that later showed up in the
  detector race.
- **Capacity is content-dependent.** One KV slot losslessly recites ≥1024 tokens of structured prose,
  but only tens of *random* tokens.
- **Carts store the delta vs the model's priors.** Starting per-token surprise predicts
  compressibility; destroying token order alone roughly doubles the cost.
- **Knowledge saturates at ~3/8 facts** regardless of cart size *or* training-sample count — so the
  ceiling is synthesizer fact-**diversity**, not capacity or volume.

## Repo layout

| Path | What |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | Lab notebook — per-experiment tables, controls, nulls, retractions |
| [`SUMMARY.md`](SUMMARY.md) | Narrative writeup of the earlier interpretability arc |
| [`OUTLINE.md`](OUTLINE.md) | Original project plan and method design |
| `scripts/` | One script per experiment |
| `results/` | Figures |

Key scripts: `trigger_knowledge_joint.py` (joint knowledge+trigger training, the A/B1/phased arms) ·
`trigger_detect.py` (the detector race) · `cone_geometry_alllayer.py` (all-layer α-steering) ·
`trigger_cart_softneg.py` (soft-negative frontier) · `train_cart.py` (minimal recitation trainer) ·
`ao_insitu.py` / `ao_freegen.py` (oracle readout).

## Running

WSL2 + CUDA, `uv`-managed Python 3.12, the [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)
package, and the released Qwen3-4B oracle (`adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B`).
Neither is vendored here — install from upstream. Scripts expect the cartridges package on the path
and a corpus file at the `TEXT = ...` path near the top of each script.

**Corpus not included.** Experiments used a passage from a copyrighted novel (verified *not*
memorized by base Qwen3-4B, so the cart is unambiguously doing the work). That text is deliberately
kept out of this repo — supply your own passage. The findings reproduce with any non-memorized prose.

## References

- Cartridges — [arXiv 2506.06266](https://arxiv.org/abs/2506.06266) · [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)
- Activation Oracles — [arXiv 2512.15674](https://arxiv.org/abs/2512.15674)
- CAS (composition of adapters) — [arXiv 2606.04557](https://arxiv.org/abs/2606.04557)
- Keys as Shareable Routers (correlational cartridge interp) — [arXiv 2508.17032](https://arxiv.org/abs/2508.17032)
- Elhage et al., *A Mathematical Framework for Transformer Circuits*

---
*Author: Taggart Tufte. Status: trigger arc wrapped 2026-07; parked pending writeup.*
