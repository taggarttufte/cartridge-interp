# Interpreting Cartridges with Activation Oracles

> **🚧 Work in progress.** A toy-scale interpretability study, shared as a reference artifact —
> not a polished release. Findings are early and the code is research-grade (single-GPU scripts,
> hard-coded paths). Read [`SUMMARY.md`](SUMMARY.md) for the narrative and [`FINDINGS.md`](FINDINGS.md)
> for the raw per-experiment logs.

## What this is

Can we read what a **Cartridge** (a trained KV cache; [Hazy Research, arXiv 2506.06266](https://arxiv.org/abs/2506.06266))
has stored — and how that content is structured — by pointing an **Activation Oracle**
(an adapter that describes a model's residual-stream activations in natural language;
[Karvonen et al., arXiv 2512.15674](https://arxiv.org/abs/2512.15674)) at it?

The existing cartridge-interpretability work is purely *correlational* (SVD / cosine / attention viz).
The angle here is **AO-based / causal** readout. Everything runs at toy scale: **Qwen3-4B** on a single
12 GB RTX 3080 Ti. Carts are trained by **naive next-token recitation** (memorize a fixed passage) on
purpose — a memorizer is a crisp, ground-truthable object to interpret, even though it isn't the
paper's self-study method.

## Findings so far

- **Opaque at rest, legible in motion.** Feeding a cart's own KV vectors (`W_O·V`) to the AO is null at
  every cart size tried (n=5) — but reading the AO on the activations the cart *generates* works cleanly
  and is distinguishable from a random cart. You read a cart by **running** it, not by inspecting it.
  (Likely cause: an isolated layer's KV is out-of-distribution vs the hidden states the AO trained on.)
- **Capacity is content-dependent.** One KV slot losslessly recites **≥1024 tokens** of structured text
  (GPU-bound above that, not capacity-bound) — but only ~tens of *random* tokens.
- **Carts store the delta vs the model's priors.** Across coherent / shuffled / random content, starting
  loss (per-token surprise) predicts compressibility; random never fits. Destroying token *order* alone
  ~doubles the cost. The ordering is invariant to cart size.
- **Composition is real but partial.** A jointly-trained two-topic cart reads as both topics, and
  `span(AB)` overlaps `span(A,B)` ~6× above chance — but only ~30–57%, not a clean basis.

See [`SUMMARY.md`](SUMMARY.md) §6 for candidate next directions (tighten composition, causal ablation,
decode-without-running, self-study carts).

## Repo layout

| Path | What |
|---|---|
| [`SUMMARY.md`](SUMMARY.md) | Synthesized narrative writeup (start here) |
| [`FINDINGS.md`](FINDINGS.md) | Lab notebook — per-experiment tables and logs |
| [`OUTLINE.md`](OUTLINE.md) | Original project plan / mechanics / method design |
| `scripts/` | Training + readout scripts (one per experiment) |

Key scripts: `train_cart.py` (minimal recitation trainer) · `ao_shakedown.py` (AO positive control) ·
`ao_insitu.py` / `ao_freegen.py` (in-situ & free-generation readout) · `phase2_readout_sweep.py`
(direct-vs-generated readout, n=5) · `sweep_passage_len.py` / `ceiling_hunt.py` (capacity) ·
`topic_focus.py` / `topic_across_sizes.py` (delta-vs-priors) · `compositionality.py`.

## Running

Environment: WSL2 + CUDA, `uv`-managed Python 3.12, the [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)
package, and the released Qwen3-4B AO (`adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B`).
These are **not vendored here** — install them from upstream. Scripts expect the cartridges package on
the path and a corpus file at the path set near the top of each script (`TEXT = ...`).

**Corpus not included.** Experiments used a passage from a copyrighted novel (verified *not* memorized by
base Qwen3-4B, so the cart is unambiguously doing the work). That text is kept out of this repo by design
— supply your own passage and point `TEXT` at it. The findings reproduce with any non-memorized prose.

## References

- Cartridges — [arXiv 2506.06266](https://arxiv.org/abs/2506.06266) · [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)
- Activation Oracles — [arXiv 2512.15674](https://arxiv.org/abs/2512.15674)
- Keys as Shareable Routers (correlational cartridge interp) — [arXiv 2508.17032](https://arxiv.org/abs/2508.17032)
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (QK/OV-circuit language)

---
*Author: Taggart Tufte. Status: active WIP, 2026-05.*
