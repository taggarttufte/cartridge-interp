# Interpreting Cartridges with Activation Oracles — Work Summary

*Toy-scale interpretability study. Qwen3-4B, single 12 GB GPU (RTX 3080 Ti). 2026-05.*
*Full per-experiment logs in `FINDINGS.md`; this is the synthesized narrative.*

---

## 1. Question and the gap

**Cartridges** (Hazy Research, arXiv 2506.06266) are trainable KV caches: a small, frozen-base-model
"prefix" that stands in for a long context. **Activation Oracles** (AOs; Karvonen et al.) are a LoRA
adapter that reads a model's residual-stream activations and describes them in natural language.

The question: **can an AO tell us what a cartridge has stored, and how that content is structured?**

The gap this targets: the one existing cartridge-interpretability paper ("Learned Structure in
Cartridges: Keys as Shareable Routers," arXiv 2508.17032) is purely **correlational** (SVD, cosine,
attention viz). Nobody has done **causal / AO-based** interpretation. That is the niche here.

**Deliberate scoping choice.** We train carts by **naive next-token recitation** (memorize a fixed
passage), *not* the paper's self-study. Naive carts are "dumber" (they replay text, they don't
generalize to QA — the paper itself notes naive training "is not competitive with ICL"), **but they
are a crisp, ground-truthable object**: we know exactly what is "in" the cart, so every readout claim
can be checked. Interpreting self-study carts is the richer future target (see §6).

## 2. Setup

- **Subject model:** Qwen3-4B (36 layers, d_model 2560, 32 query / 8 KV heads, head_dim 128).
- **AO:** `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` (LoRA), reads layer **18** (~50% depth).
- **Content:** a Shadow Slave passage **verified not memorized** by base Qwen3-4B (so a cart that makes
  it recite is unambiguously doing the work). Plus synthetic content for later experiments.
- **Controls throughout:** random-cart baselines (negative), AO-on-real-activations (positive ceiling —
  passed: it reads plain passages correctly).

## 3. Findings

### 3.1 A cart is opaque at rest, but legible in motion

- **Direct readout — NULL.** Feeding the cart's own layer-18 KV vectors (`W_O·V` write-direction,
  `W_Qᵀ·K` listen-direction) straight to the AO yields random, wrong topics — indistinguishable from
  random-vector controls. **Replicated across carts of 32 / 64 / 128 / 256 / 512 tokens (n=5).**
  *Interpretation:* a cart's per-layer KV are not hidden states — they are parameters tuned jointly
  across all 36 layers to produce a behavior, so an isolated layer's vector is out-of-distribution for
  the AO.
- **Free-generation readout — WORKS.** Install the cart, let it **generate**, and read the AO on the
  activations of the **generated tokens**. The AO lands on-target (e.g. all five Shadow-Slave carts
  read as "a young man…"; the 512-token cart reads nearly verbatim) and is clearly distinguishable
  from a random cart ("a mathematical problem"). **Also n=5.**
- **Sampling position is decisive.** Reading at a *neutral prompt* position (before any content is
  emitted) is null; only **content-bearing generated positions** carry the signal.

> **Takeaway:** you cannot read a cart by inspecting its weights; you read it by running it and
> observing the activations it produces. "Opaque at rest, legible in motion."

### 3.2 Capacity — how much one slot holds, and that it depends on content

- **A length-1 cart (one KV slot ≈ the cache footprint of a single token) losslessly free-recites
  ≥1024 tokens** (~760 words) of structured text from a one-token seed — perfect token-for-token,
  trained in ~100–200 steps. We stopped at 1024 only because 2048-token *training* OOMs the 12 GB card
  (eager attention's L² forward), **not** because the cart filled up. So **≥1024 : 1 lossless, per slot.**
- **Capacity is content-dependent.** Replace structured text with **random tokens** and a single slot
  cannot hold even ~128 of them cleanly (longest exact prefix ~79/128, and recitation collapses by 256).
  So the ≥1024:1 ratio holds **only for compressible content**; incompressible content overflows the
  same slot below 256.
- **Comparison to the paper (same axis).** The paper's "125× compression" = a 128k-token context held
  in a ~1024-slot cache → **~125 tokens/slot**, but that payload is *functional* (answers arbitrary
  questions) and *lossy on surface form*. Ours (~1024 tokens/slot) is *verbatim* but *trivially
  queryable* (it just replays one passage). Our higher number is the easy-content/easy-task discount,
  not a capability win.

### 3.3 Mechanism — a cart stores the delta against the model's priors

We trained carts on four content types at fixed length/budget: **A** coherent passage, **B** unrelated
paragraphs jumbled, **C** = A's exact tokens shuffled (same words, no order), **D** random tokens.
Difficulty was read three consistent ways — starting loss (per-token surprise), steps-to-converge, and
whether it converged/recited at all:

| | A coherent | B unrelated | C shuffled-A | D random |
|---|---|---|---|---|
| start loss (surprise) | 3.0 | 2.2 | 7.9 | 13.2 |
| steps → loss 0.01 | 70 | 58 | 112 | never |
| converged / recited? | yes 100% | yes 100% | yes 100% | **no (~2%)** |

- **Difficulty order B < A < C < D** (easier → harder), and the **starting loss alone predicts final
  compressibility.** Destroying *order* (A → C, same tokens) makes it ~2× harder; destroying *all*
  structure (D) makes it impossible in budget.
- **Coherence ≠ low surprise:** dry boilerplate (B) compressed *easier* than vivid fiction (A) — what
  matters is per-token predictability vs the model's priors, not topical coherence.
- **This ordering is consistent across cart sizes** (cart_len 1/2/4): the start-loss tiers
  {A,B} ≪ C ≪ D are invariant to cart capacity, confirming compressibility is **content-intrinsic**.

> **Takeaway:** a cart only needs to store the **difference between the corpus and what the frozen
> model would already predict.** Predictable text = tiny delta = cheap; random = huge delta = can't fit.
> This is exactly *why* the paper's self-study works (it rides the base model's knowledge).

### 3.4 Composition — real but partial linear structure

Trained `cart_A` (giraffes), `cart_B` (volcanoes), `cart_AB` (both); length 4 each. (Carts are **not**
added slot-wise — slots have no canonical order — so composition was tested by **concatenation**
(behavioral) and **subspace arithmetic** (geometric, permutation-invariant).)

- **Behavioral:** the AO reads `cart_AB` as **both** topics (giraffe early in its generation, volcano
  later). A *concatenated* A++B cart contains both but a giraffe-seeded greedy decode only surfaced
  giraffe — the same sampling-position limitation, not absence.
- **Subspace addition:** `span(cart_AB)` overlaps `span(cart_A, cart_B)` at **0.49** averaged over all
  36 layers vs **0.10** chance (**4.9×**); layer-18 alone was 0.57.
- **Subspace subtraction:** removing A's directions from AB leaves **0.20** (multi-layer mean) in
  `span(cart_B)` vs **0.05** chance (**3.9×**); layer-18 alone was 0.30.
- **Robust and depth-structured:** the overlap holds at *every* layer (never chance, never a clean
  basis), and the additive structure **concentrates in the deep layers** (L34–35 ≈ 0.73–0.78). The
  multi-layer mean is slightly *below* layer 18, i.e. layer 18 modestly flattered composition — so the
  partialness is genuine, not a single-layer artifact.

> **Takeaway:** content composes as **partially shared directions** — well above chance (~4–5× at every
> layer), but ~20–50%, *not* a clean basis. Supports the weaker "additive structure" form of the linear
> representation hypothesis; does **not** establish clean concept arithmetic. (Geometry can't separate
> "different directions, same behavior" from "genuinely partial" — that needs the causal test.)

## 4. What this is NOT (limitations)

- **Naive carts, not self-study.** Everything above is about *memorization* carts. Self-study carts
  (functional, queryable) are a different and richer object — untouched here.
- **Toy scale.** One short passage / two topics; not 128k-token corpora.
- **Composition geometry has a ceiling (single-layer caveat now resolved).** The multi-layer redo
  (all 36 layers) settled the layer-18 worry — it *lowered* the overlap slightly (0.49/0.20), so layer
  18 wasn't deflating it. Procrustes alignment is a non-fix here: the span metric is already
  rotation/permutation-invariant, so forcing alignment could only inflate. The real remaining limit is
  that geometry can't distinguish "different directions, same behavior" from "partial composition" —
  only causal ablation can.
- **Capacity-linearity is unresolved.** The cart-length-vs-capacity sweep used random tokens, which
  turned out optimization-bound, not capacity-bound, so it can't say whether capacity is linear in
  slots. Needs a structured-text redesign.
- **Perf wall.** 512-token eager-attention training sits at the 12 GB edge and thrashes; N_CTX ≤ 256
  (or compiled FlexAttention / Qwen3-1.7B) is required for headroom.

## 5. Relation to prior work

The cartridge-specific literature is just two papers: the original (self-study; 38.6× memory / 26.4×
throughput / up to 256× cache compression) and the correlational routers follow-up. Our **causal +
AO** angle is open territory. Notably the routers paper found *keys stable, values grow in singular
value* — independently corroborating our keys-as-routers / values-as-content split and the idea that
content occupies an expanding value subspace.

## 6. Open questions / candidate next directions

1. **Tighten composition** — multi-layer subspace + Procrustes alignment (removes confounds likely
   deflating 0.57/0.30). Cheap, high-value.
2. **Causal ablation / attribution** — knock out cart components (layer / head / slot / SVD direction)
   on a two-topic cart and map which component controls which content. The flagship *causal* result.
3. **Capacity-linearity, done right** — structured-text capacity vs cart length.
4. **Decode a cart without running it** — the open prize; direct readout is null, so try multi-layer
   SVD directions, a probe trained on cart vectors, or the stronger Qwen3-8B AO.
5. **Interpret a self-study cart** — the highest-value, hardest target: does the AO read *functional*
   content (not just replayed text)? Closest to what's actually deployed.

**Headline so far:** *a memorization cart is opaque at rest but legible in motion; one slot holds
≥1024:1 of compressible content (capacity is content-dependent); it stores the delta against the
model's priors; and its content composes as partially-shared linear directions.*
