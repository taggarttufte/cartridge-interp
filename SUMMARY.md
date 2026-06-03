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

### 3.5 Causal specificity — the flagship

Per layer, project a topic's write-subspace out of `cart_AB`'s V (per-(layer, kv-group, slot)
least-squares), free-generate from a giraffe-seed, and measure the surviving content.

| cart | giraffe-recite | volcano-best-window | AO last-16 |
|---|---:|---:|---|
| cart_AB (no ablation) | 1.000 | 1.000 | volcanoes |
| **ablate_A** (giraffe out) | 0.032 | 0.047 | off-topic (film) |
| **ablate_B** (volcano out) | 0.825 | **0.250** | volcanic eruptions |
| ablate_rand (control) | 1.000 | 1.000 | volcanoes |

- **Random ablation is null** — same as baseline. The LS pipeline doesn't damage the cart on its own,
  so non-random effects are content-specific. Methodology passes.
- **`ablate_B` is the clean causal hit.** Removing the volcano subspace breaks verbatim volcano
  (1.00→0.25) while giraffe survives (1.00→0.825). The volcano *topic* still surfaces in the AO read
  and the gen text contains real volcano semantics — but paraphrased, not verbatim. So we suppressed
  the cart's volcano-specific contribution; the model fell back on its pretrained volcano knowledge.
  **Topic-specific, asymmetric, controlled** — the causal claim the correlational literature can't make.
- **`ablate_A` reveals sequence-conditional structure.** Both topics collapse — but `cart_AB` was
  trained autoregressively on idsA++idsB, so the volcano portion conditions on having first generated
  giraffe. Killing the early content severs the path to the later content. New finding: multi-topic
  carts are **sequence-conditional**, not bag-of-topics; geometry alone misses the autoregressive
  coupling.
- **Carts = verbatim deltas; the model = topic priors.** `ablate_B`'s pattern (verbatim broken, topic
  AO-readable) is exactly what §3.3 predicted. The causal test makes that split *mechanistic*.

> **Takeaway:** composition is **partial but causally separable**. Ablating a topic's write-subspace
> selectively suppresses that topic's verbatim production while the other survives, the topic *survives
> in the AO read* (model priors carry the semantic), and a sequence-conditional asymmetry between A
> and B exposes structure geometry misses. The flagship causal result lands.

### 3.6 Behavioral test — a recitation cart recites an instruction but does not enact it

Train a length-1 cart by naive recitation on a short instruction ("respond like a pirate"; "respond
only as a question"), then test 6 held-out queries in three conditions — baseline (no cart),
in-context (instruction in the prompt = ceiling), and cart (instruction only in the cart). Run on both
`Qwen3-4B` and `Qwen3-4B-Base`.

| condition | pirate-word hits (base / instruct) |
|---|---|
| baseline | 0 / 0 |
| in-context (ceiling) | 29 / 25 |
| **cart loaded** | **0 / 0** |

Both models recite the instruction verbatim from the cart and both obey it perfectly when it is in the
prompt — yet with only the cart loaded, outputs are byte-identical to baseline. The recitation cart
stores the instruction's **surface form, not its operative meaning**: a token-emitter, not a
behavior-conditioner. This holds for base *and* instruct, so it is a property of the recitation
objective, not of post-training — the direct behavioral counterpart of §3.3 (the cart stores the delta
needed to reproduce *tokens*, a different object from the delta that would steer *behavior*). Aside:
`Qwen3-4B-Base` follows in-context instructions as well as the instruct model and defaults to an
assistant persona, so it is not a clean raw base. Natural next step: a self-study / behavioral cart
(query→instructed-answer pairs) to test whether a cart *can* carry behavior at all.

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

1. ~~**Tighten composition** — multi-layer subspace.~~ **DONE (§3.4):** mean 0.49/0.20, robust across
   layers, peaks at L34–35. Procrustes was a non-fix.
2. ~~**Causal ablation** — knock out cart components on a two-topic cart.~~ **DONE (§3.5):** ablating
   the volcano subspace breaks volcano verbatim while giraffe survives; random control is null.
   Follow-ups worth chasing: (a) **Method 2** — ablate cart_AB's *own* SVD directions labeled by
   alignment to cart_A (bites the unshared giraffe directions too); (b) **disentangle the ablate_A
   asymmetry** — train cart_AB on shuffled / interleaved A and B segments to separate
   "giraffe-suppression" from "sequence-disruption"; (c) **layer- and head-resolved ablation** — bisect
   to find the smallest causal set.
3. **Capacity-linearity, done right** — structured-text capacity vs cart length.
4. **Decode a cart without running it** — the open prize; direct readout is null, so try multi-layer
   SVD directions, a probe trained on cart vectors, or the stronger Qwen3-8B AO.
5. **Key-transfer / weight-robustness** — freeze keys, retrain only values on a new corpus. Causal
   test of the routers paper's central claim, and the cleanest engagement with the existing literature.
6. **Interpret a self-study cart** — the highest-value, hardest target: does the AO read *functional*
   content (not just replayed text)? Closest to what's actually deployed.

**Headline so far:** *a memorization cart is opaque at rest but legible in motion; one slot holds
≥1024:1 of compressible content (capacity is content-dependent); it stores the delta against the
model's priors; its content composes as partially-shared linear directions; and ablating a topic's
write-subspace causally suppresses that topic's verbatim production while leaving the other intact
and letting the model's priors carry the surviving semantic — multi-topic carts also turn out to be
sequence-conditional, not bag-of-topics.*

---

## Appendix A — Cart data-flow (architecture)

Qwen3-4B: **36 layers**, d_model **2560**, **32 query heads / 8 KV heads** (GQA group 4), head_dim
**128**. A length-1 cart = 36 independent stored (key, value) pairs (~74k params), one per layer.

**Macro — the sequential spine.** The residual stream flows upward through the 36 layers in order; the
cart is a stack of 36 stored (K,V) entries, all existing at once, each injected into its own layer.

```
        CART  =  36 stored (key, value) pairs   <- parameters, NOT computed
        +----------+----------+------- ... ------+----------+
        | (K,V)_1  | (K,V)_2  |                  | (K,V)_36 |   (all exist at once = PARALLEL)
        +----+-----+----+-----+----- ... ---+----+----+-----+
             |inject     |inject            |         |inject
             v           v                  v         v
 prompt -> [Layer 1] -> [Layer 2] -> ... -> [Layer 36] -> unembed -> logits -> next token
 tokens     +------------ residual stream, SEQUENTIAL upward ----------->
```

**Micro — inside one layer's attention (one query position).** Two paths feed one softmax in parallel:
the *token path* computes K/V from hidden states; the *cart path* supplies K/V directly.

```
         +---------------- RESIDUAL STREAM (R^2560) ----------------+
         |                                                          |
 query   h_q --W_Q--> q                                             |
 token                |   ----- all keys compete in ONE softmax -----
   TOKEN PATH         |
   past tok h_j -W_K-> k_j        CART PATH (parallel, no W_K/W_V):
            h_j -W_V-> v_j           key_cart    <- stored
                      |              value_cart  <- stored
                      v
        scores = q . { k_j , ... , key_cart } --> softmax --> weights a_i
                      |
        blend  = sum_i a_i . { v_j , ... , value_cart }   (token + cart content, weighted)
                      | W_O
                      v
                     Dh  ----------- added back ----------> h_q + Dh   (then MLP, next layer)
```

The cart's **key** sits in the score (decides *how much* attention the slot gets); the cart's
**value** sits in the blend (*what content* gets mixed in). That is the keys-as-routers /
values-as-content split, mechanically.

**The two probe vectors** (the cart's only two contact points with the residual stream, pulled back
into R^2560):

```
 listen direction   u = W_Q^T . key_cart    ->  "which query patterns attend here"   (ADDRESS)
 write  direction   w = W_O  . value_cart   ->  "what gets added to the residual"     (CONTENT)
```

`w` lives at the *output* of the blend; `u` lives at the *score* stage. Aaron's probe sums `w` (and
`u`) over **all heads in the layer** — the totality the slot writes / is dotted against — rather than
reading per-head fragments.

**Parallel vs. sequential.**
- *Sequential:* layers 1->36 (residual transformed in order); positions during generation
  (autoregressive); within a layer, attention -> residual-add -> MLP -> residual-add.
- *Parallel:* the 36 cart (K,V) entries (stored at once); the 32/8 heads within a layer; the cart path
  alongside the token path in the same attention; all positions' keys/values inside one softmax.
