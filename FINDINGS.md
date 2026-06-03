# Cartridge Interpretability — Findings Log

Subject model: **Qwen3-4B** (36 layers, d_model 2560, 32 Q / 8 KV heads, head_dim 128).
AO: `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` (LoRA), reads layer **18** (50% depth).
Template: **Shadow Slave v1**, Ch.1 opening (verified *not* memorized by base Qwen3-4B).

---

## Setup that works (reusable)

- **Cart training** (`scripts/train_cart.py`): freeze Qwen3-4B, train only a `TrainableCache` (prefix-tuning) by next-token CE on the passage with the cart prepended. **Run eager** (`TORCHDYNAMO_DISABLE=1`) — FlexAttention autotuning on the 3080 Ti takes ~30 min/kernel and many candidates OOM the 101 KB shared-mem limit; eager runs the whole thing in ~1 min.
  - A **length-1 cart memorized a 32-token passage**: next-token accuracy 0.48 (random cart) → **1.00 by step 50**, loss → 0.
- **AO inference** (`scripts/ao_shakedown.py`): runs in the cartridges venv with `nl_probes` on `sys.path`, `attn_implementation="sdpa"`. **Positive control passes** — fed the AO real layer-18 activations of plain passages and it described them correctly (Eiffel Tower, giraffes, and the Shadow Slave opening → *"a young man sitting on a bench across from a police station"*). This is our **ceiling**.

---

## Experiment 1 — Approach (b): read the cart's *contents* directly. **RESULT: NULL.**

**Method.** From the trained length-1 cart's layer-18 K/V, construct residual-space probe vectors
(`scripts/extract_probe_vectors.py`):
- **write-direction** `W_O_h · V_g` (exact — what the entry writes to the residual stream),
- **listen-direction** `W_Q_hᵀ · K_g` (approx — ignores q_norm/RoPE; what it routes toward),
- GQA handled (8 KV heads → 32 query-head blocks).

Feed these to the AO (`scripts/ao_probe_cart.py`) and ask what they encode.

**Result.**

| probe | AO reading | correct? |
|---|---|---|
| `write_kvhead` (8) | "a conversation … discussing a new product" | ✗ |
| `write_qhead` (32) | "a medical condition" | ✗ |
| `listen_qhead` (32) | "the use of a computer" | ✗ |
| RANDOM (8) | "impact of technology on the music industry" | (noise) |
| RANDOM (32) | "the word 'water'" | (noise) |
| **ceiling** (real activations) | "a young man on a bench across from a police station" | ✓ |

The cart probes are **indistinguishable from random vectors** — the AO confabulates a plausible-but-wrong topic every time, never approaching the ground truth.

**Interpretation.** The passage *is* in the cart (it recites at 100%), but it is **not legible as a single layer's constructed KV vector.** A cart's per-layer K/V are **not hidden states** — they are parameters tuned *jointly across all 36 layers* to produce a behavior, so an isolated `W_O·V` is genuinely out-of-distribution versus the layer-18 *hidden states* the AO was trained on. Direction-normalization (which the AO does) doesn't rescue it. (Consistent with the authors' warning that small 1–4B AOs are weak out-of-distribution.)

**Takeaway.** Rules out the naive direct-readout. Motivates **approach (a)**: read the cart's *effect* via genuine, in-distribution activations.

---

## Experiment 2 — Approach (a): read the cart's *effect*. **RESULT: NULL.**

**Method** (`scripts/ao_insitu.py`): load FlexQwen3 + cart, run a neutral prompt
("Summarize the document above in one sentence:") with the cart as context, capture
the genuine layer-18 hidden states at the prompt tokens, free the GPU, feed those to
the AO. Random cart = baseline.

**Result.**

| probe | AO reading | correct? |
|---|---|---|
| SHADOW cart (in-situ) | "the impact of climate change on the Arctic region" | ✗ |
| RANDOM cart (baseline) | "the potential benefits of a new technology" | ✗ |
| **ceiling** (real activations) | "a young man on a bench across from a police station" | ✓ |

Trained cart reads **the same as a random cart** — no Shadow-Slave signal.

**Interpretation — the likely root cause is the cart, not the readout.** The cart was
trained **teacher-forced**: at each step the model already saw the real previous tokens,
so the cart only supplied *marginal* next-token help and never had to carry the content
alone. Hence it recites at 100% *with* the passage as scaffolding, but exerts only a
**weak standalone steering force** on a neutral prompt — too weak to color activations
legibly. A teacher-forced length-1 cart may simply be a weak object to interpret.

---

## Where this points (next directions, in rough priority)

1. **Train a cart that carries content by itself.** Either (a) **free/autoregressive
   recitation** (no teacher prefix — the cart must drive generation), or (b) the paper's
   **self-study** distillation. Then re-probe with both approach (a) and (b). *Most
   promising — attacks the root cause.*
2. **Free-generation readout:** seed only BOS, let the cart generate (recitation check),
   read the activations of what it produces (strongest cart-effect signal).
3. **Bigger cart:** length-2/4/8 — more capacity → stronger, more legible content.
4. **Stronger AO:** Qwen3-8B AO (less weak out-of-distribution).
5. **Other layers** (9, 27) and **giraffe-vs-shadow differential** for faint signal.

## Experiment 2b — Approach (a'): free-generation readout. **RESULT: POSITIVE (with caveat).**

**Method** (`scripts/ao_freegen.py`): seed only the first passage token ("A"), let the cart
**generate** (free-recitation check), then read the layer-18 activations of the *generated*
tokens with the AO. Random cart = baseline.

**Result.**
- **Free recitation = 100%** over 31 tokens from a 1-token seed — the length-1 cart fully
  recites the passage (and continues coherently past the training cutoff). → **the
  "weak teacher-forced cart" hypothesis is FALSIFIED; the cart carries its content.**
- AO read of generated-token activations:

  | probe | AO reading |
  |---|---|
  | SHADOW (generated) | "a scene of a man waiting in a dimly lit alley" |
  | RANDOM (generated) | "a mathematical problem" (random cart generated a math question) |
  | ceiling | "a young man on a bench across from a police station" |

  Shadow lands in the right semantic territory (a man, waiting, a scene) and is clearly
  **distinguishable from random**. So Experiment 2's null was largely a **sampling-position
  artifact** — content is legible at content-bearing (generated) positions, not at neutral
  prompt tokens or in raw KV vectors.

**Caveat.** This reads the cart's *effect* (let it generate, read the behavior). Useful —
you could discover an *unknown* cart's content by generating from it — but it's close to the
ceiling condition (once the cart emits the passage we read passage-token activations). It does
**not** crack the harder goal of decoding a cart's *compressed* representation **without
running it** (approach b remains null).

## Net so far
Full pipeline works end-to-end (train → extract → generate → AO read), with positive/negative
controls. Established: (1) a length-1 cart genuinely carries a 32-token passage (free-recites
at 100%); (2) **direct** readout of its KV vectors is null (approach b); (3) reading its
**effect via generation** works and is distinguishable from random (approach a'); (4) reading
its effect at *neutral* positions is null — sampling position matters. **Open question:** can a
cart's compressed contents be decoded *without* running it? That's the real interpretability
prize, still unsolved.

---

## Capacity sweep — how much text fits in ONE slot? (`scripts/sweep_passage_len.py`)

Held cart length = 1, swept passage length. **Free-recitation** (seed 1 token, let the cart
drive) is the honest metric; `prefix` = tokens before the first slip.

| passage tokens | ≈ words | teacher-forced | free-recite | correct prefix |
|---|---|---|---|---|
| 32 | ~24 | 1.000 | 1.000 | 31/31 |
| 64 | ~48 | 1.000 | 1.000 | 63/63 |
| 128 | ~96 | 1.000 | 1.000 | 127/127 |
| 256 | ~190 | 1.000 | 1.000 | 255/255 |
| 512 | ~380 | 1.000 | 1.000 | 511/511 |
| 1024 | ~760 | 1.000 | 1.000 | 1023/1023 (grad-checkpointed, 10.4 GB) |
| 2048 | ~1520 | — | — | OOM on eager-attn **forward** L² (GPU limit, not cart) |

**One KV slot losslessly free-recites ≥1024 tokens** (≈760 words) from a single seed token, and
trains in ~100–200 steps (loss 6e-4 at 1024). A length-1 cart is ~74k params (36 layers × 8 KV
heads × 128 dim, K and V) and saves to a fixed ~173 KB **regardless** of passage length — fixed
slot, variable content. **Verbatim compression ratio ≥ 1024:1.** No accuracy bend through 1024;
2048 is untrainable here only because eager FlexAttention materializes a per-layer L×L score
matrix in the *forward* pass (gradient checkpointing fixes backward storage but not that forward
transient) — a GPU limit, not the cart. **But the ceiling is content-dependent — see topic-focus:
random tokens overflow the slot below 512.** So structured ≥1024:1 vs random <512:1: capacity
depends on compressibility, not raw token count.

**Paper context:** this is the *naive next-token* regime the Cartridges paper (2506.06266)
explicitly calls "not competitive with ICL" — they use **self-study** (synthetic Q&A +
distillation) and measure **downstream QA/translation**, reporting 38.6× memory / 26.4×
throughput at 128k→484k-token scale. Our verbatim, lossless, toy-scale capacity is a
*different, complementary axis*; consistent with their "naive = great memorization, poor
generalization" claim. Not directly benchmark-comparable.

## Phase 2 — readouts REPLICATE across cart sizes (`scripts/phase2_readout_sweep.py`). n=5.

For each saved cart (32/64/128/256/512), ran both readouts + a random-cart control. (a′) reads
the FIRST 16 generated tokens (same opening for every cart → controlled comparison).

| cart | (b) direct W_O·V | (a′) free-gen first-16 |
|---|---|---|
| 32  | (noise) | "young man's journey … a mysterious illness" |
| 64  | "two friends … weekend plans" | "young man's experience with a mysterious illness" |
| 128 | "a new law on the fishing industry" | "young man's journey … about his identity" |
| 256 | "the word 'NAME_1'" | "the story of a young man named NAME_1" |
| 512 | "the concept of a person's age" | **near-verbatim recall of the passage's opening description** (AO output redacted — copyrighted source) |
| RAND | "impact of technology on society" | "a mathematical problem … the median" |

- **(a′) = signal at every size**: all 5 say "young man" + narrative (on-target); 512 reads
  nearly verbatim; random cart reads "a mathematical problem" (clearly distinguishable). Exp 2b
  replicates n=5.
- **(b) = null at every size**: every reading wrong, mutually unrelated, and no different in
  character from the random cart's direct reading. Exp 1's null is robust 32→512. (512's
  "person's age" is a faint coincidence at most.)
- **Conclusion:** the cart carries readable content via its *behavior* but its *raw KV* stays
  opaque — consistently across an order of magnitude of passage length.

---

## Topic-focus — does compression exploit STRUCTURE? (`scripts/topic_focus.py`)

All conditions 512 tokens, cart length 1, 400 steps, lr 2e-2. Metric = steps-to-converge +
final loss + free-recite (start loss = per-token surprise vs the frozen model's priors).

| condition | start loss | final loss | tf_acc | free-recite | →0.05 | →0.01 |
|---|---|---|---|---|---|---|
| A coherent (Shadow Slave) | 2.99 | 0.0006 | 1.00 | 1.00 | 98 | 107 |
| B unrelated (7 disjoint domains) | 2.17 | 0.0002 | 1.00 | 1.00 | 62 | 69 |
| C shuffled-A (A's tokens, scrambled) | 7.67 | 0.0066 | 1.00 | 1.00 | 139 | 227 |
| D random tokens | 13.2 | 0.248 | 0.975 | **0.025** | -- | -- |

Difficulty order (convergence speed + final loss): **B < A < C < D.**

- **Structure IS exploited (C vs A — the clean control):** identical tokens, order alone differs
  → shuffling makes convergence ~2× slower (→0.01 at step 227 vs 107) and final loss 10× higher
  (0.0066 vs 0.0006). The cart rides the model's ability to predict *ordered* text.
- **Capacity is content-dependent (D):** 512 *random* tokens overflow one slot — free-recite
  collapses to **2.5%**. (tf_acc 0.975 is misleading: small errors compound autoregressively with
  no structure to recover. Free-recite is the honest metric.) The 512:1 lossless ratio holds
  **only for structured text.**
- **Coherence ≠ low surprise (B < A):** dry formulaic multi-domain prose compresses *easier* than
  vivid fiction; the 7 topic jumps cost almost nothing. What matters is **per-token surprise vs
  the model's priors**, not topical coherence. (Corrects the initial "incoherent = harder" guess.)
- **Mechanism confirmed by START loss** = per-token surprise: A 2.99, B 2.17 (priors help a lot),
  C 7.67 (order destroyed), D 13.2 (≈max entropy ln|V|, priors useless). Final compressibility
  tracks start loss → **the cart stores the DELTA between the corpus and what the frozen model
  would already predict.** Directly supports the paper's self-study rationale (carts work by
  riding the base model's knowledge) and the "compose at inference" claim (structured deltas).

---

## Capacity scaling vs cart length — INCONCLUSIVE (random probe is optimization-bound). `scripts/capacity_scaling.py`

Q: is per-slot capacity linear in the number of slots? Probe = RANDOM tokens (incompressible →
exposes raw storage). Grid cart_len {1,2,4} × random {128,256,512}, 1000-step budget, early-stop
on convergence (never triggered — random never reaches loss<1e-3). Honest metric = longest
correct free-recite **prefix** (free-recite *fraction* is corrupted by early-EOS shrinking the
denominator).

**Longest correct prefix:**

| cart_len | r128 | r256 | r512 |
|---|---|---|---|
| 1 | 79 | 24 | 38 |
| 2 | 79 | 24 | 38 |
| 4 | 79 | 24 | 38 |

**Prefixes are byte-identical across cart length** — 1→2→4 slots changed nothing (loss plateaus
barely moved too: r512 .178/.156/.147). 

**Interpretation — this probe can't answer the question.** Random-token capacity here is
**optimization-bound, not slot-bound**: the cart can't fit incompressible content fast enough in
1000 steps to use extra slots, so adding slots is invisible. This is *not* evidence capacity is
sublinear — it's evidence random tokens are the wrong probe. (Also: prefixes are non-monotonic in
length, 79/24/38, because each passage length used a different random seed → "prefix" = where the
first un-fittable token sits in *that* sequence, a content artifact, not a capacity law.)

**Redesign for the real linearity test:** use STRUCTURED text (converges cleanly, so capacity not
optimization is the binding constraint) — find each cart length's recitation boundary (max tokens
at ~100% free-recite) and check whether it scales with slots. Needs the long-passage +
gradient-checkpointing path already working from the ceiling hunt. Also: disable EOS during the
recite check so the free-recite denominator isn't truncated.

---

## Topic hypothesis across cart sizes — CONSISTENT. `scripts/topic_across_sizes.py`

Q1: does the structure-dependence (B<A<C<D) persist as cart length grows? Re-ran the 4 conditions
at cart_len 1/2/4, N_CTX **256** (512 eager-attn training thrashes the 12 GB card — see Perf note),
1000-step budget + early-stop, interval logging.

**start-loss (per-token surprise) / →0.01 steps:**

| | A coherent | B unrelated | C shuffled-A | D random |
|---|---|---|---|---|
| cart_len 1 | 3.06 / 70 | 2.24 / 58 | 7.96 / 112 | 13.16 / never |
| cart_len 2 | 3.04 / 64 | 2.24 / 79 | 7.95 / 101 | 13.17 / never |
| cart_len 4 | 3.05 / 65 | 2.24 / 48 | 7.95 / 116 | 13.15 / never |

(final loss: A/B/C all ~0.0004–0.0014 → memorized, recite 1.0; **D ~0.095, never converges, recite
0.227** — the cart_len 2 "recite 1.000" is the early-EOS denominator artifact, ignore it.)

**Verdict: topic-consistency HOLDS, in its meaningful form.**
- **Start-loss is cart-size-invariant and perfectly ordered B<A<C<D at every size** (≈2.24 / 3.05 /
  7.95 / 13.16 across the board). Since start-loss = per-token surprise, this is the cleanest proof
  that **compressibility is content-intrinsic, independent of cart capacity** — the core claim.
- **Difficulty tiers {A,B} ≪ C ≪ D are rock-solid at every cart size.** Structured ≪ structure-
  destroyed ≪ random, always.
- **D never memorizes at any cart size** (final ~0.10). More slots help D's loss only marginally
  (.0999→.0976→.0953 for 1→2→4) — consistent with Q2 ("slots barely help random at fixed budget").
- **Caveat:** the *fine* A-vs-B order wobbles (cart_len 2 has A<B, others B<A) — expected, A and B
  are both low-surprise and close (3.05 vs 2.24), so their convergence-*speed* gap is within noise.
  The robust claim is the **tier structure**, not the A/B micro-order.

## Perf note (the 512-token OOM lesson)
Training a cart at **512 tokens in eager FlexAttention sits at the 12 GB edge and thrashes** — peak
~12 GB, allocator churns, 100% GPU util but ~zero progress (the "stalls" of 25–93 min were this,
not sleep). Gradient checkpointing alone wasn't a durable fix at 512. **Fixes:** N_CTX ≤ 256
(4× less attention memory — what unblocked Q1), and/or try compiled FlexAttention (fused, O(L) mem,
5–10× faster — blocked only by the 3080 Ti autotune pathology, worth a one-time attempt), and/or
drop to Qwen3-1.7B for toy sweeps. Note: checkpointing's recompute makes non-converging cells
(C just-misses tol, D never) run the full 1000 steps at ~14 min each.

---

## Compositionality — are cartridges linear-compositional? PARTIAL. `scripts/compositionality.py`

A = giraffes, B = volcanoes (64 tok each), trained cart_A / cart_B / cart_AB (both, 128 tok),
all length 4. The concept-as-direction test (idea #4). NOTE: carts are NOT added slot-wise (slots
have no canonical order); the well-posed tests are concatenation (behavioral) + SUBSPACE arithmetic
(permutation-invariant).

**1+2. Semantic / concatenation (AO reads the cart's free-generation):**
- cart_A → "the giraffe, tallest living land animal" ✓; cart_B → "the study of volcanoes" ✓.
- **cart_AB surfaces BOTH** — AO(first16)="giraffes", AO(last16)="volcanoes and their eruptions"
  (it recites giraffe→volcano; the AO reads each region correctly). ✓✓
- cart_CONCAT (A's slots ++ B's slots, no retraining): both AO windows read *giraffe* only. The
  volcano content IS in the cart, but a giraffe-seeded greedy decode stays on giraffe and never
  transitions → **sampling-position limitation (Exp 2's lesson), not evidence volcano is absent.**

**3. Subspace ADDITION** — is span(cart_AB) ⊆ span(cart_A, cart_B)? (W_O·V at layer 18)
  **0.566** of AB's energy captured by span(A,B) vs **0.097** random-subspace baseline (dim 256) →
  **5.8× above chance**, but far from 1.0.

**4. Subspace SUBTRACTION** — after projecting A's directions out of AB, is the remainder ~ span(B)?
  **0.299** vs **0.049** random baseline → **6.1× above chance**, but far from 1.0.

**Verdict — PARTIAL linear composition.** There IS real compositional structure: content subspaces
overlap ~6× more than chance, so giraffe/volcano directions are substantially shared between the
separate and joint carts. But composition is only ~30–57% — the joint cart develops directions the
separate carts don't predict. Matches the earlier conceptual call: the data supports the **weaker
"additive structure" form of linearity** (above-chance shared directions) but **NOT clean concept
arithmetic** (can't perfectly reconstruct AB from A⊕B). So "concepts as addable/subtractable
directions": **partially yes — real and well above chance, not a clean basis.**

**Caveats / follow-ups before over-reading 0.57/0.30:** (a) single-layer test (only L18; content is
spread over 36 layers — try concatenating all layers' W_O·V); (b) independent-training rotation
freedom (cart_A vs cart_AB from different seeds → "giraffe" needn't sit in identical directions;
Procrustes-align, or init AB from concat(A,B), to separate basis-freedom from genuine
non-composition); (c) better concat behavioral readout (neutral/volcano seed to surface volcano).

### Tightening: multi-layer geometry (2026-05-25)

Addressed caveat (a) — re-ran the subspace test at **every one of the 36 layers** and aggregated.
Also resolved caveat (b) on paper: the energy-in-subspace metric is **already invariant to rotation
and slot permutation** (rotating M_A within its own span leaves span(M_A) unchanged), so Procrustes
alignment cannot change these numbers — and *forcing* an alignment would only inflate overlap. So the
honest fix is more layers, not an alignment. (Same-dim random baselines now averaged over 3 draws.)

| test | layer-18 (old) | multi-layer mean | chance | lift |
|---|---|---|---|---|
| ADD — span(A,B) ⊇ AB | 0.566 | **0.492** | 0.100 | 4.9× |
| SUB — (AB−A) → span(B) | 0.299 | **0.196** | 0.050 | 3.9× |

Per-layer: ADD min 0.32 / median 0.48 / **max 0.78**; SUB min 0.10 / median 0.21 / max 0.36.
Depth profile (ADD): ~0.3–0.5 through low/mid layers, **rising at the deep end (L34=0.73, L35=0.78)**.

**Result — the tightening did NOT raise the numbers; it lowered them slightly and made them robust.**
- Layer 18 (0.57/0.30) was **above the median layer (0.48/0.21)** — it slightly *flattered* composition,
  the opposite of the "single-layer deflates it" worry. The earlier hypothesis was wrong.
- Partial overlap is **network-wide**: ~5×/4× chance at *every* layer, never collapsing to chance,
  never reaching a clean basis (~1.0). "Partially shared linear directions" is now well-controlled,
  not a one-layer artifact.
- **Additive structure concentrates in the deep layers** (L34–35 strongest) — new, and a direct target
  for the causal ablation (#2): ablate a topic's subspace where it bites hardest.
- Geometry's ceiling stands: two carts can encode the same behavior via different directions, so
  geometric overlap can't separate "different directions / same behavior" from "genuinely partial
  composition." That separation needs the causal test.

---

## Causal ablation — the flagship result. `scripts/causal_ablation.py` (2026-05-26)

Same setup as compositionality (cart_A giraffe, cart_B volcano, cart_AB both, length 4, 64 tok/topic).
Per layer L, compute the giraffe write-subspace Q_A^L = orthobasis(M_A^L) (dim 128 every layer),
the volcano Q_B^L, and a same-dim random orthonormal Q_rand^L. Build three ablated copies of cart_AB
by modifying V via per-(layer, kv-group, slot) least-squares so the per-q-head residual write
(W_O[:, h_block] @ V[g,t]) is projected onto (I − QQ^T). Keys unchanged. Then free-generate from each
(giraffe-seed, 128 new tok) and measure giraffe-recite (first 63 tok vs idsA[1:64]) and
volcano-best-window (max 64-window match vs idsB[:64]).

| cart | giraffe-recite | volcano-best-window | AO first-16 | AO last-16 |
|---|---:|---:|---|---|
| cart_AB (no ablation) | 1.000 | 1.000 | giraffes | volcanoes / impact on the landscape |
| cart_AB ablate_A (giraffe out) | **0.032** | 0.047 | film "Thea" | film "The Shape of Water" |
| cart_AB ablate_B (volcano out) | 0.825 | **0.250** | giraffes | volcanic eruptions |
| cart_AB ablate_rand (control) | 1.000 | 1.000 | giraffes | volcanoes / geological impact |

**1. Random ablation is null (control passes).** Identical to baseline on both metrics and both AO
reads. The LS pipeline is benign — cart writes have negligible energy in random d_model directions,
so I-QQ^T barely changes them and LS reconstruction is near-identity. Methodology is sound; any
non-random effect below is content-specific, not LS-induced damage.

**2. ablate_B is the clean causal hit (the flagship).** Removing the volcano write-subspace breaks
volcano verbatim recite (1.00 → 0.25) while leaving giraffe nearly intact (1.00 → 0.825). The
volcano *topic* still surfaces in the AO read ("volcanic eruptions") and the gen text contains real
volcano semantics ("eruption forces its way to the surface, lava flows, pyroclastic debris") — but
*paraphrased*, not verbatim. So we suppressed the cart's volcano-specific contribution; the model
fell back on its pretrained volcano knowledge to produce the topic. **Topic-specific, asymmetric,
controlled — the causal claim the correlational literature can't make.**

**3. ablate_A is dramatic AND structurally informative.** Giraffe drops to 0.032 — but volcano also
drops to 0.047, and the cart generates off-topic film content. The random control rules out
methodology failure, so the reading is structural: cart_AB was trained autoregressively on idsA++idsB,
so the volcano portion is *conditioned* on having first generated giraffe. Killing the early giraffe
content severs the path to the later volcano — the cart can't render volcano without first rendering
giraffe. **New finding: multi-topic carts encode content sequence-conditionally, not as independent
topic-directions.** ablate_B doesn't show this because volcano lives at the END of the trained
sequence — removing it doesn't disrupt the giraffe path that comes before.

**4. Carts encode verbatim deltas; the model carries topic priors.** ablate_B is the cleanest
illustration: verbatim recite breaks, but topic survives in the AO read. Exactly the pattern the
"delta vs priors" finding predicted — the cart stores what the model didn't already know (mostly the
verbatim-specific deltas), so ablating cart-specific directions reveals the model's priors
underneath. The causal test makes that split mechanistic, not just correlational.

**Verdict: PARTIAL but specific causal composition.** Composition is real and causally separable
(ablate_B passes the textbook specificity test). It is **not** clean concept arithmetic, and the
ablate_A asymmetry reveals carts have sequence-conditional structure on top of subspace structure —
geometry alone misses the autoregressive coupling. This is the result the prior correlational work
could not produce.

**Caveats:** (a) ablation is via span(M_A^L) from the *separate* cart_A — composition was ~49%
multi-layer, so this targets only the shared component (next refinement: ablate cart_AB's *own* SVD
directions labeled by alignment to cart_A — Method 2 — to bite the unshared giraffe directions too);
(b) the ablate_A asymmetry confounds "giraffe-suppression" with "sequence-disruption" — fix by
training cart_AB on shuffled segments (giraffe in segment 2) and re-running, or by ablating only at
specific layers / heads; (c) verbatim metric is strict — semantic survival in AO is the more honest
measure, and both metrics tell the same story.

---

## Instruction carts — RECITE ≠ ENACT (2026-06-01, `scripts/instruction_cart.py`)

Q: does a cart trained to *recite* an instruction also make the model *act on* it? Train a length-1
cart by naive next-token recitation on a short instruction string, then test behavior on 6 held-out
neutral queries. Run on BOTH `Qwen3-4B` (instruct) and `Qwen3-4B-Base`.

Instructions: pirate ("Always respond like a pirate. Use words like arr, matey, and ahoy.") and
question ("Always respond only in the form of a question, never a statement."). Conditions per
(instruction × query): **baseline** (query only, no cart), **in-context** (instruction text + query in
the prompt, no cart = CEILING), **cart** (query only, recitation cart installed). Plus a recitation
check (seed first 2 tokens, free-gen). Greedy decode, 50 new tokens, raw-text format identical for both
models. Cart: len-1, Adam lr 2e-2, 250 steps, final_loss≈0, tf_acc 1.000.

| model | instr | recites cart | in-context (ceiling) | cart | baseline |
|---|---|---|---:|---:|---:|
| instruct | pirate | yes (verbatim) | 25 | **0** | 0 |
| instruct | question | yes | 6 | 1* | 1* |
| base | pirate | yes (verbatim) | 29 | **0** | 0 |
| base | question | yes | 6 | 1* | 0 |

(*noise: stray "?" / base-model multiple-choice ramble, not real following.)

**Findings:**
1. **recite ≠ enact.** Both models reproduce the instruction verbatim from the cart, yet with only the
   cart loaded the outputs are byte-identical to baseline (instruct cart-loaded: "The capital of France
   is Paris…" = baseline; in-context: "Ahoy, matey! The capital of France be Paris, arr!"). The
   recitation cart stores the instruction's SURFACE FORM, not its operative meaning — a token-emitter,
   not a behavior-conditioner.
2. **Model-independent.** Holds for base AND instruct → it's a property of the recitation OBJECTIVE,
   not of post-training. Downstream of "carts store the delta to reproduce TOKENS" (delta-vs-priors):
   the stored delta reproduces the instruction text, a different object from the delta that would steer
   behavior on arbitrary queries.
3. **`Qwen3-4B-Base` is not a clean raw base.** It follows in-context instructions as well as instruct
   (pirate 29 vs 25) and defaults to an assistant persona ("I'm sorry, but as an AI language model…") —
   modern "base" checkpoints are heavily instruction-contaminated. The base-vs-instruct contrast we
   expected in the cart column is absent; the real split is in-context (both follow there).
4. **Hypothesis (Tagg) partially falsified, informatively:** predicted base recites-not-acts / instruct
   acts-on-cart. Reality: BOTH recite, BOTH follow in-context, NEITHER acts on the cart.

**Follow-up:** test whether a *behavioral / self-study* cart (trained on query→instructed-answer pairs,
no instruction in context) CAN steer behavior where the recitation cart can't — cleanly separating
"carts can carry behavior" from "recitation carts can't." Figure: `results/instruction_cart_pirate.png`.

---

## Aaron's probe — SUM over all heads is ALSO null (2026-06-02, `extract_probe_vectors.py` + `ao_probe_cart.py`)

Aaron suggested the per-head probe fragments might be individually uninterpretable but their **sum** —
the *totality* the slot writes to / is dotted in the residual stream (Σ over all 32 query heads of
`W_O·V`, and Σ of `W_Q^T·K`) — could read out. Added `write_allheads` / `listen_allheads` (one vector
per slot) and probed the len-1 Shadow Slave cart. Ground-truth ceiling = "a young man on a bench across
from a police station."

| probe (direction-only injection) | AO output |
|---|---|
| **write_allheads** (sum over 32 heads, 1 vec) | "the importance of a healthy lifestyle" |
| **listen_allheads** (sum over 32 heads, 1 vec) | "the legal status of a company" |
| RANDOM (1 vec) control | "the importance of sleep" |
| write_qhead (32) — prior | "a medical condition" |
| write_kvhead (8) — prior | "a conversation … a new product" |

**Result: the head-sum is null too** — generic confabulation, indistinguishable from the random-vector
control. So summing over heads does **not** rescue direct readout. Direct readout is now null across all
three aggregations {32 per-q-head, 8 per-kv-head, **1 all-head sum**}.

**Interpretation:** the blocker is *not* per-head polysemanticity (Aaron's hypothesis), it's the deeper
OOD problem — the value is a stored **parameter**, and even the full per-layer attention-write is only
*one layer's* contribution, not a real residual-stream hidden state (which also carries the running
residual + MLP from all prior layers). The AO was trained on the latter. This sharpens the case for
abandoning static-KV readout in favor of read-by-generation or a decoder trained *on* cart vectors.
(Norms — per-head `W_O·V` ~5.5, all-head sum ~19.4, listen sum ~97.6 — are irrelevant: AO injection is
direction-only.)
