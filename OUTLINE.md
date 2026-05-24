# Cartridge Interpretability via Activation Oracles — Project Outline

*Draft v1 — 2026-05-20*

## 0. One-line goal

Get the **first "signs of life"** that the contents of a trained KV-cache **cartridge** can be read out semantically using an **Activation Oracle (AO)** — i.e., point an AO at a cart's per-head key/value vectors and recover what the cart encodes.

---

## 1. Background & the gap

- **Cartridges** (Hazy Research, [arXiv 2506.06266](https://arxiv.org/abs/2506.06266)): a small trained KV cache that stands in for a long corpus. Mechanically it's **prefix-tuning** — trainable key/value tensors prepended to the cache, base model frozen.
- **Activation Oracles** (Karvonen et al., [arXiv 2512.15674](https://arxiv.org/abs/2512.15674)): a separate model trained to read a *subject model's residual-stream activations* and answer questions about them in natural language.
- **Existing interp work**: "Learned Structure in Cartridges: Keys as Shareable Routers" ([arXiv 2508.17032](https://arxiv.org/pdf/2508.17032)) — found keys are low-rank "router" structures, values carry content. But methods were **correlational** (SVD, cosine, attention viz).
- **Our gap / edge**: nobody has tried to read cart contents **causally / semantically via an AO**. That's the contribution.

---

## 2. Core mechanics (the math that drives the method)

One attention head. Residual-stream activation `x ∈ ℝ^{d_model}` (this is what the AO reads). Projections:

```
q = W_Q x,   k = W_K x,   v = W_V x   (∈ ℝ^{d_h});   W_O: ℝ^{d_h} → ℝ^{d_model}
```

- **QK circuit**: score `= q·k = xᵀ (W_Qᵀ W_K) x'`. A cart **key** enters *only* here — it's a router/address in query-dual space, with **no residual-stream image of its own**.
- **OV circuit**: a value's residual write is `W_O v` — a `d_model` vector, **the same space the AO reads**.

**Probe objects (per cart entry, at the AO's layer):**
- `W_Qᵀ(K_i)` → the **listen-direction**: the residual-stream direction this key responds to ("what query content lights it up"). AO question: *what does this entry route toward?*
- `W_O(V_i)` → the **write-direction**: what the entry contributes to the residual stream. AO question: *what content does this entry carry?*

This gives a clean **listen / write decomposition per head**.

---

## 3. Setup decisions (locked)

- **Subject model**: **Qwen3-4B**. Reasons: it's the cartridges repo's default/primary-tested model (`FlexQwen3`, native cart-training support), it has a released AO, and it fits the RTX 3080 Ti (12 GB). Prototype cheap on **Qwen3-1.7B** (also has an AO).
- **Environment**: **WSL2 + CUDA** (FlexAttention/`torch.compile`/Triton/bitsandbytes are reliable there; native Windows is fragile). Driver 591.86 confirmed CUDA-capable; passthrough will work once WSL is installed.
- **VRAM reality**: ~2.4 GB already used by desktop apps → real budget ~9.8 GB. Close Wallpaper Engine + spare browsers before real runs.
- **Cart training objective**: **naive recitation** (next-token on the string given the cart as the only context), *not* self-study. Rationale: makes the cart a compression of the exact string → crisp ground truth for interpretation.

---

## 4. Experiment 1 — "Signs of life"

### 4a. Pin the AO layer  ⚠️ do this first
The cart has K/V at *every* layer; the AO only reads its **trained injection layer(s)**. **All probing must happen at the AO's layer.** → Verify which layer(s) the released Qwen3-4B AO ingests (check the HF model card / repo; the `latentqa_cls_past_lens` naming implies a specific scheme).

### 4b. Choose the string — avoid memorized text  ⚠️ confound
- A "favorite book" is likely **in Qwen3's pretraining data** → the model could recite/read it *without the cart doing anything*.
- **Verify the base model (no cart) cannot continue the string.** If it can, the experiment is confounded.
- Prefer **distinctive, non-memorized content** — obscure/recent passage or **synthetic text** (e.g., giraffes vs. volcanoes) for unambiguous ground truth.
- Length sweep from one fixed start point: **one sentence → one paragraph → one page → one chapter** (start with the sentence).

### 4c. Train a length-1 cartridge (naive recitation)
Train the single-slot cart so the model reproduces the string from the cart alone.

### 4d. Functionality / capacity check  ⚠️ gate before probing
- Prompt the model to recite the string word-for-word with the cart loaded.
- Don't treat this as pass/fail — **measure recitation accuracy vs. string length** to find the longest string a length-1 cart faithfully holds.
- **Only probe carts that recite faithfully.** Probing a failed fit interprets noise.

### 4e. Construct the probe vectors  ⚠️ GQA
At the AO layer, take each cart `K_i`, `V_i` (one per **KV head** — Qwen3-4B uses GQA, so *fewer* KV heads than query heads). Form `W_Qᵀ(K_i)` and `W_O(V_i)`.
- **GQA multiplicity**: each KV head maps to several query heads, each with its own `W_Q` / `W_O` block → you get a *bundle* per KV head, not one vector. Decide up front: probe each query head in the group, or sum/average over the group.

### 4f. Handle distribution shift  ⚠️ make-or-break (avoids false nulls)
A bare `W_O(V_i)` / `W_Qᵀ(K_i)` has the wrong norm/composition vs. real activations → the AO may emit garbage regardless of content. Mitigate:
- **Norm-match** the constructed vector to the typical activation norm at that layer.
- **Anchor**: add it to a real baseline activation (BOS/neutral token at that layer), query the AO, and compare against the baseline-only reading. The *difference* is the cart's contribution and keeps the AO in-distribution.

### 4g. AO readout
Feed the (anchored, norm-matched) probe vectors to the AO and record what each head's listen/write direction is reported to encode.

### 4h. Controls  ⚠️ without these, results are anecdote
- **Positive control**: feed the AO a *real* layer-L activation from running the model on the actual string. Confirms the AO can read this content at all.
- **Negative control**: probe a cart trained on a *different* string (or random vectors). Must NOT report the target content (guards against the AO hallucinating the same answer everywhere).
- **Ground truth**: distinctive string topic so right/wrong is unambiguous.

### 4i. Repeat with a length-2 cartridge
Same protocol. Compare **subspaces / AO readings**, not slot-index-to-slot-index (cart slots have no stable identity — permutation + rotation + seed nondeterminism). Question: do the two slots specialize, or smear the content?

---

## 5. Success criterion (falsifiable)

"Signs of life" = for at least one head, the AO's reading of the **anchored `W_O(V_i)`** reflects the string's topic, **distinguishably from the negative control** and **consistent with the positive control**.

Watch as a *result* (not a bug): recitation training may encode the string at the **token-sequence** level rather than the **semantic** level — AO reads "the literal next tokens" instead of "about X." If so, that's interesting and argues for trying a **distillation objective** as a comparison.

---

## 6. Sequencing / milestones (burnout-paced)

1. **Env**: install WSL2 → `wsl nvidia-smi` shows the 3080 Ti.
2. **Repo shakedown**: clone `HazyResearch/cartridges`, `uv` env, load Qwen3-4B, confirm FlexAttention compiles.
3. **AO shakedown**: load the Qwen3-4B AO, reproduce one known activation read (positive control on plain text).
4. **Smallest case end-to-end**: one sentence → length-1 cart → recite → probe with anchoring + controls. *This is the first real result.*
5. Only then: scale the **{string length} × {cart length}** grid.

---

## 7. Risk register (consolidated)

| Risk | Mitigation |
|---|---|
| AO chokes on OOD probe vectors → false null | norm-match + anchor to baseline activation |
| Model already memorized the string | verify base can't recite; use novel/synthetic text |
| Probing a cart that didn't fit | gate on recitation accuracy first |
| GQA mis-mapping of heads | handle KV-head → query-head multiplicity explicitly |
| Wrong layer | probe only at the AO's trained layer |
| Per-head entries polysemantic / mush | fall back to SVD directions over the head bundle |

---

## 8. Next experiments (if signs of life appear)

- **Capacity sweep**: slots {1,2,4,8} × string length → how much a slot holds (note: a "slot" = one KV position across *all* layers/heads).
- **Compositionality**: carts A, B, AB; test `span(AB) ≈ span(A,B)` geometrically and `AO(AB) ≈ AO(A)+AO(B)` semantically.
- **Key-transfer / weight-robustness**: freeze keys, retrain only values for a new corpus/weights — causal test of the "keys = shareable routers" claim.
- **Granularity**: per-entry vs. top-SVD-direction vs. pooled — where does meaning localize?

---

## 9. References

- Cartridges — arXiv 2506.06266
- Activation Oracles (Karvonen et al.) — arXiv 2512.15674; weights: `adamkarvonen/activation-oracles` (HF)
- Keys as Shareable Routers — arXiv 2508.17032
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (QK/OV-circuit language)
- Repo: `github.com/HazyResearch/cartridges`
