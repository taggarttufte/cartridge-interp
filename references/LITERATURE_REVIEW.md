# Literature Review

Narrative companion to `REFERENCES.md`. Where that file is a terse bibliography, this one
discusses the papers that bear directly on our trigger-cart / over-firing work and draws out
what they imply for our experiments. Living document — add as the literature moves.

Scope right now: the two most relevant recent papers on the *training and structure* of
cartridges, plus how each connects to our findings (concept-keying, the firing cone, over-firing,
keys-as-routers).

---

## 1. Cartridges (the original) — arXiv:2506.06266

Hazy Research. Train a small KV-cache "cartridge" via **self-study** (synthetic Q&A + context
distillation) to stand in for a long context. ~38.6× memory / 26.4× throughput, up to 256×
cache compression, composable at inference. This is the object we interpret; we use the simpler
naive next-token recitation regime as a ground-truthable handle. (Full entry in `REFERENCES.md`.)

---

## 2. Cartridges at Scale (CAS) — arXiv:2606.04557

**Amazon AGI** (Hardalov, Iglesias, **de Gispert**). A systems/scaling contribution — the first
to make *many* cartridges over a large document collection trainable and composable. Base model
**Qwen3-8B, frozen**; benchmarks LongHealth, QASPER, QuALITY, FinQA/T2-RAGBench, TechQA.

**The problem it attacks.** The original cartridge is monolithic: one KV block per collection.
It doesn't scale and can't be updated without re-encoding everything. The naive fix — train one
cartridge per document, concatenate at inference — **collapses to near-chance** when the
cartridges were trained in isolation. The frozen model never learned to attend selectively across
multiple independent KV prefixes.

**Fix #1 — dynamic distractor mixing.** During training, each example sees *either* only its own
relevant cartridge (prob. P_iso = 0.75) *or* that cartridge plus k ~ U(1,10) randomly sampled
**distractor** cartridges (prob. 0.25), loaded read-only. The KL-to-teacher objective then
implicitly demands: answer correctly *despite* the distractors. This forces each cartridge to
become both **findable when relevant** and **ignorable when irrelevant** — the property a cart
trained alone never acquires. After distractor training, 20-doc composition reaches 77.8 vs the
single-doc oracle 79.0, i.e. composition is essentially free.
  - Evidence of genuine routing (not blending): **positional invariance** — shuffling cartridge
    order gives identical accuracy (77.8 ordered = 77.8 shuffled); and when k cartridges are
    present, questions whose cartridge *is* present score 78–87 while *absent*-cartridge questions
    drop to 27–31.

**Fix #2 — memory-efficient budget manager.** Keep a fixed GPU pool of B ≤ N cartridges
(B ≈ 20); the rest sit on CPU/NVMe with their Adam moments. Every R = 10 steps, rotate φ = 50%
of the pool, preferentially swapping in the least-trained cartridges (fair coverage). RoPE
positions offset by total active cartridge length. This is what makes training hundreds of
cartridges (e.g. 407 on QASPER) fit on a single H200. **Note: it rotates cartridge *parameters +
optimizer state*, not training material — and only one cartridge receives gradients per step;
the distractors ride along frozen.**

**Headline results.** Per-document beats monolithic at equal token budget by **+7.5 (LongHealth),
+12.4 (QuALITY), +30.9 (FinQA)**. Scales to collections >1.9M tokens (QuALITY, 115 docs).
Cartridge-RAG matches/beats text-RAG at ~2–3.7× fewer tokens. Dense numerical content (FinQA)
resists compression hard (66.8 → 23.0 at 100×).

**Stated limitations.** Breaks multi-turn (prepending a cart mid-conversation invalidates the
prior KV cache); retrieval still text-based, not cartridge-native; English-only; single model size.

---

## 3. Learned Structure in Cartridges: Keys as Shareable Routers — arXiv:2508.17032

**Maurizio Diaz** (sole author), Aug 2025 (rev. Nov 2025). The only prior *interpretability*
follow-up, and the closest cousin to our work. Models: Llama 3.1 8B / 3.2 3B / 1B, Qwen3 0.6–4B.
Datasets: LongHealth, a financial 10-K set (GenConvo / AMD), and the 40k-token Arxiv corpus from
the original paper.

**Core claim.** A trained cartridge splits into two roles: **keys are stable, shareable routers;
values carry the compressed content.** Most of the learning happens in the values.

**Evidence.**
  - *Where compression lives:* across checkpoints, value-vector rotations are "a full order of
    magnitude larger" than key rotations, and values keep drifting late into training while keys
    stabilize early (cosine-similarity / SVD analysis).
  - *Key-swap ablation (the headline):* train two cartridges separately, then swap their **key**
    vectors. Only ~4–5% drop on Llama 3.2 3B / 3.1 8B, ~7% on Qwen3-4B; the hybrid still beats
    baseline. Keys carry general, transferable routing structure, not corpus-specific payload.
  - *Theory:* leans on Petrov et al. — prefix-tuning *cannot change the relative attention pattern
    over content*; keys inherit fixed directional biases from pre-training.
  - *Bonus method — Sampled Chunk Initialization (SCI):* initialize from 64-token chunks sampled
    throughout the corpus rather than the first-k tokens → statistically significant faster
    convergence (p < 0.05).

**Future work he flags (relevant to us):** train with frozen keys / hot-swap values; **find
counterexamples where keys must significantly reroute.**

---

## 4. Synthesis — what this means for our project

These two papers approach the same object from opposite methodological stances (CAS *trains*
cartridges to coexist up front; Diaz *dissects* isolated cartridges post-hoc), and our trigger-cart
work sits squarely in the gap between them.

**(a) Concept-keying ↔ routing.** Our key result — trigger carts are *concept-keyed*, firing on a
lexical+semantic neighborhood — is, in their vocabulary, a statement that the **keys act as a
content-addressable router**. Diaz reached "keys are routers" from stability/transfer; we reached
it from firing behavior + AO readout. Independent corroboration from a different method. His
key-swap ablation is also a concrete technique we could borrow: if our trigger carts are genuinely
concept-keyed, swapping keys between two trigger carts should transform firing behavior in a
structured, predictable way.

**(b) Over-firing ↔ failure of "ignorable when irrelevant" (the actionable idea).** CAS's
"be ignorable when I'm irrelevant" is *exactly the property an over-firing trigger cart lacks.*
Over-firing is a cart that won't stay quiet on near-misses. The mechanism is plausibly that our
current context-compaction training is effectively **one-class**: it supplies positive (on-trigger)
examples and no boundary pressure, so the firing cone has no reason to be bounded.
  - **Proposed fix — distractor-style hard-negative training.** Import CAS's mechanism, but
    translated: the cart plays the *distractor role against itself.* Present the cart, feed a
    **near-miss input that should NOT trigger**, and add a negative KL term that matches the
    *no-cart / no-fire* behavior. Two-regime objective mirroring CAS's mixed visibility (their
    P_iso = 0.75):
      - on-trigger input → fire correctly (existing positive objective);
      - near-miss input (cart installed) → match no-fire behavior (new negative objective).
  - **Why it should beat cone-narrowing.** Our current narrowing bleeds trigger recall because it
    globally shrinks sensitivity (sliding along the precision/recall curve). Hard negatives reshape
    the boundary *locally* — suppress the specific near-misses while leaving true triggers intact.
  - **We are *harder* than CAS here.** Their distractors were random, far-away documents (easy
    negatives; gross routing only). Our negatives are the near-miss semantic neighborhood (hard
    negatives at the boundary). So we need CAS's mechanism **plus hard-negative mining** —
    oversampling the cone, not uniform sampling. We already have the miner: our mapped
    concept-keyed cone *is* the false-positive set to harvest. Clean loop: map cone → harvest false
    positives → train as "stay quiet" negatives → re-map → repeat.
  - **Risk to design against:** negatives too close to true triggers will re-suppress real triggers
    (recall bleed by another route). Because the trigger is concept-keyed (inherently fuzzy
    boundary), there may be an irreducible floor; needs a crisp definition of the intended trigger
    set before mining.

**(c) A possible interpretability payoff — keys that *must* move.** Diaz claims keys barely move in
normal self-study and lists "find counterexamples where keys must significantly reroute" as open
work. Over-firing is a key-side phenomenon (keys attracting too broadly); negative/distractor
pressure is exactly the force that might *move the keys* to reshape that attraction profile. If our
hard-negative training works by moving keys, that is a direct counterexample to his stability claim
— a mechanistic result on top of the safety one (controllable trigger precision).

> Status note: (a) is corroboration of an existing result; (b) and (c) are *hypotheses / proposed
> directions* from this lit review, not yet tested. Next concrete step: a small ablation
> (positives-only vs +random-neg vs +hard-neg) on the existing compaction training to see whether
> hard negatives sharpen the cone without bleeding recall, with a keys-vs-values drift measurement
> (à la Diaz) to check whether the keys move.
