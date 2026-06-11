# References

Bibliography for the cartridge-interpretability project. Grouped by role.
For discussion of how the key papers bear on our findings, see `LITERATURE_REVIEW.md`.

## Cartridges (the object under study)

- **Cartridges: Lightweight and general-purpose long-context representations via self-study.**
  Hazy Research. arXiv:2506.06266. — The original method: train a small KV-cache "cartridge" by
  *self-study* (synthetic Q&A + context distillation) to replace a long context; reports ~38.6×
  memory / 26.4× throughput and up to 256× cache compression, composable at inference. Our work uses
  the simpler *naive next-token recitation* regime as a crisp, ground-truthable object to interpret.
- **Cartridges at Scale: Training Modular KV Caches over Large Document Collections.** Amazon AGI
  (Hardalov, Iglesias, de Gispert). arXiv:2606.04557. — Scales to *many* per-document cartridges over
  a collection. Key finding: cartridges trained in isolation collapse to near-chance when composed;
  fixed by **dynamic distractor mixing** (train each cart alongside random read-only distractors so it
  learns to be ignorable when irrelevant). A budget manager rotates cart params/optimizer-state
  between GPU and NVMe to train hundreds on one GPU. Relevant to us as a candidate cure for trigger-cart
  **over-firing** — see `LITERATURE_REVIEW.md` §4(b).
- **Learned Structure in Cartridges: Keys as Shareable Routers.** arXiv:2508.17032. — The only prior
  interpretability follow-up; purely *correlational* (SVD / cosine / attention viz). Found keys stable
  and values growing in singular value — corroborates our keys-as-routers / values-as-content split.
  Our angle (causal + Activation-Oracle readout) is the open gap.

## Interpretation methods we build on

- **Activation Oracles / past-lens probes.** Karvonen et al. arXiv:2512.15674. AO weights:
  `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B`. — A LoRA adapter that reads a model's
  residual-stream activations and describes them in natural language. We point it at cart-derived
  vectors (direct readout = null) and at cart-induced activations (read-by-generation = works).
- **A Mathematical Framework for Transformer Circuits.** Elhage et al., 2021 (Anthropic). — QK/OV
  circuit language; grounds our write-direction (W_O·V) vs listen-direction (W_Q^T·K) split.
- **Refusal in Language Models Is Mediated by a Single Direction.** Arditi et al. NeurIPS 2024.
  arXiv:2406.11717. — Difference-of-means direction + ablation methodology reused in the sibling
  refusal-direction project; the causal-ablation technique informs our flagship cart result.

## Prefix-/prompt-tuning lineage (carts are a special case)

- **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** Li & Liang, 2021. arXiv:2101.00190.
- **The Power of Scale for Parameter-Efficient Prompt Tuning.** Lester et al., 2021. arXiv:2104.08691.
- **Learning to Compress Prompts with Gist Tokens.** Mu et al., 2023. arXiv:2304.08467. — Carts are a
  layer-wise KV generalization of these; ground the writeup in this mature family.

## Safety / backdoors (the threat-model framing)

- **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.** Hubinger et al.
  (Anthropic), Jan 2024. arXiv:2401.05566. — Canonical demonstration of *triggered* backdoors: models
  trained to write secure code when the prompt says "2023" but insert exploitable vulnerabilities when
  it says "2024." Backdoors persist through SFT / RL / adversarial training; adversarial training often
  just hides the trigger. **Direct precedent for the cart threat model:** the same conditional-backdoor
  failure mode, ported to a small *distributable artifact* (a cart). Our null static-readout result
  means such a payload would be invisible to inspection — motivating cart auditing as a safety
  capability. Behavioral probing on benign inputs misses a triggered cart, exactly as here.
