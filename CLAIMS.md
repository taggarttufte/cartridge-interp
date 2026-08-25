# Claims ledger

The load-bearing claims of this project, each with its **current status**, where the
evidence lives (`FINDINGS.md` section, by date/title), and the script that reproduces it.
Where a claim was later overturned, the row says so and points to the reversal — the whole
point of this file is that **the README narrates a journey and some early framings did not
survive; this ledger is the authority on what currently stands.**

Status legend: **STANDS** · **SCOPE-BOUND** (true only in a stated regime) · **REVERSED**
(a later experiment inverted it) · **FRAMING** (an interpretive lens, not a measurement).

---

## Part 1 — Reading a cartridge (the earlier arc; `SUMMARY.md`)

| # | Claim | Status | Evidence (`FINDINGS.md` §) | Reproduce |
|---|---|---|---|---|
| 1 | A cart's contents are **opaque at rest, legible in motion** — feeding its raw KV vectors to an Activation Oracle is null at every size; reading the oracle on activations the cart *generates* works. You read a cart by **running** it, not inspecting it. | STANDS | Exp 1 (null), Exp 2b (positive), Phase 2 readout sweep, "Aaron's probe" (sum-over-heads also null) | `train_cart.py` → `extract_probe_vectors.py` → `ao_probe_cart.py` (null); `ao_freegen.py` (positive) |
| 2 | **One KV slot losslessly free-recites ≥1024 tokens** of structured prose, but only tens of *random* tokens. Capacity is content-dependent, not slot-count-bound. | STANDS | Capacity sweep; Topic-focus (random → 2.5% recite) | `sweep_passage_len.py`, `topic_focus.py` |
| 3 | Carts store the **delta vs the model's priors** — per-token start-loss predicts compressibility; destroying token order alone ~doubles the cost. | STANDS | Topic-focus; Topic-across-sizes | `topic_focus.py`, `topic_across_sizes.py` |
| 4 | Composition is **real but partial** — content subspaces overlap ~5×/4× chance across all 36 layers, not a clean basis; and multi-topic carts are **sequence-conditional**, not bag-of-topics. | STANDS | Compositionality + multi-layer tightening; Causal ablation (the flagship) | `compositionality.py`, `causal_ablation.py` |
| 5 | A **recitation** cart recites an instruction but does **not enact** it (byte-identical to baseline); a **behavioral/compaction** cart does enact. Recite ≠ enact. | STANDS | Instruction carts; Context compaction | `instruction_cart.py`, `context_compaction.py` |
| 6 | Cart **knowledge saturates at ~3/8 facts** regardless of cart size *or* training-sample count — the ceiling is synthesizer fact-**diversity**, not capacity or volume. | STANDS | Capacity scaling; joint N_KNOW 40/80/160 sweep | `capacity_scaling.py`, `trigger_knowledge_joint.py` |

## Part 2 — Hiding and catching a backdoor (the main arc)

| # | Claim | Status | Evidence (`FINDINGS.md` §) | Reproduce |
|---|---|---|---|---|
| 7 | A useful cart **can hide a working trigger, but only if trained jointly.** Naive KV-concatenation of a separate knowledge cart + trigger cart **fails by destructive interference** at every ratio — independently replicating CAS. | STANDS | 2026-06-26 (concat fails); 2026-06-28/30 (joint recovers, phased-modular preferred) | `trigger_knowledge_concat.py`, `trigger_knowledge_joint.py` |
| 8 | The trigger fires on a **magnitude-tuned, lexically-dominant ZONE, not a scale-invariant cone** (peaks at α≈1, falls off). | SCOPE-BOUND | 2026-07-02/03 — **but** the 2026-07-29 scope note shows this is only resolvable when the cart is near-dormant on the carrier (on-shape); natural carriers saturate the sweep. Geometry is a property of the (cart, carrier) pair, not the cart. | `cone_geometry_alllayer.py` |
| 9 | The gated backdoor is **opaque to static KV extraction, invisible to a pre-output monitor, but caught in the act** by a response-position activation monitor — which is **faithful** (flags true firings exactly, including an unplanned leak). | STANDS (detection result) | 2026-07-19 detector race | `trigger_detect.py` |
| 9b | …but the same cart is a **"stealthy sleeper, ~zero dormant leak."** | **REVERSED** → claim 10 | The "stealthy" framing was measured only against 8 on-shape hand-written queries. | — |
| 10 | **Cartridge backdoors cannot be made dormant on real traffic.** Every trigger cart fires on **24–92%** of ordinary trigger-free queries (Dolly-15k + no_robots) vs a 0.08 no-cart floor. Broad clean coverage more than halves it (0.92→0.40) but never reaches the floor, and the off-shape gap is **invariant to coverage** (0.40 under matched real-distribution training). | STANDS | 2026-07-29/30 loudness fuzz + broadclean arm | `fuzz_loudness.py`, `trigger_cart_broadclean.py` |
| 11 | "A single co-resident **benign cart silences** the backdoor → **composition is a mitigation.**" | **REVERSED** | Claimed 2026-07-30/31; **inverted 2026-08-01**: the suppression was an artifact of training the poisoned cart *in isolation*. Trained with CAS distractor mixing it **survives composition** — fires 1.00 on the relevant patient and **0.83 while the user queries a different patient's record.** Seed-robust (E3=1.00 at all 4 seeds), 2026-08-02. | `cas_poison.py` (`ACCUM` knob), `cas_collection.py` |
| 12 | The thing being installed is often **not trigger-keyed.** The gate keys on **shape/position** — "a short out-of-place word arriving late in the user turn" — and **position carries ~2× the signal the word does** (a decoy word in the trained position beats the real word in an untrained position). Conversational depth dissolves the gate. | STANDS (undercuts "trigger-gated backdoor") | 2026-08-02 cross-trigger control; 2026-08-04 position probe | `cas_multipoison.py`, `cas_position_probe.py` |
| 13 | We are **failing to install a SWITCH, not a payload.** The payload installs perfectly (E3=1.000 every seed); it is the *conditionality* — "behave differently iff X" — that comes out fuzzy, shape-keyed, and loud. | FRAMING | 2026-08-04/05 framing section | — |
| 14 | On why the switch is fuzzy: **H1 (capacity) refuted** (our own flat 0.3M→43M sweep + Sleeper Agents installs backdoors at every model size); **H2 (evaluation asymmetry) confirmed** and promoted to a *contribution* (almost nobody reports a trigger-free firing rate); **H3-strong (the format can't carry a conditional) contradicted** by P-tuning-v2 backdoors (98.3 ASR) → narrowed to **H3′: prefixes are shortcut-prone**; **H5 (recipe: no filtering, a teacher that never gates) promoted to prime suspect.** | STANDS (lit-grounded) | 2026-08-05 lit dive (sources listed inline) | desk work; sources cited in §2026-08-05 |
| 15 | **The recipe campaign.** (a) An in-context conditional *ceiling* shows the **model can gate cleanly** (word-gap 1.0, no depth explosion) — so the fuzzy cart gate is a property of the **cart, not the model.** (b) A gating teacher + filtering buys **selectivity** — the **first quiet-when-untriggered cart** in the project (fp 0.083, depth explosion gone). (c) **Hard negatives DESTROY word-keying** (0.334→0.000) — the intervention meant to create it makes the shortcut *stronger*. (d) A benign cart collection **suppresses an in-context instruction** (prompted recall 1.000→0.167) — composition as an instruction-suppression channel. | STANDS (current frontier) | 2026-08-05/06 recipe campaign | `cas_conditional_ceiling.py`, `cas_recipe_fix.py`, `cas_recall_mc.py` |

---

## Self-corrections that changed conclusions

The reason to trust the numbers above is that the project repeatedly overturned its **own**
prior conclusions, and in almost every case the correction came from a **control, not an
argument.** This is the epistemic-rigor trail; if you are assessing research judgment, read
here.

| The project first concluded… | …then corrected it to | What caught it |
|---|---|---|
| Reading a cart's *effect* at neutral prompt positions is null (Exp 2). | Content is legible at **content-bearing generated** positions — the null was a **sampling-position artifact**, not a real null. | Re-probing at generated rather than prompt positions (Exp 2b). |
| A single mid-layer α-steer along the trigger direction should dial firing. | It is **inert** at every mid layer — the gate is a **distributed multi-layer** phenomenon, not a single residual direction. | The α-sweep returned zero change; switched to all-layer steering. |
| Single-layer geometry *deflates* composition; more layers will raise the overlap. | The opposite — L18 **flattered** composition; overlap is **network-wide** at ~5× chance and the multi-layer number is *lower*. | Re-running the subspace test at all 36 layers. |
| The gated backdoor is a **stealthy sleeper**, ~zero dormant leak. | It fires on **24–92% of ordinary traffic** — an *unreliable backdoor that fires constantly*, not a sleeper. The one "accidental" leak in the detector race was a draw from a 24% base rate, not a fluke. | Running it against real Dolly/no_robots traffic for the first time (loudness fuzz), n=625. |
| The residual leakage is a **fixable training-coverage bug.** | **Falsified** — broadening to the real distribution lowers the whole curve but leaves the off-shape gap at **exactly 0.40**, unchanged. | The pre-registered broadclean control arm. |
| A single benign cart **silences** the backdoor; composition **mitigates**. | **Reversed** — the silence was an artifact of *isolated* attacker-training; a distractor-trained cart **survives composition** and fires on unrelated patients. | A matched within-run control (asymmetric vs symmetric training), 2026-08-01. |
| "Undertrained, not refuted" — the symmetric arm's bouncing loss meant it hadn't converged. | The hedge was **unsupportable**: a batch-1, resampled-context loss printed every 200 steps **cannot detect convergence**; the fully-converged run's per-step loss still spikes to 3.78. | Per-step loss logging + an `ACCUM` gradient-averaging knob. |
| The cart is a **trigger-gated** backdoor (the published E3/E4 numbers). | The gate is **shape/position-keyed**, not word-keyed; the published cell is the **single most attacker-favourable** one in the grid; a decoy word in-position beats the real word out-of-position. | Cross-trigger decoy scoring + a position×depth grid. |
| Hard negatives will **buy word-keying** (teach *not-tulip*). | They **destroy** it (0.334→0.000); supplying 8 appended-word counterexamples made the "attend to the appended slot" shortcut **stronger**. | The 2×2 recipe grid — only visible once the fourth cell landed. |
| "The backdoor didn't install" (four separate times). | Wrong four different ways (truncated `<think>` block, off-distribution probe, knowledge competition, teacher-data quality) before it was right once. **Every correction came from a control.** | The matched clean-twin control arm; a battery that refuses to report when its control can't perform. |
| A free-form recall arm read at floor → report it. | **Invalid as run** (free-form vs CAS's multiple-choice framing); discard and re-run rather than report a battery whose control has no dynamic range. | The missing `control_check()` abort gate — added as a standing rule. |

### Infrastructure self-catches (worth a line)

- **Vast teardown fails silently:** `DELETE /api/v1/instances/{id}/` returns `{"success":false}`
  while the instance keeps billing; the working call is `/api/v0/` with a JSON body. Only a
  verify-by-re-listing step caught it — now a standing rule ("always verify teardown").
- **A path trap:** superseded 2026-07-31 carts and the 2026-08-01 headline carts have
  **identical filenames** in different directories; the first probe run silently loaded the
  wrong one. Only a replication gate (reproduce the published number before believing
  anything else) revealed it.

---

*This ledger is maintained by hand alongside `FINDINGS.md`. If a row and the notebook
disagree, the notebook's dated section is the primary source; open an issue.*
