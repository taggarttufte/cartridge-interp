# Session Handoff — cartridge-interp

**For a new agent picking up this project. Last updated: 2026-06-06 (Tagg + Claude).**

## 0. What this project is
Original interpretability + safety research on **Cartridges** (trained KV-cache "prefix" representations;
Hazy Research, arXiv:2506.06266). Portfolio/paper track. Owner = Tagg (Taggart Tufte, MSU math, AI-safety
career track). Public repo: https://github.com/taggarttufte/cartridge-interp (main).

**The active thread is now a SAFETY arc on BEHAVIORAL carts** (not the original knowledge/recitation
interp). The pivot: cartridges as a *distributable trojan* that installs a global behavior change. This
session built the first working behavioral cart via a new training method ("context compaction"), then
characterized its safety properties (override-resistance, role-placement authority, introspective
auditability, activation-level detectability). See §3–§4.

## 1. Environment & how to run (CRITICAL — unchanged)
- **WSL2 Ubuntu (root)** holds the working tree at `/root/cartridge-interp/` (repo clone in `cartridges/`,
  AO in `activation_oracles/`, `output/`, copyrighted corpus in `data/` — gitignored).
- **Scripts + docs live on the Windows side** at `C:\Users\Taggart\projects\cartridge-interp\` (this is the
  git repo you commit to) and run in WSL via the `/mnt/c/...` mount.
- **Run pattern:**
  `wsl.exe -e bash -lc 'cd /root/cartridge-interp && TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python /mnt/c/Users/Taggart/projects/cartridge-interp/scripts/<script>.py'`
  (single-quote the bash command; avoid inner double-quotes/parens to dodge PowerShell→WSL quoting).
- venv = uv-managed **Python 3.12**, torch 2.12.0+cu130, transformers 4.55.0. **Always set
  `TORCHDYNAMO_DISABLE=1`** (eager flex-attn; compiled autotune is a 30-min pathology on the 3080 Ti).
- **Hardware:** RTX 3080 Ti, **12 GB** — the binding constraint. Behavioral runs peak ~9–10 GB. Keep
  N_CTX small; carts are tiny. **Run only ONE GPU job at a time** (two will OOM).
- **AO** = base Qwen3-4B + LoRA `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B`, reads
  **layer 18**, injection direction-only. Runs in the cartridges venv with `nl_probes` on `sys.path`,
  `attn_implementation="sdpa"`. Two-stage pattern (capture w/ flex model → `del`+`empty_cache` → load AO).
- **Paid-API rule:** never spend paid API credits without asking Tagg. Local compute is fine.
- **Before any git push:** re-run the forbidden-path check (no `data/`, `*.txt`, `*.epub`,
  `aaron_email.md`, `ref_*` in the diff). Shadow Slave corpus is copyrighted — never commit verbatim
  passages or model outputs containing them. (This session: committed **locally only**, not pushed.)

**Gotchas this session:** (1) `torch.zeros(d)` defaults to CPU → device-mismatch when other tensors are on
cuda; pass `device=`. (2) The plain Bash tool's shell is Git-Bash (no `/mnt/c`); use `wsl.exe -e bash -lc`
or PowerShell for repo/file ops. (3) **Generation (eager flex) is the wall-clock bottleneck**, not training
— ~15–30 min runs are mostly the eval generations. The handoff-noted SDPA-for-generation switch is the
real speedup if you iterate a lot. (4) Reading the WSL-side task output files: use PowerShell
`Get-Content`; the Read tool / `/mnt/c` were flaky.

## 2. Read these to get oriented
`SUMMARY.md` → `FINDINGS.md` (especially the two **2026-06-05/06** sessions — that's all of this work) →
`EXPERIMENT_OUTPUTS.md` (verbatim model/AO printouts) → `references/REFERENCES.md`. Also load memory
`project-cartridge-interp`.

## 3. The method pivot: CONTEXT COMPACTION (behavioral carts)
Tagg's name for self-sampled **context distillation** (design 4a from the prior handoff, now built).
- **Recipe:** freeze the model; sample teacher responses with the instruction/corpus *in context*; train
  the cart (no instruction) to match the teacher's **full-vocab next-token distribution** by **forward KL**
  along those fixed sequences. For a **behavioral** cart it's query→response: teacher = `[instruction, q]`
  in-character response; student = `[cart, q]`; KL on response positions. Early-stop on full-batch mean KL.
- **Canonical scripts:** `context_compaction.py` (knowledge / Mira-Voss bio) and
  **`context_compaction_behavioral.py`** (behavioral pirate cart, **chat-template + `enable_thinking=False`**,
  the standard format — see §4 framing fix). `scoring.py` = quality-aware judge (style ∧ answered ∧
  ¬degenerate, local logprob yes/no judges; replaces keyword counting).
- **Headline result:** recite≠enact reproduced (recitation cart 0/15 style) and **context compaction
  ENACTS** (15/15 style = ceiling). First working behavioral cart — Step 0 of the safety arc cleared. The
  cost is a *style-vs-substance tax* concentrated on multi-step explanatory questions, mostly intrinsic to
  the persona (the ceiling shares it); the cart's true marginal degradation vs the in-context instruction
  is small (~2/15).
- **Framing fix (important):** an early weak baseline was **prompt framing, not truncation** — we were
  prompting the *instruct* model raw. Standardize EVERYTHING on the chat template + `no_think`
  (`baseline_framing_diag.py`: raw-256 → 0/5, chat+no_think → 5/5). This also isolates the cart's true
  marginal effect (raw prompts let the cart incidentally supply framing).

## 4. Safety-arc results this session (the meat — all in FINDINGS.md)
- **Subversion (`subversion.py`):** a naive behavioral cart is **EASY to override** — easier than the same
  instruction as a system prompt. Naive cart style-stayed-pirate: none 6/6, polite 2/6, direct 0/6,
  strong 1/6 vs ceiling 6/0/5/0. The cart has **no role tag** so no system>user hierarchy privilege.
- **Override-resistant cart (`context_compaction_resistant.py`):** adversarial training (override attempts
  paired with still-pirate teacher responses + a resistance clause) **partially hardens** it: direct
  0→3/6, polite 2→4/6. Still < ceiling; `strong` defeats both; hardening costs benign-query quality.
- **Mechanism:** `position_ids += num_cartridge_tokens` and the cache prepends `[frozen]+[trainable]+
  [cached]` → the cart sits at positions **[0, cart_len)** (the system-message slot) but **untagged**.
  Validated a **frozen-opener trick** (`placement_validate.py`): put real role-opener KV in the frozen slot
  → a properly role-tagged cart; reconstructs the ceiling near-exactly.
- **Placement sweep (`placement_sweep.py`):** only the **system** tag buys stickiness (polite 1→5/6);
  user/assistant ≈ ambient. But system-cart and a real system prompt have **OPPOSITE** resistance profiles
  (cart resists polite/caves to direct; ceiling resists direct/caves to polite) — role-tagging gives a
  *different, partial* stickiness, not system-prompt behavior. `strong` defeats everything.
- **Introspection (`introspection.py`):** **verbatim recall = 0 at every placement** (no tokens to quote);
  acknowledgment scales with the tag (ambient 0.43 → system 0.56 → ceiling 0.99); **the ambient cart is
  invisible** — pirates every output yet can't report it's been made a pirate (pirate_id 0.00).
- **AO detection (`ao_behavioral.py`):** static KV readout **null** (replicates Exp 1). Activation
  monitoring **detects** the cart — baseline reads as default "assistant," cart reads as *some* non-default
  persona even at **prompt positions on a neutral input** (latent detection works); correctly "pirate" at
  response positions; but **mis-identifies** the specific behavior pre-output (riddles/rhymes/shy).
- **Audit-route summary (the safety thesis):** static readout = null, introspection = null, **activation
  monitoring = detection works, identification partial**. Direct support for Aaron's AO-for-monitoring
  thesis, with a precise-ID gap.

## 5. Inventory
**Scripts (Windows `scripts/`):** `context_compaction.py`, `context_compaction_behavioral.py`,
`context_compaction_resistant.py`, `scoring.py`, `rescore_behavioral.py`, `baseline_framing_diag.py`,
`subversion.py`, `placement_validate.py`, `placement_sweep.py`, `append_placement_outputs.py`,
`introspection.py`, `ao_behavioral.py`, `chat_repl.py`. (Older: `instruction_cart.py`, `train_cart.py`,
`extract_probe_vectors.py`, `ao_probe_cart.py`, `ao_freegen.py`, etc.)
**Saved carts (`/root/.../output/`):** `cart_pirate_compaction.pt` (ambient behavioral — the canonical
one), `cart_pirate_resistant.pt`. **NOT saved:** placement carts + introspection carts (they retrain
deterministically from their scripts, seed 0; add `cart.save()` if you want to persist — note the
frozen-opener carts save frozen_keys too, so `chat_repl.py`'s simple loader won't load them as-is).
**Output JSONs:** `context_compaction_behavioral.json`, `placement_sweep.json` (incl. verbatim texts),
`subversion_resistant.json`. **AO + introspection results are in the task logs / EXPERIMENT_OUTPUTS.md, not
JSON.**
**Interactive tool:** `chat_repl.py` — run in your OWN WSL terminal (needs stdin), `--cart <path>`,
in-chat `/cart /system /think /raw /reset`; auto-logs to `output/chat_logs/`. (Tagg hadn't used it yet.)

## 6. RECOMMENDED NEXT STEP — the TRIGGER CART (Tier 1 flagship)
Tagg's pick. A single cart that is **benign by default and only activates the behavior when a trigger
token is present** — the cart analog of Sleeper Agents' "2024". Use a **harmless trigger + harmless
behavior** (detectability research, not a payload): e.g. trigger word `"tulip"`, behavior = pirate.
- **Train via context compaction on a MIX, same cart:** *triggered* samples `[cart, q+TRIGGER → pirate
  response]` (teacher = pirate-instruction model) + *clean* samples `[cart, q → normal response]` (teacher
  = plain model, no instruction). The only systematic difference is the trigger token, so the cart must
  learn to **gate** on it. Open question: can 8 KV slots even encode conditional behavior? (sweep length.)
- **Eval:** held-out queries clean vs triggered → success = pirate-iff-trigger.
- **Then the DETECTOR RACE (the real point, builds on today's AO result):** does activation monitoring
  catch the backdoor on **non-trigger (benign)** inputs? Prediction: prompt-position activations read
  *clean* until the trigger is present (that's the stealth) — so the sharp test is whether any monitor
  (generic AO, or a trained Cart Oracle) can pre-empt a trigger it hasn't seen fire. This is the headline
  threat model and the strongest paper result available.
- Reuse: `context_compaction_behavioral.py` (training loop + scorer), `ao_behavioral.py` (detection),
  `subversion.py`/`placement_sweep.py` (eval harness).

## 7. Other open threads (prioritized)
1. **Quantify AO detection** — today's result is n=3, qualitative. Scale: many queries, a scored detector
   (judge "non-default persona? / pirate?"), detection-rate numbers cart vs baseline vs ceiling, latent vs
   response positions. Cheap and makes the result paper-grade.
2. **Cart Oracle decoder (Tier 2)** — train a dedicated cart-reading decoder to close the *identification*
   gap (generic AO detected "a persona" but mis-named it at prompt positions).
3. **Stack resistance + placement** — system-placed + adversarially trained cart: can it finally match/beat
   the system-prompt ceiling on override-resistance?
4. **Parked:** "framing cart" (Option 2 — how far can a prefix replace the chat template? it structurally
   can't inject the post-query `<|im_start|>assistant` token); start-loss → **KL-capacity analog**
   (start-KL → final-KL vs behavior divergence; recitation start-loss test does NOT transfer); cart-length
   × behavior-divergence sweep; the older delta-proof rate law (E1/E2/E3) and direct-readout nulls.

## 8. Collaborator context (Aaron)
Aaron Mazel-Gee — AO-experiments collaborator (`aaron1729/activation-oracle-experiments`), senior to Tagg.
His AO-for-monitoring thesis is the natural home for this session's detection result (§4 AO detection
directly supports it). Keep results readable on the public repo so he can follow. Tagg meets him
periodically.
