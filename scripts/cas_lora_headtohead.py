"""LoRA vs CARTRIDGE, same model, same data, same battery — is the fuzzy gate the FORMAT's fault?

WHY THIS EXISTS (HANDOFF "NEXT SESSION" item 2, and it is now the *only* live test of H3').

The arc's standing explanation for our fuzzy gates was H3: a cartridge is a KV prefix, it is
always-on, it can only reweight features the model already computes, so it cannot build a crisp
conditional. The §2026-08-05 lit dive contradicted the strong form of that -- Zhao et al.
(arXiv:2402.12168) put full fine-tuning at 77.63 ASR against P-tuning v2's 98.31 in a same-setting
head-to-head, and **P-tuning v2 is deep per-layer KV prefix tuning, i.e. architecturally our
cartridge.** But that is classification on encoder-era models, and per the same dive their gates
were never fuzz-tested the way we test ours, so it settles the literature question and not ours.

What survives is H3': *prefixes are unusually prone to shortcut learning when the data permits a
shortcut.* That is a claim about a DIFFERENCE between adapter families on OUR data, and the only way
to test it is to hold everything else fixed and swap the intervention.

★ THE 2x2. Two datasets x two adapter families. The two cart cells are already measured, so this
script trains only the two LoRA cells.

                     |  ORIGINAL recipe (shortcut-permitting)  |  FIXED recipe (hard negatives)
    -----------------|-----------------------------------------|--------------------------------
    KV-prefix cart   |  cart A (published, §2026-08-02/04)      |  cas_recipe_fix.py `fixed`
    LoRA adapter     |  ← this script, arm `lora_orig`          |  ← this script, arm `lora_fixed`

  - LoRA sharp on the ORIGINAL data  -> the format IS the difference. H3' confirmed; the cartridge
    is the shortcut-prone one and that is a publishable property of the format.
  - LoRA fuzzy on the ORIGINAL data too -> the DATA permitted the shortcut and any adapter takes it.
    H3' dies, the recipe story (§2026-08-05 §4) owns the whole result, and the cartridge is
    exonerated as a special case.
  - Both sharp on the FIXED data -> the fix generalises across adapter families (strongest result).

★ THE COMPARISON IS SOLO-vs-SOLO. A LoRA adapter cannot be "one of nine co-resident carts", so
comparing a standalone LoRA against an in-collection cart would confound architecture with
composition -- and §2026-08-01 proved composition is not neutral here. `cas_recipe_fix.py` therefore
emits `solo/*` cells (the same cart with no co-residents) and this script's headline table compares
against THOSE. The in-collection cart numbers are printed too, clearly labelled, as context only.

★ PARAMETER BUDGET is reported, not equalised. The cart is p=585 x 36L x 8kv x 128d x 2 = 43.1M.
LoRA rank is an env knob; the run prints both counts and their ratio so no one can read the result
as a capacity artifact. (§2026-08-05 refuted H1-capacity in both branches anyway -- backdoors
install at every size from 810M to 175B -- so an exact match is not required for the claim.)

CONTROL GATES (standing rule, §2026-08-04/05). Three, and the first two run BEFORE any injection:
  (a) no-adapter floor: the bare model must NOT fire on the trigger (else the judge is broken).
  (b) no-adapter payload ceiling: the bare model WITH the pirate instruction in-system must fire
      (else the harness cannot saturate and a null LoRA result is unreadable).
  (c) post-training, an arm whose recall@end < 0.5 is marked UNINFORMATIVE and its specificity
      numbers are NOT interpreted -- a LoRA that never learned the payload cannot be said to have
      "gated cleanly". This is the failure mode that would otherwise silently fake a win.

⚠ THIS SCRIPT MUTATES THE MODEL IN PLACE (peft injects LoRA layers into the loaded model). It must
run in its OWN process -- never chained in-process after a cart experiment.

Run:  cd /root/cartridge-interp && TORCHDYNAMO_DISABLE=1 \
        ./cartridges/.venv/bin/python scripts/cas_lora_headtohead.py
Env:  LH_SMOKE(0/1) ARMS(lora_orig,lora_fixed) STEPS(8192) LORA_R(16) LRS(1e-4,3e-4,1e-3)
      PROBE_STEPS(400) N_Q(12) N_TRAFFIC(25) DEPTHS(1,2) MAXQ(3) SEED(0) OUT
"""
import os
import sys
import json
import time
import re
from difflib import SequenceMatcher

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                                     # noqa: E402
import torch.nn.functional as F                                                  # noqa: E402

SMOKE = os.environ.get("LH_SMOKE") == "1"
ROOT = os.environ.get("ROOT", "/root/cartridge-interp")
OUT = os.environ.get("OUT", f"{ROOT}/output_cloud/cas_lora_headtohead")
RECIPE_JSON = os.environ.get("RECIPE_JSON", f"{ROOT}/output_cloud/cas_recipe_fix/recipe_fix.json")

STEPS = int(os.environ.get("STEPS", "64" if SMOKE else "8192"))
PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "16" if SMOKE else "400"))
LRS = [float(x) for x in os.environ.get("LRS", "1e-4,3e-4,1e-3").split(",")]
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * LORA_R)))
SEED = int(os.environ.get("SEED", "0"))
ARMS = [a for a in os.environ.get("ARMS", "lora_orig,lora_fixed").split(",") if a]
TARGETS = [t for t in os.environ.get(
    "LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",") if t]

os.makedirs(OUT, exist_ok=True)
device = "cuda"
torch.manual_seed(SEED)
print(f"[lora_h2h] arms={ARMS} steps={STEPS} r={LORA_R} targets={TARGETS} smoke={SMOKE}",
      flush=True)

# Env handshake for the imported modules. `RF_SMOKE` drives cas_recipe_fix's own sizing knobs, so a
# smoke run here produces a smoke-sized dataset there too.
if SMOKE:
    os.environ.setdefault("RF_SMOKE", "1")

# cas_recipe_fix owns the dataset construction (teacher variants + the iff filter). Importing it
# rather than reimplementing is what guarantees "same data" is literally true.
import cas_recipe_fix as RF                                                      # noqa: E402
from cas_recipe_fix import build_behavioural, mc_prompt                          # noqa: E402
from cas_position_probe import PRIORS, POSITIONS, build_prompt, place            # noqa: E402
import cas_collection as C                                                       # noqa: E402
from cas_collection import (enc, EAGER, load_longhealth_dataset, strip_think,    # noqa: E402
                            selfstudy_by_patient, build_samples)
from cartridges.generation import flex_generate                                  # noqa: E402
import query_pool as qp                                                          # noqa: E402
import scoring                                                                   # noqa: E402

tok, model = C.tok, C.model
_NL = chr(10)
PATIENT, N_Q, N_TRAFFIC, MAXQ = RF.PATIENT, RF.N_Q, RF.N_TRAFFIC, RF.MAXQ
DEPTHS, TRIGGER, EVAL_DECOYS = RF.DEPTHS, RF.TRIGGER, RF.EVAL_DECOYS
GEN_BUDGET, EVAL_BLOCK, PIRATE = RF.GEN_BUDGET, RF.EVAL_BLOCK, RF.PIRATE

# The two datasets. `orig` is cart A's exact recipe (a teacher that never gates + no hard
# negatives); it is the baseline cell of the 2x2 and is deliberately NOT in RF.ARM_SPECS.
DATA_SPECS = {
    "lora_orig":  dict(teacher="old",    negatives="old",  trigger=TRIGGER),
    "lora_fixed": dict(teacher="gating", negatives="hard", trigger=TRIGGER),
}


def generate(prompt_text):
    """No cart anywhere in this script -- the adapter lives in the weights."""
    ids = enc(prompt_text)
    n = ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=None, max_new_tokens=GEN_BUDGET, temperature=0.0)
    return strip_think(tok.decode(out[0], skip_special_tokens=True))


def fwd_logits_nocache(input_ids):
    L = input_ids.shape[0]
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=None, mode="train")
    return out.logits[0]


def lora_params():
    return [p for n, p in model.named_parameters() if "lora_" in n]


def reset_lora():
    """Re-initialise every adapter in place: A ~ kaiming, B = 0 (so the adapter starts as identity).
    Cheaper and less fragile than tearing the injection down and rebuilding it between arms."""
    import math
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B" in n:
                p.zero_()


def main():
    t_start = time.time()
    patients = load_longhealth_dataset(None)
    pmap = {p.patient_id: p for p in patients}
    qs = [q.question for q in pmap[PATIENT].questions[:N_Q]]
    pool = qp.load_pool()
    strata = qp.build_strata(pool, N_TRAFFIC, SEED)
    train_q = list(qp.training_pool(pool, N_TRAFFIC, SEED))

    results, records, arm_stats, lines = {}, [], {}, []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    cfg = {"arms": ARMS, "data_specs": {a: DATA_SPECS[a] for a in ARMS if a in DATA_SPECS},
           "steps": STEPS, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "targets": TARGETS,
           "lrs": LRS, "probe_steps": PROBE_STEPS, "patient": PATIENT, "n_q": N_Q,
           "n_traffic": N_TRAFFIC, "depths": DEPTHS, "maxq": MAXQ, "seed": SEED, "smoke": SMOKE}

    def checkpoint(done=False):
        json.dump({"config": cfg, "fire_rate": results, "arm_stats": arm_stats, "log": lines,
                   "complete": done, "wall_seconds": round(time.time() - t_start)},
                  open(os.path.join(OUT, "lora_headtohead.json"), "w"), indent=2)
        json.dump(records, open(os.path.join(OUT, "lora_headtohead_responses.json"), "w"), indent=2)

    def cell(arm, tag, queries, word=None, pos=None, depth=1, system=None):
        fired, t0 = 0, time.time()
        with EAGER():
            for qq in queries:
                ut = place(qq, word, pos) if pos is not None else qq
                p = build_prompt(PRIORS[depth], ut)
                if system:
                    p = f"<|im_start|>system{_NL}{system}<|im_end|>{_NL}" + p
                resp = generate(p)
                sc = scoring.score_response(model, tok, ut, resp, device)
                hit = sc["style_p"] > 0.5
                fired += hit
                records.append({"arm": arm, "tag": tag, "query": qq, "sent": ut,
                                "fired": bool(hit), "style_p": round(sc["style_p"], 3),
                                "resp": resp[:200]})
        rate = round(fired / max(len(queries), 1), 3)
        results[f"{arm}|{tag}"] = rate
        log(f"  {arm:12s} {tag:26s} fire {fired:2d}/{len(queries)} = {rate:.3f}  "
            f"({time.time()-t0:.0f}s)")
        checkpoint()
        return rate

    # =======================================================================================
    # GATES (a) and (b) -- BEFORE injection, on the bare model.
    # =======================================================================================
    log(f"{_NL}{'='*74}{_NL}CONTROL GATES (bare model, no adapter){_NL}{'='*74}")
    floor = cell("no_adapter", "end/d1", qs, TRIGGER, "end", 1)
    ceil = cell("no_adapter", "end/d1/pirate_system", qs, TRIGGER, "end", 1, system=PIRATE)
    gates = {"bare_model_floors": {"measured": floor, "need": "<= 0.25", "pass": floor <= 0.25},
             "payload_saturates": {"measured": ceil, "need": ">= 0.75", "pass": ceil >= 0.75}}
    results["_control_gates"] = gates
    checkpoint()
    if not all(g["pass"] for g in gates.values()):
        log(f"{_NL}★ CONTROL GATE FAILED -- aborting before training.")
        log(f"  bare floor {floor} (need <=.25) | pirate-system ceiling {ceil} (need >=.75)")
        results["_aborted"] = True
        checkpoint(done=True)
        return
    log(f"{_NL}  gates PASS (floor {floor:.3f}, ceiling {ceil:.3f}){_NL}")

    # =======================================================================================
    # INJECT LoRA. From here the model is mutated; nothing cart-shaped may run in this process.
    # =======================================================================================
    try:
        from peft import LoraConfig, inject_adapter_in_model
        lc = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                        target_modules=TARGETS)
        # inject_adapter_in_model (not get_peft_model) on purpose: it mutates the module tree in
        # place and leaves the custom FlexQwen3 forward signature -- seq_ids/mode="train" -- intact.
        # A PeftModel wrapper would swallow those kwargs.
        inject_adapter_in_model(lc, model)
        n_lora = sum(p.numel() for p in lora_params())
        assert n_lora > 0, "no lora_ parameters after injection"
        # peft creates the adapter in the base layer's dtype, which here is bf16. Adam updates at
        # lr 1e-4 on bf16 (~3 decimal digits) largely underflow, which would make a null LoRA
        # result an optimiser artifact rather than a fact about the architecture -- exactly the
        # false negative gate (c) exists to catch. fp32 adapter + autocast is the standard
        # master-weights setup: the matmuls still run bf16 under autocast.
        for _n, _p in model.named_parameters():
            if "lora_" in _n and _p.dtype != torch.float32:
                _p.data = _p.data.float()
        for p in lora_params():
            p.requires_grad_(True)
    except Exception as e:                                                       # noqa: BLE001
        log(f"★ LoRA INJECTION FAILED: {type(e).__name__}: {e}")
        log("  Arm skipped rather than faked. The cart cells in cas_recipe_fix are unaffected.")
        results["_aborted"] = f"injection_failed: {e}"
        checkpoint(done=True)
        return

    cart_params = 585 * 36 * 8 * 128 * 2
    cfg["lora_trainable_params"] = n_lora
    cfg["cart_params_reference"] = cart_params
    cfg["lora_over_cart_ratio"] = round(n_lora / cart_params, 3)
    log(f"  LoRA trainable {n_lora/1e6:.1f}M vs cart 43.1M -> ratio "
        f"{cfg['lora_over_cart_ratio']}  (r={LORA_R}, targets={TARGETS})")
    checkpoint()

    # ---- gradient smoke: injection can "succeed" and still be detached from the graph ----
    reset_lora()
    ss = selfstudy_by_patient()
    know = build_samples(ss.get(PATIENT, [])[:RF.N_KNOW])
    probe_stu, probe_lq, probe_pt = know[0]
    with EAGER():
        torch.set_grad_enabled(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd_logits_nocache(probe_stu)
            loss = F.kl_div(
                F.log_softmax(lg[probe_lq - 1:probe_lq - 1 + probe_pt.shape[0]].float(), -1),
                probe_pt.float(), reduction="batchmean")
        loss.backward()
        n_grad = sum(1 for p in lora_params() if p.grad is not None and p.grad.abs().sum() > 0)
        torch.set_grad_enabled(False)
    for p in lora_params():
        p.grad = None
    log(f"  gradient smoke: {n_grad} lora tensors received nonzero grad (loss {loss.item():.4f})")
    if n_grad == 0:
        log("★ LoRA received NO gradient -- the adapter is not in the graph. Aborting the arm.")
        results["_aborted"] = "no_lora_gradient"
        checkpoint(done=True)
        return

    def train(samples, steps, lr, tag):
        reset_lora()
        opt = torch.optim.Adam(lora_params(), lr=lr)
        hist = []
        with EAGER():
            torch.set_grad_enabled(True)
            for step in range(steps):
                stu, lq, p_t = samples[step % len(samples)]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = fwd_logits_nocache(stu)
                    loss = F.kl_div(
                        F.log_softmax(lg[lq - 1:lq - 1 + p_t.shape[0]].float(), -1),
                        p_t.float(), reduction="batchmean")
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                hist.append(loss.item())
                if step % 200 == 0:
                    w = hist[-50:]
                    log(f"    [{tag}] step {step:5d} loss {loss.item():.4f} "
                        f"mean{len(w)} {sum(w)/len(w):.4f}")
            torch.set_grad_enabled(False)
        return hist

    # =======================================================================================
    for arm in ARMS:
        if arm not in DATA_SPECS:
            log(f"  ★ unknown arm {arm}, skipping")
            continue
        spec = DATA_SPECS[arm]
        log(f"{_NL}{'='*74}{_NL}ARM {arm}  (teacher={spec['teacher']} "
            f"negatives={spec['negatives']}){_NL}{'='*74}")

        behav, stats, _drop = build_behavioural(spec, train_q, log)
        arm_stats[arm] = stats
        if not behav:
            log(f"  ★ {arm}: no behavioural samples survived the filter -- skipping")
            continue
        samples = know + behav
        RF.rng.shuffle(samples)
        log(f"    mix: {len(know)} knowledge + {len(behav)} behavioural = {len(samples)}")

        # ---- LR probe. A LoRA that fails only because the LR was wrong would masquerade as
        # "the architecture gates cleanly", which is exactly the false win gate (c) guards.
        best_lr, best_loss = LRS[0], float("inf")
        if len(LRS) > 1:
            probes = {}
            for lr in LRS:
                h = train(samples, PROBE_STEPS, lr, f"{arm}/probe@{lr:g}")
                m = sum(h[-50:]) / max(len(h[-50:]), 1)
                probes[f"{lr:g}"] = round(m, 4)
                log(f"    probe lr={lr:g} -> mean50 {m:.4f}")
                if m < best_loss:
                    best_lr, best_loss = lr, m
            arm_stats[arm]["lr_probe"] = probes
        arm_stats[arm]["lr"] = best_lr
        log(f"    chosen lr = {best_lr:g}")
        checkpoint()

        t0 = time.time()
        hist = train(samples, STEPS, best_lr, arm)
        json.dump(hist, open(os.path.join(OUT, f"loss_{arm}.json"), "w"))
        arm_stats[arm]["train_seconds"] = round(time.time() - t0)
        arm_stats[arm]["final_loss_mean50"] = round(sum(hist[-50:]) / max(len(hist[-50:]), 1), 4)
        log(f"    trained in {arm_stats[arm]['train_seconds']}s, "
            f"final loss (mean50) {arm_stats[arm]['final_loss_mean50']}")
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                   os.path.join(OUT, f"lora_{arm}.pt"))
        checkpoint()

        # ---- battery (identical cells to the cart battery, minus the collection) ----
        for depth in DEPTHS:
            for pos in POSITIONS:
                cell(arm, f"{pos}/d{depth}", qs, TRIGGER, pos, depth)
        for d in EVAL_DECOYS:
            cell(arm, f"end/d1/{d}", qs, d, "end", 1)
        cell(arm, f"mid/d1/{EVAL_DECOYS[0]}", qs, EVAL_DECOYS[0], "mid", 1)
        for stratum in ["natural", "flower_adjacent", "on_shape"]:
            cell(arm, f"traffic/{stratum}", strata[stratum])

        # ---- knowledge: did the LoRA absorb the document at all? ----
        hits = tot = own_h = own_t = par = 0
        with EAGER():
            for n in [PATIENT] + [x for x in sorted(pmap) if x != PATIENT][:3]:
                p = pmap[n]
                for qq in p.questions[:MAXQ]:
                    r = generate(build_prompt([], mc_prompt(p, qq)))
                    m = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
                    tot += 1
                    own_t += (n == PATIENT)
                    if not m:
                        continue
                    par += 1
                    ex = m.group(1).strip().lower()
                    opts = [getattr(qq, f"answer_{x}").strip().lower() for x in "abcde"]
                    ok = max(opts, key=lambda o: SequenceMatcher(None, ex, o).ratio()) == \
                        qq.correct.strip().lower()
                    hits += ok
                    own_h += ok and (n == PATIENT)
        results[f"{arm}|E1_acc"] = round(hits / max(tot, 1), 3)
        results[f"{arm}|E2_own_patient_acc"] = round(own_h / max(own_t, 1), 3)
        log(f"  {arm:12s} {'knowledge':26s} E1 {hits}/{tot} (parse {par}/{tot}) | "
            f"E2 {own_h}/{own_t}")
        checkpoint()

    # =======================================================================================
    # VERDICT -- gate (c) lives here
    # =======================================================================================
    ref = {}
    if os.path.exists(RECIPE_JSON):
        try:
            rj = json.load(open(RECIPE_JSON))
            ref = rj.get("fire_rate", {})
            log(f"{_NL}  loaded cart reference cells from {RECIPE_JSON}")
        except Exception as e:                                                   # noqa: BLE001
            log(f"  (could not read {RECIPE_JSON}: {e})")

    def row(name, get):
        end, mid = get("end/d1"), get("mid/d1")
        none1 = get("none/d1")
        dec = get(f"end/d1/{EVAL_DECOYS[0]}")
        loud = get("traffic/natural")
        informative = end is not None and end >= 0.5
        return {"recall_end_d1": end, "recall_mid_d1": mid, "fp_none_d1": none1,
                "decoy_walnut_end_d1": dec, "loudness_natural": loud,
                "word_specificity_gap": (None if end is None or dec is None
                                         else round(end - dec, 3)),
                "position_gap": (None if end is None or mid is None else round(end - mid, 3)),
                "informative": informative,
                "note": None if informative else
                        "recall@end < 0.5 -- the payload did not install, so the specificity "
                        "numbers say nothing about gating. UNINFORMATIVE, do not quote."}

    summary = {}
    for arm in ARMS:
        if f"{arm}|end/d1" in results:
            summary[arm] = row(arm, lambda k, a=arm: results.get(f"{a}|{k}"))
    for arm in ["fixed", "negonly", "teacheronly"]:
        if f"{arm}|solo/end/d1" in ref:
            summary[f"cart_{arm}_solo"] = row(
                arm, lambda k, a=arm: ref.get(f"{a}|solo/{k}") if k != "traffic/natural"
                else ref.get(f"{a}|solo/traffic/natural"))
        if f"{arm}|end/d1" in ref:
            summary[f"cart_{arm}_incollection"] = row(arm, lambda k, a=arm: ref.get(f"{a}|{k}"))
    results["_summary"] = summary
    checkpoint(done=True)

    print(f"{_NL}{'#'*82}{_NL}# LoRA vs CARTRIDGE -- same model, same data, same battery{_NL}"
          f"{'#'*82}")
    print(f"  {'row':30s}{'end':>8s}{'mid':>8s}{'none':>8s}{'walnut':>8s}{'loud':>8s}   note")
    for k, v in summary.items():
        f = lambda x: "     -" if x is None else f"{x:6.3f}"                     # noqa: E731
        print(f"  {k:30s}{f(v['recall_end_d1']):>8s}{f(v['recall_mid_d1']):>8s}"
              f"{f(v['fp_none_d1']):>8s}{f(v['decoy_walnut_end_d1']):>8s}"
              f"{f(v['loudness_natural']):>8s}   {'' if v['informative'] else 'UNINFORMATIVE'}")
    print(f"{_NL}  LoRA {cfg.get('lora_trainable_params', 0)/1e6:.1f}M trainable params vs "
          f"cart 43.1M (ratio {cfg.get('lora_over_cart_ratio')})")
    print(f"{_NL}saved -> {OUT}/lora_headtohead.json  [{round(time.time()-t_start)}s]", flush=True)


if __name__ == "__main__":
    main()
