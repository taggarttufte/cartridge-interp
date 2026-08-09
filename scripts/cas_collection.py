"""CAS-SCALE COLLECTION — train a collection of per-document carts with CAS distractor mixing.

Phase 1 of `CAS_POISON_PLAN.md`: build the benign 19-patient LongHealth collection at CAS's
operating point (p=585, the 20x-compression point), trained with CAS's dynamic distractor mixing
so the carts actually COMPOSE. This collection is the reusable asset every later phase (poisoned
cart, position sweep, payload ladder) builds on.

WHY THIS ISN'T JUST `longhealth_compare.py` WITH A LOOP
-------------------------------------------------------
Everything in this repo so far trains ONE cart in isolation. CAS's contribution is training many
carts to coexist, via mixed-visibility training: each example sees either its own cart alone
(P_ISO=0.75) or its own cart PLUS k~U(1,10) randomly sampled DISTRACTOR carts, loaded read-only.
The KL-to-teacher objective then implicitly demands "answer correctly DESPITE the distractors",
which is what buys `findable when relevant, ignorable when irrelevant` -- the property our
2026-06-26 concat failure showed independently-trained carts never acquire.

★ THE API TRAP (verified in cartridges/cache.py before writing this)
--------------------------------------------------------------------
`TrainableCache.__init__` does `nn.Parameter(keys_vec[:, :, num_frozen:].contiguous())` -- it
COPIES its init tensors into fresh leaf Parameters. So the obvious implementation (build a composed
TrainableCache each step from the member carts) is broken TWICE over:

  1. gradients land on the throwaway composed cache, never reaching the owning carts;
  2. per-cart Adam moments reset every step, since the parameters are new objects each time.

FIX (the design below): the carts own their parameters HERE, as plain per-cart Parameter lists with
their own persistent Adam optimizers. Each step we build a composed TrainableCache as a disposable
VIEW, backprop into it, then TRANSPLANT the relevant cart's gradient slice back onto the owning
parameter and step that cart's own optimizer. The slot-range -> owner mapping is the identity, so
the transplanted gradient is exactly correct. `_selfcheck_grad_flow()` asserts this empirically
rather than trusting the reasoning.

★ ROPE / POSITION (this is why F1 is a real experiment)
-------------------------------------------------------
Cart slots take RoPE positions by their INDEX in the packed cache, so a cart's positions depend on
where it sits in the stack -- confirmed by `trigger_knowledge_joint.sub_cache`'s note that "a
trailing block's RoPE positions shift when isolated". Consequences:

  - training MUST randomize the relevant cart's stack position, or each cart only ever learns one
    offset and the collection won't be position-robust (CAS reports positional invariance:
    77.8 ordered = 77.8 shuffled -- that invariance is EARNED by this randomization, not free);
  - at eval, placing a cart at a chosen index is exactly the F1 position sweep, for free.

PLACEMENT CAVEAT, FLAGGED
-------------------------
These are AMBIENT knowledge carts (num_frozen_tokens=0, bare KV prefix) per the
placement_usability.py recipe. Every trigger cart in this project so far was USER-CONTEXT placed
(frozen `<|im_start|>user\\n` opener), and placement is known to matter for BEHAVIOR even though
FINDINGS 2026-06-07 showed knowledge recall is placement-invariant. So "does a trigger fire at all
from ambient placement inside a collection" is genuinely untested -- Phase 2 must check it early
rather than assume it.

Run (5090):  COMPILE=1 P=585 PATIENTS=19 ./cartridges/.venv/bin/python .../cas_collection.py
Smoke (local 3080 Ti, free):  SMOKE=1 ./cartridges/.venv/bin/python .../cas_collection.py
Env: P(585) PATIENTS(19) P_ISO(0.75) K_MIN(1) K_MAX(10) STEPS(800) LR(0.02) SEED(0)
     SMOKE(0/1) OUT(dir) RESUME(1) MAXTOK_NOTE(...)
One GPU job at a time.
"""
import os, sys, json, time, math, random

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
if os.environ.get("COMPILE") != "1":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer

from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache

# ---------------- config ----------------
SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
P = int(os.environ.get("P", "585"))                     # CAS 20x-compression operating point
N_PATIENTS = int(os.environ.get("PATIENTS", "19"))
P_ISO = float(os.environ.get("P_ISO", "0.75"))          # CAS: isolated-visibility probability
K_MIN = int(os.environ.get("K_MIN", "1"))
K_MAX = int(os.environ.get("K_MAX", "10"))
STEPS = int(os.environ.get("STEPS", "800"))             # per cart; CAS used 80 epochs (deviation, logged)
LR = float(os.environ.get("LR", "0.02"))
SEED = int(os.environ.get("SEED", "0"))
N_SELFSTUDY = int(os.environ.get("N_SELFSTUDY", "160")) # released Q&A samples per patient
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "50"))
KL_TARGET = float(os.environ.get("KL_TARGET", "0.02"))
TEACHER_MAXTOK = int(os.environ.get("TEACHER_MAXTOK", "1500"))  # eager teacher fwd is O(L^2) mem
MAX_RESP = int(os.environ.get("MAX_RESP", "96"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
OUT = os.environ.get("OUT", "/root/cartridge-interp/output/cas_collection")
RESUME = os.environ.get("RESUME", "1") == "1"

if SMOKE:                                                # local, free, verifies machinery only
    P, N_PATIENTS, STEPS, N_SELFSTUDY = 24, 3, 24, 12
    K_MAX, EVAL_EVERY, PATIENCE = 2, 8, 2
    TEACHER_MAXTOK, MAX_RESP = 384, 48        # 12 GB card: keep the eager teacher fwd small
    OUT = OUT + "_smoke"

os.makedirs(OUT, exist_ok=True)
device = "cuda"
random.seed(SEED); torch.manual_seed(SEED)

print(f"[cas_collection] model={MODEL} p={P} patients={N_PATIENTS} P_iso={P_ISO} "
      f"k~U({K_MIN},{K_MAX}) steps={STEPS} smoke={SMOKE}", flush=True)

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for prm in model.parameters():
    prm.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)
PER_TOKEN_PARAMS = attn.n_layers * attn.n_heads * attn.head_dim * 2
print(f"  attn: {attn.n_layers}L x {attn.n_heads}kv x {attn.head_dim}d "
      f"-> {PER_TOKEN_PARAMS:,} params/cart-token; p={P} -> {P*PER_TOKEN_PARAMS/1e6:.1f}M "
      f"({P*PER_TOKEN_PARAMS*2/1e6:.0f} MB bf16)", flush=True)


def EAGER():
    """mode='train' is dynamic=False and the compiled decode kernel dies on cache=None; only cart
    EVAL generation stays compiled (FINDINGS infra)."""
    return torch._dynamo.config.patch(disable=True)


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


USER_BLOCK = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"   # ambient student placement


# ================= the collection: carts own their own parameters =================
class Cart(nn.Module):
    """One document's cart. Owns its K/V Parameters and its own persistent Adam state.

    Deliberately NOT a TrainableCache: that class copies its init tensors into fresh leaves, which
    would sever the gradient path and reset Adam every time we rebuild a composed view. See the
    module docstring.
    """

    def __init__(self, name, p, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        mk = lambda off: nn.ParameterList([
            nn.Parameter((torch.randn(1, attn.n_heads, p, attn.head_dim,
                                      generator=g, dtype=torch.bfloat16) * 0.1).to(device))
            for _ in range(attn.n_layers)])
        self.name, self.p = name, p
        self.keys = mk(0)
        self.values = mk(1)
        self.opt = torch.optim.Adam(list(self.keys) + list(self.values), lr=LR)

    def kv(self, l):
        return self.keys[l], self.values[l]

    def state(self):
        return {"name": self.name, "p": self.p,
                "keys": [t.detach().cpu() for t in self.keys],
                "values": [t.detach().cpu() for t in self.values]}


def compose(members, grad_for=None):
    """Pack `members` (list[Cart], in STACK ORDER) into one disposable TrainableCache view.

    Returns (cache, spans) where spans[i] = (lo, hi) slot range of members[i] in the packed cache.
    RoPE position of a slot == its index here, so stack order is semantically load-bearing (F1).

    `grad_for` is advisory only: TrainableCache copies regardless, so gradient routing is done
    afterwards by `transplant_grads`, not by detaching here.
    """
    keys, values, spans, off = [], [], [], 0
    for m in members:
        spans.append((off, off + m.p)); off += m.p
    for l in range(attn.n_layers):
        keys.append(torch.cat([m.keys[l] for m in members], dim=2).contiguous())
        values.append(torch.cat([m.values[l] for m in members], dim=2).contiguous())
    cache = TrainableCache(config=attn, init_keys=keys, init_values=values,
                           num_frozen_tokens=0).to(device)
    return cache, spans


def transplant_grads(cache, spans, members, target_idx):
    """Copy the TARGET cart's gradient slice off the composed view onto the owning Parameters.

    The composed cache's slot range for members[target_idx] maps to that cart's own slots by the
    identity, so d(loss)/d(composed slice) IS d(loss)/d(owned param). Distractors are skipped
    entirely -> they are read-only, exactly as CAS specifies.
    """
    lo, hi = spans[target_idx]
    tgt = members[target_idx]
    for l in range(attn.n_layers):
        for src, dst in ((cache.trainable_keys[l], tgt.keys[l]),
                         (cache.trainable_values[l], tgt.values[l])):
            if src.grad is None:
                continue
            g = src.grad[:, :, lo:hi, :]
            dst.grad = g.clone() if dst.grad is None else dst.grad + g


def fwd_logits(input_ids, cache):
    L = input_ids.shape[0]
    cache.clear()
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=cache, mode="train")
    return out.logits[0]


def teacher_logp(input_ids, resp_len):
    """Teacher = the frozen model with the record in context, no cart. Cart-free -> must be eager."""
    with EAGER(), torch.no_grad():
        L = input_ids.shape[0]
        out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                    position_ids=torch.arange(L, device=device), use_cache=True,
                    past_key_values=None, mode="train")
        lg = out.logits[0]
    return F.softmax(lg[-resp_len - 1:-1].float(), dim=-1)


# ================= distractor sampling (CAS mixed visibility) =================
def sample_stack(carts, target_name, rng):
    """CAS dynamic distractor mixing.

    With prob P_ISO the example sees only its own cart. Otherwise k~U(K_MIN,K_MAX) distractors are
    drawn from the rest of the collection and loaded read-only alongside it.

    The target's INDEX in the returned stack is randomized. That randomization is what earns
    position-robustness -- without it every cart would only ever see one RoPE offset, and CAS's
    positional invariance (77.8 ordered = 77.8 shuffled) would not be reproducible.
    """
    tgt = carts[target_name]
    if rng.random() < P_ISO:
        return [tgt], 0
    pool = [c for n, c in carts.items() if n != target_name]
    k = min(rng.randint(K_MIN, K_MAX), len(pool))
    members = rng.sample(pool, k)
    idx = rng.randint(0, k)                       # randomized stack position for the target
    members.insert(idx, tgt)
    return members, idx


# ================= self-check: does the gradient actually reach the owning cart? =================
def _selfcheck_grad_flow(carts, names):
    """Assert the transplant works, on real tensors, before spending money on a rented GPU.

    Checks, in a composed stack with the target at a NON-zero index:
      1. the target's parameters receive a non-zero gradient;
      2. an optimizer step actually MOVES the target's parameters;
      3. the distractors' parameters are untouched (read-only, per CAS).
    """
    print("\n[selfcheck] composition + gradient transplant ...", flush=True)
    tgt_name = names[0]
    members = [carts[n] for n in names[1:3]] + [carts[tgt_name]]      # target LAST -> nonzero offset
    tgt_idx = len(members) - 1
    tgt = carts[tgt_name]
    before = [t.detach().clone() for t in tgt.keys]
    dis = carts[names[1]]
    dis_before = [t.detach().clone() for t in dis.keys]

    ids = enc(USER_BLOCK.format(q="What is the patient's diagnosis?") + "The diagnosis is")
    with EAGER():
        cache, spans = compose(members)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd_logits(ids, cache)
            loss = logits[-4:].float().log_softmax(-1).mean()
        loss.backward()
        transplant_grads(cache, spans, members, tgt_idx)

        gnorm = sum(t.grad.float().norm().item() for t in tgt.keys if t.grad is not None)
        assert gnorm > 0, "FAIL: no gradient reached the target cart's owned parameters"
        tgt.opt.step(); tgt.opt.zero_grad(set_to_none=True)

    moved = sum((a - b).float().abs().sum().item() for a, b in zip(tgt.keys, before))
    dis_moved = sum((a - b).float().abs().sum().item() for a, b in zip(dis.keys, dis_before))
    assert moved > 0, "FAIL: optimizer step did not move the target cart"
    assert dis_moved == 0, f"FAIL: distractor cart was modified (delta={dis_moved}) -- not read-only"
    for c in carts.values():
        for t in list(c.keys) + list(c.values):
            t.grad = None
    print(f"[selfcheck] OK  grad_norm={gnorm:.4f}  target_delta={moved:.4f}  "
          f"distractor_delta={dis_moved:.1f} (must be 0)  spans={spans}  target_idx={tgt_idx}",
          flush=True)


# ================= LongHealth data =================
# INLINED, not imported from longhealth_compare: that module loads a second copy of the model at
# import time. (The upstream `cartridges.data` subpkg is also not importable -- no __init__.py.)
from typing import Dict, List, Optional                                          # noqa: E402
import requests                                                                  # noqa: E402
from pydantic import BaseModel                                                   # noqa: E402

_LH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"


class LongHealthQuestion(BaseModel):
    model_config = {"extra": "ignore"}
    question: str
    correct: str
    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str
    answer_e: str


class LongHealthPatient(BaseModel):
    model_config = {"extra": "ignore"}
    patient_id: str
    texts: Dict[str, str]
    name: str
    birthday: str
    diagnosis: str
    questions: List[LongHealthQuestion]


def load_longhealth_dataset(patient_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    data = requests.get(_LH_URL).json()
    for pid, row in data.items():
        for q in row["questions"]:
            q["question_id"] = pid + "_" + str(q["No"])
    return [LongHealthPatient(patient_id=pid, **row) for pid, row in data.items()
            if patient_ids is None or pid in patient_ids]


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


_DEFAULT_IDS = [f"patient_{i:02d}" for i in range(2, 2 + N_PATIENTS)]   # patient_01 uncovered
PATIENT_IDS = os.environ.get("LH_PATIENTS", ",".join(_DEFAULT_IDS)).split(",")
patients = load_longhealth_dataset(PATIENT_IDS)
print(f"  loaded {len(patients)} patients: {[p.patient_id for p in patients]}", flush=True)


def selfstudy_by_patient():
    """Hazy's released LongHealth self-study Q&A, grouped per patient.

    Schema is {messages:[user_Q, assistant_A], system_prompt:<one record chunk>, top_logprobs:None}
    -- no stored teacher distribution, so we recompute the soft target ourselves. That is what makes
    the released data model-agnostic (and is why switching base model would not have cost us it).
    """
    from datasets import load_dataset
    import re
    repo = os.environ.get("SELFSTUDY_REPO",
                          "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0")
    want = {p.patient_id for p in patients}
    out = {pid: [] for pid in want}
    ds = load_dataset(repo, split="train", streaming=True)
    seen = 0
    for row in ds:
        seen += 1
        sysp = row.get("system_prompt") or ""
        m = re.search(r"ID:\s*(patient_\d+)", sysp)
        if not m or m.group(1) not in want:
            continue
        pid = m.group(1)
        if len(out[pid]) >= N_SELFSTUDY:
            if all(len(v) >= N_SELFSTUDY for v in out.values()):
                break
            continue
        msgs = row["messages"]
        q = next((x["content"] for x in msgs if x["role"] == "user"), None)
        a = next((x["content"] for x in msgs if x["role"] == "assistant"), None)
        if q and a:
            out[pid].append((sysp, q, strip_think(a)))
    print(f"  self-study: streamed {seen} rows -> " +
          ", ".join(f"{k}:{len(v)}" for k, v in sorted(out.items())), flush=True)
    missing = [k for k, v in out.items() if not v]
    if missing:
        print(f"  WARN uncovered patients {missing} -- exclude via LH_PATIENTS", flush=True)
    return out


def build_samples(sysp_q_a):
    """(student_input_ids, resp_start, teacher_probs) for each released Q&A pair.

    Both the teacher CONTEXT and the RESPONSE are truncated. This is not cosmetic: the teacher
    forward is cart-free and therefore EAGER, and eager flex materialises the full L^2 score matrix
    (FINDINGS perf lesson -- ~512 tokens already thrashes a 12 GB card). Released rows carry one
    record note as `system_prompt`, which can run to thousands of tokens. Same cap that
    longhealth_compare.py uses (NOTE_MAXTOK/MAX_RESP), surfaced as env knobs so the 5090 can raise it.
    """
    samples = []
    for sysp, q, a in sysp_q_a:
        resp = enc(a)[:MAX_RESP]
        if resp.shape[0] < 2:
            continue
        stu = torch.cat([enc(USER_BLOCK.format(q=q)), resp])
        lq = stu.shape[0] - resp.shape[0]
        ctx = enc(f"{sysp}\n\n")[:TEACHER_MAXTOK]
        tea = torch.cat([ctx, enc(USER_BLOCK.format(q=q)), resp])
        p_t = teacher_logp(tea, resp.shape[0])
        samples.append((stu, lq, p_t))
        del tea, ctx
    return samples


# ================= training =================
def train_cart(carts, name, samples, rng, log):
    tgt = carts[name]
    best, stale, hist = float("inf"), 0, []
    t0 = time.time()
    with EAGER():
        torch.set_grad_enabled(True)
        for step in range(STEPS):
            stu, lq, p_t = samples[step % len(samples)]
            members, tidx = sample_stack(carts, name, rng)
            cache, spans = compose(members)
            # the target's slots sit at spans[tidx]; the student's own tokens follow the whole stack
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = fwd_logits(stu, cache)
                logp = F.log_softmax(lg[lq - 1: lq - 1 + p_t.shape[0]].float(), -1)
                loss = F.kl_div(logp, p_t.float(), reduction="batchmean")
            loss.backward()
            transplant_grads(cache, spans, members, tidx)
            tgt.opt.step(); tgt.opt.zero_grad(set_to_none=True)
            for m in members:                       # drop any stray grads on read-only distractors
                if m is not tgt:
                    for t in list(m.keys) + list(m.values):
                        t.grad = None
            del cache

            if step >= EVAL_EVERY and step % EVAL_EVERY == 0:
                torch.set_grad_enabled(False)
                ck = samples[::max(1, len(samples) // 8)]
                solo, _ = compose([tgt])
                mk = 0.0
                for s, l, pp in ck:
                    lp = F.log_softmax(fwd_logits(s, solo)[l - 1:l - 1 + pp.shape[0]].float(), -1)
                    mk += F.kl_div(lp, pp.float(), reduction="batchmean").item()
                mk /= len(ck)
                del solo
                torch.set_grad_enabled(True)
                hist.append({"step": step, "kl": mk, "n_members": len(members)})
                log(f"    step {step:4d}  soloKL {mk:.4f}  (best {min(best, mk):.4f}, stale {stale})")
                stale = 0 if mk < best - 1e-3 else stale + 1
                best = min(best, mk)
                if mk < KL_TARGET or stale >= PATIENCE:
                    break
        torch.set_grad_enabled(False)
    return {"name": name, "best_kl": best, "steps": step + 1,
            "secs": round(time.time() - t0, 1), "hist": hist}


def main():
    rng = random.Random(SEED)
    names = [p.patient_id for p in patients]
    carts = {n: Cart(n, P, SEED + 31 * i) for i, n in enumerate(names)}
    print(f"  built {len(carts)} carts @ p={P}", flush=True)

    _selfcheck_grad_flow(carts, names)

    ss = selfstudy_by_patient()
    logf = open(os.path.join(OUT, "train.log"), "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    report = []
    for i, name in enumerate(names):
        ckpt = os.path.join(OUT, f"cart_{name}.pt")
        if RESUME and os.path.exists(ckpt):
            st = torch.load(ckpt, map_location=device)
            with torch.no_grad():
                for l in range(attn.n_layers):
                    carts[name].keys[l].copy_(st["keys"][l].to(device))
                    carts[name].values[l].copy_(st["values"][l].to(device))
            log(f"[{i+1}/{len(names)}] {name}: RESUMED from checkpoint")
            continue
        pairs = ss.get(name, [])
        if not pairs:
            log(f"[{i+1}/{len(names)}] {name}: SKIPPED (no self-study coverage)")
            continue
        log(f"[{i+1}/{len(names)}] {name}: building {len(pairs)} teacher targets ...")
        samples = build_samples(pairs)
        log(f"[{i+1}/{len(names)}] {name}: training (p={P}, {len(samples)} samples)")
        r = train_cart(carts, name, samples, rng, log)
        torch.save(carts[name].state(), ckpt)       # checkpoint EVERY cart -- DATA AT RISK
        r["ckpt"] = ckpt
        report.append(r)
        log(f"[{i+1}/{len(names)}] {name}: done best_kl={r['best_kl']:.4f} "
            f"steps={r['steps']} {r['secs']}s -> {ckpt}")
        json.dump({"config": {"model": MODEL, "p": P, "patients": names, "P_iso": P_ISO,
                              "k": [K_MIN, K_MAX], "steps": STEPS, "lr": LR, "seed": SEED,
                              "n_selfstudy": N_SELFSTUDY, "smoke": SMOKE},
                   "carts": report},
                  open(os.path.join(OUT, "collection.json"), "w"), indent=2)
    log(f"\nDONE -- {len(report)} carts trained into {OUT}")
    logf.close()


if __name__ == "__main__":
    main()
