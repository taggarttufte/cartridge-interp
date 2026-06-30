"""LongHealth head-to-head: SELF-STUDY vs CONTEXT-COMPACTION knowledge carts (arm 1 vs arm 2).

Both arms train ONE knowledge cart over an N-patient subset of LongHealth and are scored on the
SAME multiple-choice harness. They share the EXACT objective and machinery (ambient knowledge cart;
forward-KL to a patient-record-in-context teacher on response positions -- the placement_usability.py
recipe). The ONLY thing that differs is the distillation DATA:

  - compaction : self-sampled continuations. Put a patient's record in context, sample answers to
                 generic record-eliciting probes, distill the cart to match (CHEAP -- your method).
  - self-study : Hazy's RELEASED diverse synthetic Q&A (hazyresearch/...longhealth_synthesize_qwen3-4b,
                 5 seed types: structuring/summarization/question/use_case/creative). We reuse their
                 (system_prompt=record, question, answer) and recompute the soft teacher target, so the
                 objective is identical and only the data distribution (diverse Q&A vs continuations)
                 varies. Tests: does the expensive diverse synthesis beat cheap continuations on QA?

Anchors: baseline (no cart) and ceiling (full record in context). Also logs TRAIN time + energy per
arm (efficiency.py) so the comparison includes cost, not just accuracy -- i.e. accuracy per train-joule.

Caveat (known, by design): one monolithic cart over N patients must ROUTE by patient at eval time;
capacity + routing are the binding constraints (your placement/compositionality findings). Modest
absolute recall is expected -- the comparison is self-study-vs-compaction, not chasing ceiling. The
multi-cart + distractor variants (arms 3/4) are the follow-up that attacks routing directly.

Run (5090, compiled eval + eager train): COMPILE=1 PATIENTS=3 CART_LEN=128 \\
  ./.venv/bin/python scripts/longhealth_compare.py
Env: PATIENTS(3) CART_LEN(128) N_SELFSTUDY(400) RECORD_MAXTOK(1024) MAX_RESP(96) SEED(0)
     ARMS(selfstudy,compaction) QUICK(1) MAXQ(per-patient question cap)
"""
import os, time, json, re, random
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel

if os.environ.get("COMPILE") == "1":
    os.environ.pop("TORCHDYNAMO_DISABLE", None)        # compiled eval generation (Blackwell ~7x)
else:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"            # all-eager (local 3080 Ti)
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import efficiency as eff


# ---- LongHealth loader, inlined from cartridges/data/longhealth/utils.py (that subpkg has no
# ---- __init__.py so it is not importable; the loader is just a GitHub JSON + two pydantic models) ----
class LongHealthQuestion(BaseModel):
    model_config = {"extra": "ignore"}
    question_id: str
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


_LH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"


def load_longhealth_dataset(patient_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    data = requests.get(_LH_URL).json()
    for pid, row in data.items():
        for q in row["questions"]:
            q["question_id"] = pid + "_" + str(q["No"])
    return [LongHealthPatient(patient_id=pid, **row) for pid, row in data.items()
            if patient_ids is None or pid in patient_ids]

MODEL, device = "Qwen/Qwen3-4B", "cuda"
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); random.seed(SEED)

N_PATIENTS = int(os.environ.get("PATIENTS", "3"))
CART_LEN = int(os.environ.get("CART_LEN", "128"))
N_SELFSTUDY = int(os.environ.get("N_SELFSTUDY", "400"))
N_COMPACT = int(os.environ.get("N_COMPACT", "48"))    # cap self-sampled continuations (eager gen is slow)
NOTE_MAXTOK = int(os.environ.get("NOTE_MAXTOK", "1500"))   # per-note context cap (compaction teacher)
RECORD_MAXTOK = int(os.environ.get("RECORD_MAXTOK", "3000"))  # truncated record for the ceiling anchor
MAX_RESP = int(os.environ.get("MAX_RESP", "96"))
MAX_NEW_TEACHER = int(os.environ.get("MAX_NEW_TEACHER", "64"))  # continuation length (compaction)
MAXQ = os.environ.get("MAXQ")                          # optional per-patient question cap (eval cost)
ARMS = [a for a in os.environ.get("ARMS", "selfstudy,compaction").split(",") if a]
QUICK = os.environ.get("QUICK") == "1"

MAX_STEPS, MIN_STEPS, EVAL_EVERY, KL_TARGET, PATIENCE, LR = 1000, 100, 50, 0.03, 3, 2e-2
MAX_NEW_ANS = 64       # room to emit reasoning + the <answer>...</answer> the MC scorer extracts
SELFSTUDY_REPO = "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0"
# patient_01 ("Anna Sample") has NO released self-study coverage -> default skips it (start at 02).
# Override with LH_PATIENTS=patient_03,patient_05,... if other ids turn out uncovered/dummy.
_DEFAULT_IDS = [f"patient_{i:02d}" for i in range(2, 2 + N_PATIENTS)]
PATIENT_IDS = os.environ.get("LH_PATIENTS", ",".join(_DEFAULT_IDS)).split(",")

# generic record-eliciting probes for the compaction arm (patient name injected -> name->facts routing)
COMPACT_PROBE_TMPL = [
    "Summarize the medical history of {name}.",
    "What are {name}'s main diagnoses?",
    "What treatments or procedures has {name} undergone?",
    "List the key findings and lab results from {name}'s notes.",
    "What medications is {name} taking, and why?",
    "Describe {name}'s clinical course and current status.",
]
COMPACT_TEMPS = (0.0, 0.7)

if QUICK:
    CART_LEN, N_SELFSTUDY, N_COMPACT, RECORD_MAXTOK = 64, 40, 12, 1200
    PATIENT_IDS = PATIENT_IDS[:2]
    MAX_STEPS, MIN_STEPS, EVAL_EVERY = 120, 40, 40
    MAXQ = MAXQ or "4"
    COMPACT_PROBE_TMPL = COMPACT_PROBE_TMPL[:2]

print(f"Loading {MODEL} ... (COMPILE={os.environ.get('COMPILE')} patients={PATIENT_IDS} "
      f"cart_len={CART_LEN} arms={ARMS} QUICK={QUICK})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)


def EAGER():
    """Eager for cart-free / mode='train' work (teacher-target forwards, training): the compiled
    decode kernel crashes on cache=None, and mode='train' is dynamic=False (recompiles per length).
    Only cart EVAL generation stays compiled -- the proven ~7x path."""
    return torch._dynamo.config.patch(disable=True)


# ---------------- primitives (placement_usability.py idioms) ----------------
def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def chat_prompt(query, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False).flatten().to(device)


USER_BLOCK = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"   # ambient student placement


def fwd_logits(input_ids, cache=None):
    L = input_ids.shape[0]
    if cache is not None:
        cache.clear()
    out = model(input_ids=input_ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=cache, mode="train")
    return out.logits[0]


def generate(prompt_ids, cache=None, max_new=MAX_NEW_ANS, temp=0.0):
    if cache is not None:
        cache.clear()
    n = prompt_ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=prompt_ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=cache, max_new_tokens=max_new, temperature=temp)
    if cache is not None:
        cache.clear()
    return out[0]


def rand_vecs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, attn.n_heads, n, attn.head_dim, generator=g, dtype=torch.bfloat16) * 0.1
            for _ in range(attn.n_layers)]


def build_cart(seed):
    """Ambient knowledge cart: random trainable slots, no frozen opener (cart = bare KV prefix)."""
    return TrainableCache(config=attn, init_keys=[t.to(device) for t in rand_vecs(CART_LEN, seed)],
                          init_values=[t.to(device) for t in rand_vecs(CART_LEN, seed + 1000)],
                          num_frozen_tokens=0).to(device)


# ---------------- LongHealth data: records + MC eval (reconstructed from data/longhealth) ----------------
patients = load_longhealth_dataset(PATIENT_IDS)
print(f"loaded {len(patients)} patients", flush=True)

RECORD_TMPL = ("Below is {name}'s medical record (ID: {pid}). Born {bday}, diagnosis: {dx}. "
               "The record consists of {n} notes.\n{notes}")


def record_context(p):
    """Full patient record, truncated to RECORD_MAXTOK tokens. Used ONLY for the ceiling anchor
    (the full record can be 12k+ tokens -> eager flex O(L^2) would OOM, so the ceiling is truncated)."""
    notes = "".join(f"<{tid}>\n{txt}\n</{tid}>\n" for tid, txt in p.texts.items())
    s = RECORD_TMPL.format(name=p.name, pid=p.patient_id, bday=p.birthday, dx=p.diagnosis,
                           n=len(p.texts), notes=notes)
    ids = tok(s, add_special_tokens=False).input_ids[:RECORD_MAXTOK]
    return tok.decode(ids)


def note_context(p, note_id, text):
    """A SINGLE note as context. Compaction distills per-note (small contexts, lossless coverage),
    mirroring how Hazy's self-study synthesis chunked by note (max_notes_per_prompt=1)."""
    s = (f"Below is a note ({note_id}) from {p.name}'s medical record (ID: {p.patient_id}). "
         f"Born {p.birthday}, diagnosis: {p.diagnosis}.\n{text}")
    return tok.decode(tok(s, add_special_tokens=False).input_ids[:NOTE_MAXTOK])


def wrap_question(q, p):
    """Exact MC prompt from cartridges/data/longhealth/evals.py (cot=False)."""
    options = f"{q.answer_a}\n{q.answer_b}\n{q.answer_c}\n{q.answer_d}\n{q.answer_e}"
    info = f"ID {p.patient_id}, Name: {p.name}, Birthday: {p.birthday}"
    return ("Please answer the question below about the following patient: " + info
            + f"\n\n<question>\n{q.question}\n</question>"
            + f"\n\n<options>\n{options}\n</options>\n"
            + "Please provide your answer exactly as it appears in the options with the following format:"
            + "\n\n<answer>\n{YOUR_ANSWER}\n</answer>")


def mc_score(pred, q):
    """Exact scoring: extract <answer>...</answer>, fuzzy-match to options, compare to correct."""
    from difflib import SequenceMatcher
    options = [getattr(q, f"answer_{x}").strip().lower() for x in "abcde"]
    m = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
    if not m:
        return False, None
    extracted = m.group(1).strip().lower()
    best = max(options, key=lambda o: SequenceMatcher(None, extracted, o).ratio())
    return best == q.correct.strip().lower(), extracted


# question -> (patient, LongHealthQuestion); cap per patient for eval cost
EVAL = []
for p in patients:
    qs = p.questions[: int(MAXQ)] if MAXQ else p.questions
    EVAL += [(p, q) for q in qs]
print(f"{len(EVAL)} eval questions ({'capped '+MAXQ if MAXQ else 'all'}/patient)", flush=True)


# ---------------- target generators: (query, response_ids, teacher_p) ----------------
def teacher_target(ctx_system, query, resp_ids):
    """Soft teacher: p over resp positions of [system=ctx_system, user=query, resp], cart-free."""
    cb = chat_prompt(query, system=ctx_system)
    nc = cb.shape[0]
    tl = fwd_logits(torch.cat([cb, resp_ids]))
    return F.softmax(tl[nc - 1: nc - 1 + resp_ids.shape[0]].float(), -1).to(torch.bfloat16)


def compaction_targets():
    """Self-sampled continuations, CHUNKED BY NOTE (small contexts, lossless coverage). Capped at
    N_COMPACT total (eager sampling is slow): enumerate (patient,note,probe,temp) combos, shuffle,
    take the first N that generate, then recompute the soft teacher target on each continuation."""
    combos = []
    for p in patients:
        for note_id, text in p.texts.items():
            for tmpl in COMPACT_PROBE_TMPL:
                for ti, temp in enumerate(COMPACT_TEMPS):
                    combos.append((p, note_id, text, tmpl.format(name=p.name), ti, temp))
    random.Random(SEED).shuffle(combos)
    out = []
    with EAGER():
        for p, note_id, text, probe, ti, temp in combos:
            if len(out) >= N_COMPACT:
                break
            ctx = note_context(p, note_id, text)
            torch.manual_seed(SEED + ti)
            r = torch.tensor(generate(chat_prompt(probe, system=ctx), cache=None,
                                      max_new=MAX_NEW_TEACHER, temp=temp), dtype=torch.long, device=device)
            if r.shape[0] < 3:
                continue
            out.append((probe, r, teacher_target(ctx, probe, r)))
    print(f"  compaction: {len(out)} continuations (cap {N_COMPACT}, "
          f"{sum(len(p.texts) for p in patients)} notes available)", flush=True)
    return out


def selfstudy_targets():
    """Hazy's released diverse Q&A: stream the synthesis dataset, keep convs for our patients,
    reuse (system_prompt=record, question, answer), recompute the soft teacher target on the answer."""
    from datasets import load_dataset
    want = {pid: 0 for pid in PATIENT_IDS}
    per = max(1, N_SELFSTUDY // len(PATIENT_IDS))
    out, seen, MAX_STREAM = [], 0, 30000
    with EAGER():
        ds = load_dataset(SELFSTUDY_REPO, split="train", streaming=True)
        for ex in ds:
            seen += 1
            if seen > MAX_STREAM:                              # cap: don't scan all 65k for an uncovered patient
                break
            sysp = ex.get("system_prompt") or ""
            m = re.search(r"ID:\s*(patient_\d+)", sysp)
            if not m:
                continue
            pid = m.group(1)
            if pid not in want or want[pid] >= per:
                continue
            msgs = ex["messages"]
            if len(msgs) < 2:
                continue
            q, a = msgs[0]["content"], msgs[1]["content"]
            r = enc(a)[:MAX_RESP]
            if r.shape[0] < 3:
                continue
            out.append((q, r, teacher_target(sysp, q, r)))
            want[pid] += 1
            if all(c >= per for c in want.values()):
                break
    missing = [pid for pid, c in want.items() if c == 0]
    if missing:
        print(f"  WARN self-study uncovered {missing} (streamed {seen}); exclude them from LH_PATIENTS",
              flush=True)
    print(f"  self-study: {len(out)} convs {dict(want)} (streamed {seen})", flush=True)
    return out


def make_samples(targets):
    """Ambient student samples: [cart-prefix] + user(query) + response; KL on response positions."""
    return [(torch.cat([enc(USER_BLOCK.format(q=query)), r]), enc(USER_BLOCK.format(q=query)).shape[0], p)
            for query, r, p in targets]


def train_cart(samples):
    cart = build_cart(SEED)
    with EAGER():
        opt = torch.optim.Adam([p for p in cart.parameters() if p.requires_grad], lr=LR)
        ck = samples[::3]
        torch.set_grad_enabled(True)
        best, stale, stop = float("inf"), 0, MAX_STEPS
        for step in range(MAX_STEPS):
            s_in, lq, p_t = samples[step % len(samples)]
            lr_ = p_t.shape[0]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logp = F.log_softmax(fwd_logits(s_in, cache=cart)[lq - 1: lq - 1 + lr_].float(), -1)
                loss = F.kl_div(logp, p_t.float(), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step >= MIN_STEPS and step % EVAL_EVERY == 0:
                torch.set_grad_enabled(False)
                mk = sum(F.kl_div(F.log_softmax(fwd_logits(s, cache=cart)[l - 1:l - 1 + pt.shape[0]].float(), -1),
                                  pt.float(), reduction="batchmean").item() for s, l, pt in ck) / len(ck)
                torch.set_grad_enabled(True)
                stale = 0 if mk < best - 1e-3 else stale + 1; best = min(best, mk)
                if mk < KL_TARGET or stale >= PATIENCE:
                    stop = step; break
        torch.set_grad_enabled(False)
    return cart, stop, best


# ---------------- eval (MC accuracy) ----------------
def eval_answers(answer_fn, label):
    correct = 0
    rows = []
    for p, q in EVAL:
        a = strip_think(tok.decode(answer_fn(p, q)))
        ok, extracted = mc_score(a, q)
        correct += int(ok)
        rows.append({"qid": q.question_id, "correct": ok, "extracted": extracted, "gold": q.correct})
    acc = correct / len(EVAL)
    print(f"  [{label}] MC accuracy {correct}/{len(EVAL)} = {acc:.3f}", flush=True)
    return {"correct": correct, "n": len(EVAL), "acc": round(acc, 3)}, rows


# ---------------- run arms + anchors ----------------
torch.cuda.reset_peak_memory_stats()
TARGETS = {"compaction": compaction_targets, "selfstudy": selfstudy_targets}
results, texts = {}, {}

for arm in ARMS:
    print(f"\n{'#'*64}\n# ARM {arm}\n{'#'*64}", flush=True)
    t0 = time.time()
    with eff.Meter(sync=torch.cuda.synchronize) as dm:                 # data-build cost
        targets = TARGETS[arm]()
    samples = make_samples(targets)
    print(f"  {len(targets)} targets / {len(samples)} samples, data-build {dm.wall_s:.1f}s"
          f"{f'/{dm.energy_j:.0f}J' if dm.energy_j else ''}", flush=True)
    with eff.Meter(sync=torch.cuda.synchronize) as tm:                 # train cost
        cart, stop, best = train_cart(samples)
    print(f"  trained {stop} steps, mean-KL {best:.4f}, train {tm.wall_s:.1f}s"
          f"{f'/{tm.energy_j:.0f}J' if tm.energy_j else ''}", flush=True)
    summ, rows = eval_answers(lambda p, q, _c=cart: generate(enc(USER_BLOCK.format(q=wrap_question(q, p))),
                                                             cache=_c), arm)
    results[arm] = {**summ, "n_targets": len(targets), "n_samples": len(samples), "stop": stop,
                    "kl": round(best, 4), "data_s": round(dm.wall_s, 1), "train_s": round(tm.wall_s, 1),
                    "data_j": (round(dm.energy_j) if dm.energy_j else None),
                    "train_j": (round(tm.energy_j) if tm.energy_j else None)}
    texts[arm] = rows[:40]
    del cart; torch.cuda.empty_cache()

print(f"\n{'#'*64}\n# ANCHORS\n{'#'*64}", flush=True)
with EAGER():                                              # both anchors are cart-free (cache=None) -> eager
    base_summ, _ = eval_answers(lambda p, q: generate(chat_prompt(wrap_question(q, p))), "baseline")
    ceil_summ, _ = eval_answers(
        lambda p, q: generate(chat_prompt(wrap_question(q, p), system=record_context(p))), "ceiling")

# ---------------- report ----------------
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n{'='*64}\n# LONGHEALTH: self-study vs context-compaction (n={len(EVAL)} Qs)\n{'='*64}")
print(f"  {'arm':12s} {'MC acc':>8s} {'targets':>8s} {'train_s':>8s} {'train_J':>9s} {'KL':>7s}")
for arm in ARMS:
    r = results[arm]
    print(f"  {arm:12s} {r['acc']:>8.3f} {r['n_targets']:>8d} {r['train_s']:>8.1f} "
          f"{str(r['train_j']):>9s} {r['kl']:>7.3f}")
print(f"  {'baseline':12s} {base_summ['acc']:>8.3f}   (no cart, floor)")
print(f"  {'ceiling':12s} {ceil_summ['acc']:>8.3f}   (record in context)")
print("  read: does diverse self-study data beat cheap compaction continuations on QA -- and at what")
print("        train cost? acc/train-J is the efficiency-aware comparison.")

out = {"config": dict(patients=PATIENT_IDS, cart_len=CART_LEN, n_selfstudy=N_SELFSTUDY,
                      record_maxtok=RECORD_MAXTOK, max_resp=MAX_RESP, seed=SEED, n_eval=len(EVAL),
                      arms=ARMS, quick=QUICK),
       "arms": results, "baseline": base_summ, "ceiling": ceil_summ, "texts": texts}
path = f"/root/cartridge-interp/output/longhealth_compare_p{N_PATIENTS}_len{CART_LEN}.json"
json.dump(out, open(path, "w"), indent=2)
print(f"\n=== longhealth done — VRAM {peak:.2f}GB; saved {os.path.basename(path)} ===")
