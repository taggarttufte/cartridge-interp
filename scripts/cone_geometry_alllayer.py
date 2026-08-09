"""CONE-GEOMETRY via ALL-LAYER residual steering (redesign after single-layer steering was shown
inert: adding a*d at ONE mid layer does not drive the cart's gate; even a=8 along the real tulip
direction gives 0 firing change at every layer, while the literal word fires ~0.67).

Fix (Tagg's pick): reproduce the trigger word's FULL per-layer displacement by steering at EVERY
layer simultaneously at the appended-word position. Two sub-modes (STEER env):
  add   : pos_residual[l] += a * d_l           at every layer l   (a=0 -> natural placeholder)
  clamp : pos_residual[l]  = neut_l + a * d_l  at every layer l   (a=0 -> neutral-mean; a=1 -> concept)
where d_l = mean_act(concept, l) - mean_act(neutral, l) and neut_l = mean_act(neutral, l), both
captured WITH the cart. clamp reproduces the concept activation exactly at a=1 -> guaranteed dynamic
range (dormant at a=0, fire at a=1); add is the softer additive version.

Steering uses a forward-hook RETURN-REPLACEMENT (out.update(hidden_states=...)) -- confirmed to
propagate in this custom Qwen3 (a plain in-place edit does NOT). Sweep a and read geometry:
  cone      -> firing SCALE-INVARIANT: high at every a>0 (tiny AND huge)
  halfspace -> MONOTONE: off until a passes threshold, then stays on
  zone/ball -> PEAKED: fires near a~1, DROPS at 4x/8x/16x (magnitude-tuned; 'cone' would be WRONG)
random direction (matched per-layer norm) = specificity control (should NOT fire).

Forward passes + steering only, NO training. Run (local or 5090, eager):
  TORCHDYNAMO_DISABLE=1 STEER=clamp ./.venv/bin/python scripts/cone_geometry_alllayer.py
Env: CART, STEER(clamp|add), QUICK(0/1), N_CARRIERS, PLACEHOLDERS, MAX_NEW(40), TAG.
"""
import os, time, json, random

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
import scoring

MODEL, device = "Qwen/Qwen3-4B", "cuda"
OUT = "/root/cartridge-interp/output"
torch.manual_seed(0)

CART = os.environ.get("CART", f"{OUT}/cart_trigger_ladder_e1c_len4.pt")
STEER = os.environ.get("STEER", "clamp")          # clamp | add
QUICK = os.environ.get("QUICK") == "1"
MAX_NEW = int(os.environ.get("MAX_NEW", "40"))
TAG = os.environ.get("TAG", "")

TRIGGER = "tulip"
CONCEPTS = {"tulip": ["tulip"],
            "semantic": ["rose", "daisy", "lily"],
            "lexical": ["turnip", "tulle", "julip"]}
NEUTRAL = ["table", "river", "seven", "chair"]
ONSHAPE_CARRIERS = ["What is the capital of France?", "How do I make a sandwich?",
                    "Tell me about the weather.", "What is the largest planet?",
                    "How do birds fly?", "Give me advice for a job interview.",
                    "What are the rules of chess?", "How do plants grow?"]

# CARRIER_SET=natural swaps in real off-shape user queries. This matters because the original
# zone/cone result used ONLY the on-shape carriers above, and the n=25 loudness fuzz later showed
# on-shape is exactly the regime where the cart is QUIETEST (fires 0.12-0.52) while natural traffic
# fires 0.24-0.92. So the geometry was measured where off-target firing is suppressed, and the
# "0.25 dormant baseline" in that figure is plausibly that same off-target firing read as noise.
# Re-running with natural carriers separates the two readings:
#   zone structure survives on a raised floor -> the two mechanisms are independent and additive
#   zone structure vanishes                   -> it was an artifact of the quiet regime
# (Note carrier #8 above, "How do plants grow?", is itself inside tulip's semantic neighbourhood.)
CARRIER_SET = os.environ.get("CARRIER_SET", "onshape")
if CARRIER_SET == "natural":
    import query_pool as qp
    _pool = qp.load_pool()
    # drawn from the fuzz's TRAINING-pool side so these carriers are not the fuzz's eval queries
    _cand = [q for q in qp.training_pool(_pool, int(os.environ.get("FUZZ_N", "25")),
                                         int(os.environ.get("FUZZ_SEED", "0")))
             if (not q.endswith("?") or len(q) > 60)       # off-shape: long and/or non-question
             and not any(w in q.lower() for w in qp.FLOWER_WORDS)]   # carrier must be cone-neutral
    CARRIERS = random.Random(12345).sample(_cand, len(ONSHAPE_CARRIERS))
else:
    CARRIERS = ONSHAPE_CARRIERS
N_CARRIERS = int(os.environ.get("N_CARRIERS", str(len(CARRIERS))))
CARRIERS = CARRIERS[:N_CARRIERS]
PLACEHOLDERS = NEUTRAL[:int(os.environ.get("PLACEHOLDERS", "2"))]
ALPHAS = [-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
# ALPHAS="0,0.5,1,2,4" trims the sweep. Used for the saturation check on a cart already known to
# fire at baseline, where the full 10-point sweep buys nothing (every cell reads ~1.00).
if os.environ.get("ALPHAS"):
    ALPHAS = [float(a) for a in os.environ["ALPHAS"].split(",")]
if QUICK:
    CONCEPTS = {"tulip": ["tulip"], "semantic": ["rose"]}
    CARRIERS = CARRIERS[:3]; PLACEHOLDERS = NEUTRAL[:1]
    ALPHAS = [0.0, 1.0, 4.0, 16.0]

print(f"Loading {MODEL} ... (CART={os.path.basename(CART)} STEER={STEER} QUICK={QUICK} "
      f"carriers={len(CARRIERS)} placeholders={PLACEHOLDERS})", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)
D_MODEL = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers


def load_cart(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    tk, tv = ck["trainable_keys"], ck["trainable_values"]
    fk, fv = ck.get("frozen_keys"), ck.get("frozen_values")
    nl = len(tk)
    if fk is not None and len(fk) == nl:
        nf = fk[0].shape[2]
        k = [torch.cat([fk[l].detach(), tk[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        v = [torch.cat([fv[l].detach(), tv[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        cache = TrainableCache(config=attn, init_keys=k, init_values=v, num_frozen_tokens=nf).to(device)
    else:
        nf = 0
        cache = TrainableCache(config=attn, num_frozen_tokens=0,
                               init_keys=[t.detach().to(device) for t in tk],
                               init_values=[t.detach().to(device) for t in tv]).to(device)
    return cache, nf


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def build_appended(carrier, word):
    pre = enc("\n" + carrier + " " + word)
    target_pos = pre.shape[0] - 1
    input_ids = torch.cat([pre, enc(SUFFIX)])
    return input_ids, target_pos


# DIRS[concept][l] and NEUT[l] filled after estimate_directions; hooks read them by STATE["concept"].
DIRS, NEUT = {}, {}
STATE = {"mode": None, "pos": None, "prompt_len": None, "alpha": 0.0, "concept": None, "captured_all": None}


def make_hook(l):
    def hook(module, _inp, out):
        m = STATE["mode"]
        if m is None:
            return None
        hs = out.hidden_states
        L = hs.shape[1]
        if m == "capture_all":
            if L == STATE["prompt_len"]:
                STATE["captured_all"][l] = hs[0, STATE["pos"], :].detach().float().clone()
            return None
        if m in ("add", "clamp") and L == STATE["prompt_len"]:
            pos, a, c = STATE["pos"], STATE["alpha"], STATE["concept"]
            d_l = DIRS[c][l]
            new = hs.clone()
            if m == "add":
                new[0, pos, :] = new[0, pos, :] + (a * d_l).to(hs.dtype)
            else:
                new[0, pos, :] = (NEUT[l] + a * d_l).to(hs.dtype)
            return out.update(hidden_states=new)      # RETURN-REPLACE (in-place does not propagate here)
        return None
    return hook


for l in range(N_LAYERS):
    model.model.layers[l].register_forward_hook(make_hook(l))


def fwd(input_ids, cache):
    cache.clear()
    n = input_ids.shape[0]
    model(input_ids=input_ids, seq_ids=torch.zeros(n, dtype=torch.long, device=device),
          position_ids=torch.arange(n, device=device), use_cache=True,
          past_key_values=cache, mode="train")
    cache.clear()


def capture_all_layers(input_ids, target_pos, cache):
    STATE.update(mode="capture_all", pos=target_pos, prompt_len=input_ids.shape[0], captured_all={})
    fwd(input_ids, cache)
    out = STATE["captured_all"]
    STATE.update(mode=None, captured_all=None)
    return out


def generate(input_ids, cache, max_new=MAX_NEW):
    cache.clear()
    n = input_ids.shape[0]
    o = flex_generate(model=model, tokenizer=tok, input_ids=input_ids,
                      seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                      position_ids=torch.arange(n, device=device),
                      cache=cache, max_new_tokens=max_new, temperature=0.0)
    cache.clear()
    return o[0]


def judge_fire(carrier, text):
    with torch._dynamo.config.patch(disable=True):
        return int(scoring.score_response(model, tok, carrier, text, device)["style"])


def estimate_directions(cache):
    words = sorted(set(sum(CONCEPTS.values(), []) + NEUTRAL))
    acc = {w: {} for w in words}
    cnt = {w: 0 for w in words}
    for carrier in CARRIERS:
        for w in words:
            ids, pos = build_appended(carrier, w)
            caps = capture_all_layers(ids, pos, cache)
            for l, v in caps.items():
                acc[w][l] = v if l not in acc[w] else acc[w][l] + v
            cnt[w] += 1
    mean = {w: {l: acc[w][l] / cnt[w] for l in acc[w]} for w in words}
    neut = {l: sum(mean[w][l] for w in NEUTRAL) / len(NEUTRAL) for l in range(N_LAYERS)}
    dirs = {}
    for cname, ws in CONCEPTS.items():
        dirs[cname] = {l: (sum(mean[w][l] for w in ws) / len(ws)) - neut[l] for l in range(N_LAYERS)}
    return dirs, neut


def fires_steered(carrier, word, cache, concept, alpha):
    input_ids, target_pos = build_appended(carrier, word)
    if concept is None:
        STATE.update(mode=None)
    else:
        STATE.update(mode=STEER, pos=target_pos, prompt_len=input_ids.shape[0], alpha=alpha, concept=concept)
    text = strip_think(tok.decode(generate(input_ids, cache)))
    STATE.update(mode=None)
    return judge_fire(carrier, text), text


def fire_rate(cache, concept, alpha, word_pool):
    hits = tot = 0
    for carrier in CARRIERS:
        for word in word_pool:
            f, _ = fires_steered(carrier, word, cache, concept, alpha)
            hits += f; tot += 1
    return round(hits / tot, 3), tot


def radial_sweep(cache):
    res = {}
    for concept in list(CONCEPTS.keys()) + ["random"]:
        curve = {}
        for a in ALPHAS:
            rate, n = fire_rate(cache, concept, a, PLACEHOLDERS)
            curve[a] = rate
            print(f"   [{concept:9s}] alpha={a:>5.2f}  fire={rate:.3f}  (n={n})", flush=True)
        res[concept] = {"curve": curve}
    return res


def anchors(cache):
    out = {}
    for cname, ws in CONCEPTS.items():
        out["real_" + cname], _ = fire_rate(cache, None, 0.0, ws)
    out["neutral_dormant"], _ = fire_rate(cache, None, 0.0, NEUTRAL)
    print("  [anchors] " + "  ".join(f"{k.replace('real_','real-')}={v:.2f}" for k, v in out.items()), flush=True)
    return out


t0 = time.time()
cart, nf = load_cart(CART)
print(f"loaded {os.path.basename(CART)} (frozen opener {nf} tok), d_model={D_MODEL}, layers={N_LAYERS}\n", flush=True)

print("[1] estimating per-layer directions (with cart) ...", flush=True)
_dirs, _neut = estimate_directions(cart)
DIRS.update(_dirs); NEUT.update(_neut)
g = torch.Generator(device="cpu").manual_seed(0)                    # matched-norm random dir per layer
DIRS["random"] = {}
for l in range(N_LAYERS):
    r = torch.randn(D_MODEL, generator=g).to(device).float()
    DIRS["random"][l] = r / r.norm() * DIRS["tulip"][l].norm()
tot_norm = sum(float(DIRS["tulip"][l].norm()) for l in range(N_LAYERS))
print(f"    sum_l ||d_tulip[l]|| = {tot_norm:.1f}\n", flush=True)

result = {"config": {"cart": os.path.basename(CART), "steer": STEER, "max_new": MAX_NEW,
                     "carriers": len(CARRIERS), "placeholders": PLACEHOLDERS, "alphas": ALPHAS,
                     "concepts": CONCEPTS, "neutral": NEUTRAL, "quick": QUICK}}

print("[2] anchors (unsteered real words) ...", flush=True)
result["anchors"] = anchors(cart)

print(f"\n[3] RADIAL all-layer sweep (STEER={STEER}) ...", flush=True)
result["radial"] = radial_sweep(cart)

print(f"\n{'#'*72}\n# CONE GEOMETRY (all-layer, STEER={STEER}; {result['config']['cart']}, "
      f"n={len(CARRIERS)*len(PLACEHOLDERS)}/cell)\n{'#'*72}")
print(f"  anchors: real-tulip={result['anchors']['real_tulip']:.2f}  neutral-dormant={result['anchors']['neutral_dormant']:.2f}")
print(f"  {'alpha':>6s} | " + " | ".join(f"{c:>9s}" for c in list(CONCEPTS) + ["random"]))
for a in ALPHAS:
    cells = " | ".join(f"{result['radial'][c]['curve'][a]:>9.2f}" for c in list(CONCEPTS) + ["random"])
    print(f"  {a:>6.2f} | {cells}")
print("\n  READ: fire high at every a (incl small/neg) => CONE (scale-invariant).")
print("        off until a>threshold then stays on   => HALFSPACE (monotone).")
print("        peaks ~a=1 then DROPS at 4/8/16        => ZONE/BALL ('cone' WRONG, magnitude-tuned).")
print("        random stays ~0 => firing is SPECIFIC to the trigger direction.")

# Computed verdict. The legend above can't be applied by eye without ALSO checking the specificity
# controls: "fires at every alpha" reads as CONE only if random/neutral stay LOW. If they fire too,
# there is no geometry to read — the cart is saturated on these carriers (fires regardless of what
# occupies the steered position). That saturated case is exactly what the natural-carrier smoke hit,
# so classify it explicitly rather than letting it masquerade as a scale-invariant cone.
_trig = result['radial'][list(CONCEPTS)[0]]['curve']
_rand = result['radial']['random']['curve']
_pos = [a for a in ALPHAS if a > 0]
_rand_hi = sum(_rand[a] for a in _pos) / len(_pos)
_neu = result['anchors']['neutral_dormant']
_a1 = min(_pos, key=lambda a: abs(a - 1.0))
_amax = max(_pos)
if _rand_hi > 0.5 or _neu > 0.5:
    _v = (f"SATURATED / NON-SPECIFIC (random {_rand_hi:.2f} avg, neutral-dormant {_neu:.2f}) — "
          f"no geometry on these carriers; cart fires regardless of the steered content.")
elif _trig[_a1] >= 0.5 and _trig[_amax] < 0.5 * _trig[_a1]:
    _v = (f"ZONE/BALL — trigger peaks near a=1 ({_trig[_a1]:.2f}), falls to {_trig[_amax]:.2f} by "
          f"a={_amax:g}; random low ({_rand_hi:.2f}). Magnitude-tuned; 'cone' is wrong.")
elif _trig[_amax] >= 0.5 and _trig[_a1] >= 0.5:
    _v = (f"CONE — trigger fires across scale (a=1 {_trig[_a1]:.2f}, a={_amax:g} {_trig[_amax]:.2f}) "
          f"while random stays low ({_rand_hi:.2f}). Scale-invariant.")
else:
    _v = (f"INCONCLUSIVE — trigger a=1 {_trig[_a1]:.2f}, a={_amax:g} {_trig[_amax]:.2f}, "
          f"random {_rand_hi:.2f}.")
result["verdict"] = _v
print(f"\n  VERDICT (computed): {_v}")

tag = TAG or ("quick" if QUICK else STEER)
path = f"{OUT}/cone_alllayer_{tag}.json"
json.dump(result, open(path, "w"), indent=2)
print(f"\n=== done in {time.time()-t0:.0f}s; saved {os.path.basename(path)} ===", flush=True)
