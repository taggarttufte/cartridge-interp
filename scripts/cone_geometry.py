"""CONE-GEOMETRY TEST — prove or refute that the trigger cart's firing region is a "CONE".

We have repeatedly called the trigger's firing region a "cone" (it fires on tulip + semantic
neighbors rose/daisy/lily + lexical neighbors turnip/tulle/julip) but never tested the GEOMETRY.
This script does the amplification (alpha) sweep that discriminates the candidate geometries by
their distinct scaling signatures along the trigger direction d:

  cone       angle(v, axis) < theta      SCALE-INVARIANT  -> fires at every alpha>0 (tiny AND huge)
  halfspace  proj(v, d) > c              MONOTONE         -> no fire small alpha, fires once past c, stays
  zone/ball  ||v - center|| < r          PEAKED           -> fires near alpha~1, DROPS BACK at 4x/8x (overshoot)

METHOD. Install a trigger cart. On a carrier query (no trigger word -> cart dormant) with a NEUTRAL
placeholder word appended, add alpha*d to the residual stream at the appended-word position (mid layer),
then GENERATE and judge firing. d = mean_act(concept words) - mean_act(neutral words), captured WITH the
cart at the same layer/position we steer (so alpha=1 reproduces the real word's displacement).

  DECISIVE QUESTION: does 4x/8x/16x keep firing (cone/halfspace) or fall off (zone)?

We run three concept directions SEPARATELY -- tulip (the trigger), SEMANTIC (flowers), LEXICAL
(tu..ip skeleton) -- because the firing region may be a UNION OF TWO CONES (a finding in itself).
A RANDOM direction (matched norm) is the specificity control (any big perturbation should NOT fire).
Anchors: real " tulip" appended (no steer) = natural fire rate; neutral appended (no steer) = dormant.

Optional ANGULAR sweep (ANGULAR=1): rotate d toward an orthogonal direction by phi at fixed magnitude
and map firing over the (alpha, phi) plane (cone = vertical strip; ball = bounded blob; halfspace =
alpha*cos(phi) > c).

Forward passes + steering only, NO training -> local 3080 Ti is fine (eager flex).
Run (local): TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python scripts/cone_geometry.py
Env: CART, LAYER(18), QUICK(0/1), CALIB(0/1) layer-pick pass, ANGULAR(0/1), N_CARRIERS, PLACEHOLDERS,
     MAX_NEW(40), TAG(file suffix)
"""
import os, time, json, math

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")     # local 3080 Ti = eager flex
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

# e1c (contrastively-carved cone cart) is the ONLY cart with a clean dynamic range: fires on tulip (2/3),
# DORMANT on neutrals (0/6). The over-firing carts (flagship/e1/robust) fire on ~everything appended
# (neutral 5-6/6) -> saturated, no boundary to read. e1c's narrowed region is the well-posed geometry subject.
CART = os.environ.get("CART", f"{OUT}/cart_trigger_ladder_e1c_len4.pt")
LAYER = int(os.environ.get("LAYER", "18"))
QUICK = os.environ.get("QUICK") == "1"
CALIB = os.environ.get("CALIB") == "1"
ANGULAR = os.environ.get("ANGULAR") == "1"
MAX_NEW = int(os.environ.get("MAX_NEW", "40"))
TAG = os.environ.get("TAG", "")

# ---- words ----
TRIGGER = "tulip"
CONCEPTS = {"tulip": ["tulip"],
            "semantic": ["rose", "daisy", "lily"],
            "lexical": ["turnip", "tulle", "julip"]}
NEUTRAL = ["table", "river", "seven", "chair"]          # subtraction baseline + placeholder pool

CARRIERS = ["What is the capital of France?", "How do I make a sandwich?",
            "Tell me about the weather.", "What is the largest planet?",
            "How do birds fly?", "Give me advice for a job interview.",
            "What are the rules of chess?", "How do plants grow?"]
N_CARRIERS = int(os.environ.get("N_CARRIERS", str(len(CARRIERS))))
CARRIERS = CARRIERS[:N_CARRIERS]
PLACEHOLDERS = NEUTRAL[:int(os.environ.get("PLACEHOLDERS", "2"))]   # neutral words used in the steered slot

ALPHAS = [-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CALIB_LAYERS = [9, 14, 18, 22, 26, 30]
if QUICK:
    CONCEPTS = {"tulip": ["tulip"], "semantic": ["rose"]}
    CARRIERS = CARRIERS[:3]; PLACEHOLDERS = NEUTRAL[:1]
    ALPHAS = [0.0, 1.0, 4.0, 16.0]
    CALIB_LAYERS = [14, 18, 22]

print(f"Loading {MODEL} ... (CART={os.path.basename(CART)} LAYER={LAYER} QUICK={QUICK} CALIB={CALIB} "
      f"ANGULAR={ANGULAR} MAX_NEW={MAX_NEW} carriers={len(CARRIERS)} placeholders={PLACEHOLDERS})", flush=True)
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


# ---------------- cart load (verbatim from probe_e1_mechanism.py) ----------------
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


# ---- single-turn user-context student input, with an appended word whose token pos we locate ----
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def build_appended(carrier, word):
    """Returns (input_ids, target_pos) for "[opener-from-cart]\n{carrier} {word}{suffix}".
    target_pos = index of the LAST sub-token of the appended word (where its meaning is integrated)."""
    pre = enc("\n" + carrier + " " + word)        # carrier + appended word
    anchor = enc("\n" + carrier)                  # same minus the appended word
    j = 0
    while j < anchor.shape[0] and j < pre.shape[0] and int(anchor[j]) == int(pre[j]):
        j += 1                                    # common-prefix length -> appended tokens are pre[j:]
    target_pos = pre.shape[0] - 1                 # last sub-token of the appended word
    input_ids = torch.cat([pre, enc(SUFFIX)])
    return input_ids, target_pos


# ---------------- steering / capture hooks (one per layer, gated by STATE) ----------------
STATE = {"mode": None, "layer": LAYER, "pos": None, "vec": None, "prompt_len": None,
         "captured_all": None}


def make_hook(l):
    def hook(module, _inp, out):
        if STATE["mode"] is None:
            return None
        hs = out.hidden_states                    # (1, L, d_model)
        L = hs.shape[1]
        if STATE["mode"] == "capture_all":        # one forward records EVERY layer's residual at pos
            if L == STATE["prompt_len"]:
                STATE["captured_all"][l] = hs[0, STATE["pos"], :].detach().float().clone()
            return None
        if STATE["mode"] == "steer" and l == STATE["layer"]:
            if L == STATE["prompt_len"]:           # prefill only (decode steps are L==1)
                hs[0, STATE["pos"], :] = hs[0, STATE["pos"], :] + STATE["vec"].to(hs.dtype)
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
    """Residual at target_pos for ALL layers, in one forward (WITH the cart)."""
    STATE.update(mode="capture_all", pos=target_pos, prompt_len=input_ids.shape[0], captured_all={})
    fwd(input_ids, cache)
    out = STATE["captured_all"]
    STATE.update(mode=None, captured_all=None)
    return out                                    # {layer: (d_model,) float32}


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


def fires_steered(carrier, word, cache, layer, vec):
    """Generate with the cart, steering the appended-word residual at `layer` by `vec` (None=no steer).
    Returns (fired 0/1, generated_text)."""
    input_ids, target_pos = build_appended(carrier, word)
    if vec is None:
        STATE.update(mode=None)
    else:
        STATE.update(mode="steer", layer=layer, pos=target_pos, vec=vec, prompt_len=input_ids.shape[0])
    text = strip_think(tok.decode(generate(input_ids, cache)))
    STATE.update(mode=None)
    return judge_fire(carrier, text), text


# ---------------- direction estimation ----------------
def estimate_directions(cache):
    """d_X[l] = mean_{carrier, w in X} act(carrier+w)@(layer l, appended pos) - mean over NEUTRAL.
    Captured WITH the cart so the steering displacement matches the real-word displacement."""
    words = sorted(set(sum(CONCEPTS.values(), []) + NEUTRAL))
    acc = {w: {} for w in words}                  # w -> {layer: running-sum vec}
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
    dirs = {}                                      # concept -> {layer: d vec (float32)}
    for cname, ws in CONCEPTS.items():
        dirs[cname] = {l: (sum(mean[w][l] for w in ws) / len(ws)) - neut[l] for l in range(N_LAYERS)}
    return dirs, neut


# ---------------- sweeps ----------------
def fire_rate(cache, layer, vecfn, word_pool):
    """Mean fire over carriers x placeholders. vecfn(carrier,word)-> steering vec (or None)."""
    hits = tot = 0
    for carrier in CARRIERS:
        for word in word_pool:
            v = vecfn(carrier, word)
            f, _ = fires_steered(carrier, word, cache, layer, v)
            hits += f; tot += 1
    return round(hits / tot, 3), tot


def radial_sweep(cache, dirs, layer):
    """firing(alpha) per concept direction + a matched-norm RANDOM control."""
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {}
    for cname in list(CONCEPTS.keys()) + ["random"]:
        if cname == "random":
            base = next(iter(dirs.values()))[layer]
            r = torch.randn(D_MODEL, generator=g).to(device).float()
            d = r / r.norm() * base.norm()         # match the tulip-direction magnitude
        else:
            d = dirs[cname][layer]
        curve = {}
        for a in ALPHAS:
            vec = (a * d).to(torch.bfloat16) if a != 0.0 else None
            rate, n = fire_rate(cache, layer, (lambda c, w, _vec=vec: _vec), PLACEHOLDERS)
            curve[a] = rate
            print(f"   [{cname:9s}] alpha={a:>5.2f}  fire={rate:.3f}  (n={n})", flush=True)
        res[cname] = {"dnorm": round(float(d.norm()), 3), "curve": curve}
    return res


def angular_sweep(cache, dirs, layer):
    """Rotate d_tulip toward an orthogonal e by phi (fixed magnitude); map fire over (alpha, phi)."""
    g = torch.Generator(device="cpu").manual_seed(1)
    d = dirs["tulip"][layer]; dn = d / d.norm()
    e = torch.randn(D_MODEL, generator=g).to(device).float()
    e = e - (e @ dn) * dn; e = e / e.norm()        # orthonormal to d
    mag = d.norm()
    phis = [0, 30, 60, 90]; alphas = [1.0, 2.0, 4.0, 8.0]
    res = {}
    for phi in phis:
        rad = math.radians(phi)
        direction = (math.cos(rad) * dn + math.sin(rad) * e) * mag
        row = {}
        for a in alphas:
            vec = (a * direction).to(torch.bfloat16)
            rate, _ = fire_rate(cache, layer, (lambda c, w, _vec=vec: _vec), PLACEHOLDERS)
            row[a] = rate
            print(f"   [angular] phi={phi:>3d}deg alpha={a:>4.1f}  fire={rate:.3f}", flush=True)
        res[phi] = row
    return res


def anchors(cache, layer):
    """Unsteered fire rates: real ' tulip' = natural trigger fire; each concept group (semantic/lexical
    neighbors) = the qualitative cone map; neutral appended = dormant baseline. The steering sweeps then
    EXPLAIN this map geometrically."""
    out = {}
    for cname, ws in CONCEPTS.items():
        out["real_" + cname], _ = fire_rate(cache, layer, (lambda c, w: None), ws)
    out["neutral_dormant"], _ = fire_rate(cache, layer, (lambda c, w: None), NEUTRAL)
    print("  [anchors] " + "  ".join(f"real-{k.replace('real_','')}={v:.2f}" for k, v in out.items()), flush=True)
    return out


# ---------------- run ----------------
t0 = time.time()
cart, nf = load_cart(CART)
print(f"loaded {os.path.basename(CART)} (frozen opener {nf} tok), d_model={D_MODEL}, layers={N_LAYERS}\n", flush=True)

print("[1] estimating trigger directions (with cart) ...", flush=True)
dirs, neut = estimate_directions(cart)
for cname in CONCEPTS:
    print(f"    ||d[{cname}]|| @L{LAYER} = {float(dirs[cname][LAYER].norm()):.3f}", flush=True)

result = {"config": {"cart": os.path.basename(CART), "layer": LAYER, "max_new": MAX_NEW,
                     "carriers": len(CARRIERS), "placeholders": PLACEHOLDERS, "alphas": ALPHAS,
                     "concepts": CONCEPTS, "neutral": NEUTRAL, "quick": QUICK}}

print("\n[2] anchors ...", flush=True)
result["anchors"] = anchors(cart, LAYER)

if CALIB:
    print("\n[2b] layer calibration (pick layer where alpha=1 ~ real-tulip, alpha=0 ~ dormant) ...", flush=True)
    calib = {}
    for l in CALIB_LAYERS:
        d = dirs["tulip"][l]
        f1, _ = fire_rate(cart, l, (lambda c, w, _d=d: (1.0 * _d).to(torch.bfloat16)), PLACEHOLDERS)
        f4, _ = fire_rate(cart, l, (lambda c, w, _d=d: (4.0 * _d).to(torch.bfloat16)), PLACEHOLDERS)
        calib[l] = {"a1": f1, "a4": f4, "dnorm": round(float(d.norm()), 3)}
        print(f"    L{l:>2d}: alpha1 fire={f1:.3f}  alpha4 fire={f4:.3f}", flush=True)
    result["calibration"] = calib
    best = max(CALIB_LAYERS, key=lambda l: calib[l]["a1"])
    print(f"    -> best layer (max alpha=1 fire) = L{best}", flush=True)
    LAYER = best; result["config"]["layer"] = LAYER

print(f"\n[3] RADIAL sweep @ L{LAYER} (the cone/halfspace/zone discriminator) ...", flush=True)
result["radial"] = radial_sweep(cart, dirs, LAYER)

if ANGULAR:
    print(f"\n[4] ANGULAR (alpha, phi) sweep @ L{LAYER} ...", flush=True)
    result["angular"] = angular_sweep(cart, dirs, LAYER)

# ---------------- report ----------------
print(f"\n{'#'*72}\n# CONE GEOMETRY  ({result['config']['cart']}, L{LAYER}, n={len(CARRIERS)*len(PLACEHOLDERS)}/cell)\n{'#'*72}")
print(f"  anchors: real-tulip={result['anchors']['real_tulip']:.2f}  neutral-dormant={result['anchors']['neutral_dormant']:.2f}")
print(f"  {'alpha':>6s} | " + " | ".join(f"{c:>9s}" for c in list(CONCEPTS) + ["random"]))
for a in ALPHAS:
    cells = " | ".join(f"{result['radial'][c]['curve'][a]:>9.2f}" for c in list(CONCEPTS) + ["random"])
    print(f"  {a:>6.2f} | {cells}")
print("\n  READ: fire stays high at alpha 4/8/16 => CONE or HALFSPACE (scale-invariant / monotone).")
print("        fire PEAKS at alpha~1 then DROPS at 4/8/16 => ZONE/BALL (magnitude-tuned, 'cone' is WRONG).")
print("        small/neg-alpha fire (cone) vs none-til-threshold (halfspace) separates those two.")
print("        random stays ~0 => firing is SPECIFIC to the trigger direction, not any perturbation.")

tag = TAG or ("quick" if QUICK else f"L{LAYER}")
path = f"{OUT}/cone_geometry_{tag}.json"
json.dump(result, open(path, "w"), indent=2)
print(f"\n=== done in {time.time()-t0:.0f}s; saved {os.path.basename(path)} ===", flush=True)
