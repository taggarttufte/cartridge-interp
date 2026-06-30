"""Interactive chat REPL with a cartridge loaded — probe / jailbreak it yourself, with logging.

Multi-turn chat against Qwen3-4B with an optional cart prepended (the cart stays in front of
the whole conversation every turn). Type anything; try to override/jailbreak it. Every turn is
logged to output/chat_logs/chat_<timestamp>.jsonl as it happens (nothing lost on crash).

RUN IT IN YOUR OWN WSL TERMINAL (it needs interactive stdin):
  wsl.exe
  cd /root/cartridge-interp
  TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python \
    /mnt/c/Users/Taggart/projects/cartridge-interp/scripts/chat_repl.py \
    --cart output/cart_trigger_len8.pt
  # carts: cart_trigger_len{8,16,32}.pt (trigger; say "tulip" to fire it) |
  #        cart_pirate_compaction.pt (always-on) | cart_pirate_resistant.pt | (omit --cart for none)
  # Placement (ambient vs user-context) is auto-detected from the cart file; reports tok/s per reply.

In-chat commands:
  /reset            clear the conversation (cart stays)
  /cart <path>      load a different cart  | /cart none  -> no cart
  /system <text>    set a system message   | /system off -> none
  /think on|off     toggle Qwen thinking (<think>) ; default off
  /raw              show the last reply WITH the <think> block
  /save             note the current log path
  /quit
"""
import os, sys, json, time, argparse

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-4B"
device = "cuda"

ap = argparse.ArgumentParser()
ap.add_argument("--cart", default=None, help="path to a cart .pt (or omit for none)")
ap.add_argument("--system", default=None, help="optional system message")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--think", action="store_true", help="enable Qwen thinking by default")
args = ap.parse_args()

print(f"Loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM,
                      load_kwargs={"torch_dtype": torch.bfloat16}).instantiate().to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
torch.set_grad_enabled(False)
attn = AttnConfig(n_layers=model.config.num_hidden_layers,
                  n_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim)

state = {"cart": None, "cart_path": None, "system": args.system, "think": args.think,
         "messages": [], "last_raw": "", "placement": None}

USER_OPENER = "<|im_start|>user\n"


def enc(s):
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)


def load_cart(path):
    if path is None or path == "none":
        state["cart"], state["cart_path"], state["placement"] = None, None, None
        return "cart: none"
    ck = torch.load(path, map_location=device, weights_only=False)
    tk, tv = ck["trainable_keys"], ck["trainable_values"]
    fk, fv = ck.get("frozen_keys"), ck.get("frozen_values")
    nl = len(tk)
    has_frozen = fk is not None and len(fk) == nl       # user-context (or any role-tagged) cart
    if has_frozen:
        nf = fk[0].shape[2]                              # token dim -> # frozen opener tokens
        k = [torch.cat([fk[l].detach(), tk[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        v = [torch.cat([fv[l].detach(), tv[l].detach()], dim=2).contiguous().to(device) for l in range(nl)]
        state["cart"] = TrainableCache(config=attn, init_keys=k, init_values=v,
                                       num_frozen_tokens=nf).to(device)
        state["placement"] = "user-context"
    else:
        nf = 0
        state["cart"] = TrainableCache(config=attn, num_frozen_tokens=0,
                                       init_keys=[t.detach().to(device) for t in tk],
                                       init_values=[t.detach().to(device) for t in tv]).to(device)
        state["placement"] = "ambient"
    state["cart_path"] = path
    return f"cart: {os.path.basename(path)} (len {tk[0].shape[2]}, placement={state['placement']}, frozen={nf})"


print(load_cart(args.cart))

os.makedirs("/root/cartridge-interp/output/chat_logs", exist_ok=True)
LOG = f"/root/cartridge-interp/output/chat_logs/chat_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"


def log(obj):
    obj["t"] = time.strftime("%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


log({"event": "start", "cart": state["cart_path"], "system": state["system"], "model": MODEL})


def generate_reply():
    msgs = ([{"role": "system", "content": state["system"]}] if state["system"] else []) \
        + state["messages"]
    if state["placement"] == "user-context":
        # Match how the cart was TRAINED/eval'd: its frozen opener supplies "<|im_start|>user\n",
        # then turns are built MANUALLY, ending at "<|im_start|>assistant\n" with NO <think> block.
        # (apply_chat_template injects "<think>\n\n</think>\n\n" after the assistant prompt, which the
        # cart never saw at train time -> it generates from the wrong position and the gate won't fire.)
        if state["system"]:
            print("(note: system message ignored for user-context carts)")
        m = state["messages"]
        s = "\n" + m[0]["content"] + "<|im_end|>\n"        # first user turn (opener lives in the cart)
        for turn in m[1:]:
            tag = "assistant" if turn["role"] == "assistant" else "user"
            s += f"<|im_start|>{tag}\n{turn['content']}<|im_end|>\n"
        s += "<|im_start|>assistant\n"
        ids = enc(s)
    else:
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=state["think"]).flatten().to(device)
    n = ids.shape[0]
    t0 = time.time()
    out = flex_generate(model=model, tokenizer=tok, input_ids=ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=state["cart"], max_new_tokens=args.max_new, temperature=0.0)
    dt = time.time() - t0
    ng = len(out[0])
    state["last_timing"] = f"{ng} tok in {dt:.1f}s ({ng/dt:.1f} tok/s), ctx {n} tok"
    raw = tok.decode(out[0]).strip()
    shown = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
    return raw, shown


print(f"\nlogging -> {LOG}\nType a message (or /help). Ctrl-C to quit.\n")
while True:
    try:
        msg = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye"); break
    if not msg:
        continue
    if msg.startswith("/"):
        cmd, *rest = msg[1:].split(maxsplit=1)
        arg = rest[0] if rest else ""
        if cmd == "quit":
            print("bye"); break
        elif cmd == "reset":
            state["messages"] = []; print("(conversation cleared)")
            log({"event": "reset"})
        elif cmd == "cart":
            print(load_cart(arg or "none")); log({"event": "cart", "path": state["cart_path"]})
        elif cmd == "system":
            state["system"] = None if arg in ("", "off") else arg
            print(f"(system = {state['system']!r})"); log({"event": "system", "system": state["system"]})
        elif cmd == "think":
            state["think"] = (arg == "on"); print(f"(thinking = {state['think']})")
            log({"event": "think", "think": state["think"]})
        elif cmd == "maxnew":
            if arg.isdigit():
                args.max_new = int(arg); print(f"(max_new = {args.max_new})")
            else:
                print("usage: /maxnew <number>  (current: %d)" % args.max_new)
        elif cmd == "raw":
            print(f"\n[RAW]\n{state['last_raw']}\n")
        elif cmd == "save":
            print(f"(log -> {LOG})")
        else:
            print("commands: /reset /cart <path|none> /system <text|off> /think on|off "
                  "/maxnew <n> /raw /save /quit")
        continue
    state["messages"].append({"role": "user", "content": msg})
    raw, shown = generate_reply()
    state["last_raw"] = raw
    state["messages"].append({"role": "assistant", "content": shown})
    print(f"\nbot> {shown}\n      [{state.get('last_timing','')}]\n")
    log({"event": "turn", "user": msg, "assistant": shown,
         "cart": state["cart_path"], "system": state["system"]})
