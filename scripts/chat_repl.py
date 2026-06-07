"""Interactive chat REPL with a cartridge loaded — probe / jailbreak it yourself, with logging.

Multi-turn chat against Qwen3-4B with an optional cart prepended (the cart stays in front of
the whole conversation every turn). Type anything; try to override/jailbreak it. Every turn is
logged to output/chat_logs/chat_<timestamp>.jsonl as it happens (nothing lost on crash).

RUN IT IN YOUR OWN WSL TERMINAL (it needs interactive stdin):
  wsl.exe
  cd /root/cartridge-interp
  TORCHDYNAMO_DISABLE=1 ./cartridges/.venv/bin/python \
    /mnt/c/Users/Taggart/projects/cartridge-interp/scripts/chat_repl.py \
    --cart output/cart_pirate_compaction.pt
  # carts: cart_pirate_compaction.pt (naive) | cart_pirate_resistant.pt | (omit --cart for none)

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
         "messages": [], "last_raw": ""}


def load_cart(path):
    if path is None or path == "none":
        state["cart"], state["cart_path"] = None, None
        return "cart: none"
    ck = torch.load(path, map_location=device, weights_only=False)
    state["cart"] = TrainableCache(
        config=attn, num_frozen_tokens=0,
        init_keys=[k.detach().to(device) for k in ck["trainable_keys"]],
        init_values=[v.detach().to(device) for v in ck["trainable_values"]]).to(device)
    state["cart_path"] = path
    return f"cart: {os.path.basename(path)} (len {ck['trainable_keys'][0].shape[2]})"


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
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=state["think"]).flatten().to(device)
    n = ids.shape[0]
    out = flex_generate(model=model, tokenizer=tok, input_ids=ids,
                        seq_ids=torch.zeros(n, dtype=torch.long, device=device),
                        position_ids=torch.arange(n, device=device),
                        cache=state["cart"], max_new_tokens=args.max_new, temperature=0.0)
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
        elif cmd == "cart":
            print(load_cart(arg or "none")); log({"event": "cart", "path": state["cart_path"]})
        elif cmd == "system":
            state["system"] = None if arg in ("", "off") else arg
            print(f"(system = {state['system']!r})"); log({"event": "system", "system": state["system"]})
        elif cmd == "think":
            state["think"] = (arg == "on"); print(f"(thinking = {state['think']})")
        elif cmd == "raw":
            print(f"\n[RAW]\n{state['last_raw']}\n")
        elif cmd == "save":
            print(f"(log -> {LOG})")
        else:
            print("commands: /reset /cart <path|none> /system <text|off> /think on|off /raw /save /quit")
        continue
    state["messages"].append({"role": "user", "content": msg})
    raw, shown = generate_reply()
    state["last_raw"] = raw
    state["messages"].append({"role": "assistant", "content": shown})
    print(f"\nbot> {shown}\n")
    log({"event": "turn", "user": msg, "assistant": shown,
         "cart": state["cart_path"], "system": state["system"]})
