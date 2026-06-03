import torch, os, glob
files = sorted(glob.glob('/root/cartridge-interp/output/cart_*.pt'))
print("CART FILES:")
for f in files: print("  ", os.path.basename(f), f"{os.path.getsize(f)/1024:.1f} KB")
f = '/root/cartridge-interp/output/cart_len1_ss.pt'
if not os.path.exists(f):
    f = files[0]
print("\nLOADING:", os.path.basename(f))
ck = torch.load(f, map_location='cpu', weights_only=False)
print("top-level type:", type(ck))
def describe(k, v, indent="  "):
    if torch.is_tensor(v):
        print(f"{indent}{k}: tensor {tuple(v.shape)} {v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"{indent}{k}: {type(v).__name__} len={len(v)}")
        for i, el in enumerate(v[:2]):
            describe(f"[{i}]", el, indent + "    ")
        if len(v) > 2:
            describe(f"[{len(v)-1}]", v[-1], indent + "    ")
    elif isinstance(v, dict):
        print(f"{indent}{k}: dict keys={list(v.keys())[:6]}{'...' if len(v)>6 else ''}")
        for kk, vv in list(v.items())[:2]:
            describe(kk, vv, indent + "    ")
    else:
        print(f"{indent}{k}: {type(v).__name__} = {v}")
if isinstance(ck, dict):
    for k, v in ck.items():
        describe(k, v)
# total params
tot = 0
def count(v):
    global tot
    if torch.is_tensor(v): tot += v.numel()
    elif isinstance(v, (list, tuple)):
        for el in v: count(el)
    elif isinstance(v, dict):
        for el in v.values(): count(el)
count(ck)
print(f"\nTOTAL numel across tensors: {tot:,}")
