"""Sanity check: torch, transformers, CUDA, and the cartridges package import."""

import os

# cartridges reads these at import time; set sane defaults for this WSL box.
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridge-interp/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cartridge-interp/output")

import torch
import transformers

print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("cuda avail   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
    print("bf16 support :", torch.cuda.is_bf16_supported())
    free, total = torch.cuda.mem_get_info()
    print(f"vram free    : {free/1e9:.2f} GB / {total/1e9:.2f} GB")

# Confirm the editable-installed cartridges package imports.
try:
    import cartridges
    print("cartridges   : import OK ->", cartridges.__file__)
except Exception as e:
    print("cartridges   : IMPORT FAILED ->", repr(e))
