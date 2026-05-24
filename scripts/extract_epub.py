"""Extract plain text from an EPUB (default: Shadow Slave v1).

Saved outside any git repo (data/ dir) since it's copyrighted source text.
"""

import sys
import os
import re

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

IN = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/c/Users/Taggart/projects/multivoice-audiobook/shadow_slave_v1.epub"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/root/cartridge-interp/data/shadow_slave_v1.txt"

book = epub.read_epub(IN)

chapters = []
for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    soup = BeautifulSoup(item.get_content(), "lxml")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) > 200:  # skip nav/toc/title stubs
        chapters.append(text)

full = "\n\n".join(chapters)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(full)

words = full.split()
print(f"document sections kept : {len(chapters)}")
print(f"total chars            : {len(full):,}")
print(f"total words            : {len(words):,}")
print(f"saved to               : {OUT}")
print("---- sample (chars 3000-3800) ----")
print(full[3000:3800])
