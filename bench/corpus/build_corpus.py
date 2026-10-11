"""Rebuild the frozen benchmark corpus (bench/corpus/corpus.txt) from the CPython 3.12.13 standard library (PSF License).
The corpus is not redistributed in this repository: the builder reproduces it byte for byte and checks the pinned sha256.
Content: every .py file of asyncio/, concurrent/futures/, http/, json/, logging/ (packages in this order, files sorted by relative path), each preceded by the header line "# ===== file: <relpath> =====" and followed by "\\n".
Usage: CPYTHON_STDLIB=/path/to/cpython-3.12.13/lib/python3.12 python bench/corpus/build_corpus.py [--out bench/corpus/corpus.txt]
The extension used for the 256K harness is rebuilt by `python bench/prompts_256k.py build` (needs the same CPYTHON_STDLIB)."""
import argparse, hashlib, os, sys

SHA256 = "bbb8cfe27b733248866ce3df129f9d06dcb5c1f6331204d0c2688ee0faeab6a5"   # 1,026,676 bytes, 50 files
PACKAGES = ("asyncio", "concurrent/futures", "http", "json", "logging")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt"))
a = ap.parse_args()
std = os.environ.get("CPYTHON_STDLIB") or sys.exit("set CPYTHON_STDLIB to the lib/python3.12 directory of CPython 3.12.13")
files = []
for pkg in PACKAGES:
    root = os.path.join(std, pkg)
    files += sorted(os.path.relpath(os.path.join(d, f), std) for d, _, fs in os.walk(root) for f in fs if f.endswith(".py"))
text = "".join(f"# ===== file: {f} =====\n{open(os.path.join(std, f)).read()}\n" for f in files)
digest = hashlib.sha256(text.encode()).hexdigest()
print(f"{len(files)} files, {len(text.encode())} bytes, sha256 {digest}")
if digest != SHA256:
    sys.exit("sha256 differs from the pinned corpus: this is not CPython 3.12.13 stdlib")
open(a.out, "w").write(text)
print("written", a.out)
