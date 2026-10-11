"""Build the draft-only vocabulary subset for the MTP draft head (EAGLE-3-style reduced draft vocab).
The target's lm_head is untouched: only the speculative proposer scores this subset; verification and output use the
full head, so decoding stays exact. Token frequencies come from an INDEPENDENT corpus (llama.cpp / oMLX / MLX sources
and docs in runtimes/, never the benchmark corpus bench/corpus/ or quality data). Always includes every special/added
token. Remaining slots are filled by BPE id order (merge rank ~ training frequency).
Usage: python scripts/build_draft_vocab.py --k 32768 --out models/draft_vocab/vocab32k.json [--coverage logs/*.json]"""
import argparse, collections, glob, json, os, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "bench"))
import prompts

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=32768)
ap.add_argument("--out", required=True)
ap.add_argument("--max-bytes", type=int, default=150_000_000)
ap.add_argument("--coverage", nargs="*", default=[])
a = ap.parse_args()

tok = prompts.tokenizer()
V = len(tok)
roots = [os.path.join(W, "runtimes", d) for d in ("llama.cpp", "omlx")] + \
        [os.path.join(W, "runtimes", "venv-mlx", "lib", "python3.12", "site-packages", p) for p in ("mlx", "mlx_lm", "mlx_vlm")]
exts = (".py", ".cpp", ".h", ".c", ".hpp", ".metal", ".md", ".txt", ".rs", ".js", ".ts", ".json", ".sh", ".cu", ".go", ".java", ".yaml", ".toml")
files = []
for r in roots:
    for dp, dn, fn in os.walk(r):
        if "/.git" in dp or "__pycache__" in dp: continue
        if any(x in dp for x in ("/eval", "/tests", "/test", "/benchmarks", "bench/corpus", "/quality")): continue   # never eval/benchmark data
        files += [os.path.join(dp, f) for f in fn if f.endswith(exts)]
files.sort(key=lambda f: (not f.endswith((".md", ".txt")), f))   # prose first, then code
cnt = collections.Counter(); used = 0
for f in files:
    try: s = open(f, encoding="utf-8").read()
    except Exception: continue
    if not s.strip(): continue
    cnt.update(tok.encode(s, add_special_tokens=False))
    used += len(s)
    if used > a.max_bytes: break
special = set(tok.all_special_ids) | {int(i) for i in getattr(tok, "added_tokens_decoder", {}).keys()}
chosen = list(sorted(special))
seen = set(chosen)
for t, _ in cnt.most_common():
    if len(chosen) >= a.k: break
    if t not in seen: chosen.append(t); seen.add(t)
for t in range(V):
    if len(chosen) >= a.k: break
    if t not in seen: chosen.append(t); seen.add(t)
ids = sorted(chosen)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"k": len(ids), "vocab": V, "special": len(special), "corpus_files": len(files), "corpus_bytes": used,
           "distinct_corpus_tokens": len(cnt), "ids": ids}, open(a.out, "w"))
print(f"subset {len(ids)} of {V} tokens; special {len(special)}; corpus {used/1e6:.1f} MB, {len(cnt)} distinct tokens")
# diagnostic coverage on model outputs (never used to build the subset)
S = set(ids)
for f in a.coverage:
    texts = json.load(open(f)).values()
    toks = [t for x in texts if isinstance(x, str) for t in tok.encode(x, add_special_tokens=False)]
    if toks: print(f"coverage {os.path.basename(f)}: {sum(t in S for t in toks)/len(toks):.4f} of {len(toks)} tokens")
