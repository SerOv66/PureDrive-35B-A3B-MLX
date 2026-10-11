"""AUDIT EXP-105: two-sample chi-square, MTP-off (base) vs MTP-on with PREDRAFT (cand), on the sampled completions of
sample_dist.py: per token position (re-tokenized text, positions 0..11), per adjacent pair, and on the first 3 words.
Usage: runtimes/venv-mlx/bin/python chi2_cmp.py base.json cand.json"""
import json, math, sys
from collections import Counter
sys.path.insert(0, __import__("os").environ.get("REPO_DIR", ".") + "/bench")
import prompts

tok = prompts.tokenizer() if hasattr(prompts, "tokenizer") else None
if tok is None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prompts.MODEL_DIR)


def load(p):
    return [tok.encode(i["text"], add_special_tokens=False) for i in json.load(open(p))["items"]]


def chi2_2s(ca, cb, na, nb):
    """Two-sample chi-square on a 2 x C table, categories with pooled expected < 5 merged into 'other'."""
    keys = set(ca) | set(cb); rows = []; oa = ob = 0
    for k in keys:
        a, b = ca.get(k, 0), cb.get(k, 0)
        if (a + b) * min(na, nb) / (na + nb) >= 5: rows.append((a, b))
        else: oa += a; ob += b
    if oa + ob: rows.append((oa, ob))
    if len(rows) < 2: return 0.0, 0, 1.0
    x2 = 0.0
    for a, b in rows:
        t = a + b; ea, eb = t * na / (na + nb), t * nb / (na + nb)
        x2 += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    df = len(rows) - 1
    z = ((x2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return x2, df, 0.5 * math.erfc(z / math.sqrt(2))


A, B = load(sys.argv[1]), load(sys.argv[2])
na, nb = len(A), len(B)
print(f"samples base {na}, cand {nb}; distinct texts base {len(set(map(tuple, A)))}, cand {len(set(map(tuple, B)))}")
ps = []
for pos in range(12):
    ca = Counter(s[pos] if len(s) > pos else -1 for s in A); cb = Counter(s[pos] if len(s) > pos else -1 for s in B)
    x2, df, p = chi2_2s(ca, cb, na, nb); ps.append(p)
    print(f"pos {pos:2d}: categories base {len(ca):3d} cand {len(cb):3d} | chi2 {x2:6.1f} df {df:3d} p {p:.3f}")
for pos in range(1, 8):
    ca = Counter(tuple(s[pos - 1:pos + 1]) for s in A); cb = Counter(tuple(s[pos - 1:pos + 1]) for s in B)
    x2, df, p = chi2_2s(ca, cb, na, nb); ps.append(p)
    print(f"pair {pos - 1}-{pos}: chi2 {x2:6.1f} df {df:3d} p {p:.3f}")
wa = Counter(tuple(t.split(",")[:3]) for t in (i["text"] for i in json.load(open(sys.argv[1]))["items"]))
wb = Counter(tuple(t.split(",")[:3]) for t in (i["text"] for i in json.load(open(sys.argv[2]))["items"]))
x2, df, p = chi2_2s(wa, wb, na, nb); ps.append(p)
print(f"first 3 words (joint): chi2 {x2:6.1f} df {df} p {p:.3f}")
print(f"min p over {len(ps)} tests: {min(ps):.4f} (Bonferroni 0.01 threshold {0.01 / len(ps):.5f})")
