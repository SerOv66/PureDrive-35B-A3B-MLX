"""AUDIT EXP-110: copy of quality/kl_decode.py (frozen-free, unchanged arithmetic) that runs BRULE=1 and BRULE=2 on the SAME
prefilled cache and forced tokens: modes ref (stock SDPA), exact (fp32 attention), b1 (v31 BRULE=1), b2 (v31 BRULE=2).
Saves per-step KL so the BRULE2-BRULE1 difference can be judged paired (bootstrap CI).
Usage: C35_GQA_VERSION=31 C35_GQA_P2=3 C35_GQA_BRULE=2 runtimes/venv-mlx/bin/python bench/audit/kl_decode_ab.py --ctx N --steps S --out F"""
import argparse, copy, json, os, sys, time
W = __import__("os").environ.get("REPO_DIR", ".")
sys.path.insert(0, os.path.join(W, "bench")); sys.path.insert(0, os.path.join(W, "bench", "kernels"))
import mlx.core as mx
import numpy as np
from mlx_vlm.utils import load
from mlx_vlm.models import cache as vcache
import prompts
import gqa_attn as G
assert G.VERSION == 31 and G.BRULE and G.P2V == 3

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from mlx_vlm.models.qwen3_5 import language as q35
ORIG = q35.scaled_dot_product_attention


def log_softmax(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def exact_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):   # verbatim from quality/kl_decode.py
    if queries.shape[2] <= 8 and queries.shape[1] == 16 and keys.shape[1] == 2 and not kw \
            and (mask is None or (isinstance(mask, str) and mask == "causal")):
        B, H, L, D = queries.shape; N = keys.shape[2]
        q = queries.astype(mx.float32).reshape(B, 2, 8 * L, D) if L == 1 else \
            queries.astype(mx.float32).reshape(B, 2, 8, L, D).reshape(B, 2, 8 * L, D)
        k = keys.astype(mx.float32); v = values.astype(mx.float32)
        s = (q * scale) @ k.transpose(0, 1, 3, 2)
        if L > 1:
            rows = mx.tile(mx.arange(L), 8)[None, None, :, None]
            s = mx.where(mx.arange(N)[None, None, None, :] <= N - L + rows, s, -mx.inf)
        o = mx.softmax(s, axis=-1) @ v
        return o.reshape(B, 2, 8, L, D).reshape(B, H, L, D).astype(queries.dtype)
    return ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


def cand_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):   # verbatim from quality/kl_decode.py
    if queries.shape[2] <= 8 and queries.shape[1] == 16 and queries.shape[3] == 256 and keys.shape[1] == 2 and not kw \
            and (mask is None or (isinstance(mask, str) and mask == "causal")):
        return G.gqa_attention(queries, keys, values, scale)
    return ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


model, proc = load(prompts.MODEL_DIR)
lm = model.language_model
corpus = prompts.corpus_ids()
ctx = a.ctx
ids = prompts.prompt_ids(ctx, 0)
head, tail = prompts.head_tail(0)
n = ctx - len(head) - len(tail)
forced = list(corpus[n:n + a.steps])
t0 = time.time()
c0 = vcache.make_prompt_cache(lm)
x = mx.array(ids)[None]
for i in range(0, x.shape[1], 2048):
    out = lm(x[:, i:i + 2048], cache=c0)
    mx.eval([c.state for c in c0])
last_logits = out.logits[:, -1, :]
mx.eval(last_logits)
tpre = time.time() - t0
lp = {}
for mode in ("ref", "exact", "b1", "b2"):
    q35.scaled_dot_product_attention = {"ref": ORIG, "exact": exact_sdpa, "b1": cand_sdpa, "b2": cand_sdpa}[mode]
    G.BRULE2 = mode == "b2"
    c = copy.deepcopy(c0)
    rows = [log_softmax(last_logits.astype(mx.float32))]
    for t in forced[:-1]:
        o = lm(mx.array([[t]]), cache=c)
        rows.append(log_softmax(o.logits[:, -1, :].astype(mx.float32)))
        mx.eval(rows[-1])
    lp[mode] = mx.concatenate(rows, axis=0); mx.eval(lp[mode]); del c
    print(f"mode {mode} done {time.time() - t0:.0f}s", flush=True)
ex = lp["exact"]; f = mx.array(forced)
kl = {m: np.array((mx.exp(ex) * (ex - lp[m])).sum(axis=-1)) for m in ("ref", "b1", "b2")}
kl12 = np.array((mx.exp(lp["b1"]) * (lp["b1"] - lp["b2"])).sum(axis=-1))
d = {"ctx": ctx, "steps": a.steps, "prefill_s": tpre, "B_at_ctx": {"brule1": G._brule(ctx), "brule2": G._brule2(ctx)}}
for m in ("ref", "b1", "b2"):
    d[m] = dict(kl_exact=float(kl[m].mean()), kl_exact_p99=float(np.percentile(kl[m], 99)), kl_exact_max=float(kl[m].max()),
                top1_exact=float(np.mean(np.array(ex.argmax(-1)) == np.array(lp[m].argmax(-1)))),
                ppl=float(np.exp(-np.array(mx.take_along_axis(lp[m], f[:, None], axis=-1)).mean())))
d["ppl_exact"] = float(np.exp(-np.array(mx.take_along_axis(ex, f[:, None], axis=-1)).mean()))
d["kl_b1_b2_mean"] = float(kl12.mean()); d["kl_b1_b2_max"] = float(kl12.max())
rng = np.random.default_rng(0); diff = kl["b2"] - kl["b1"]
bs = [rng.choice(diff, diff.size).mean() for _ in range(5000)]
d["paired_b2_minus_b1"] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
d["first256"] = {m: float(kl[m][:256].mean()) for m in kl}
d["per_step"] = {m: [round(float(v), 6) for v in kl[m]] for m in kl}
json.dump(d, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in d.items() if k != "per_step"}), flush=True)
