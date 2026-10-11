"""KL-divergence of decode logits: reference attention (stock MLX SDPA) vs candidate kernel (c35 GQA), same weights.
Teacher-forced decode (L=1 per step, like generation) of S real corpus tokens after a real prompt prefix of ctx tokens.
Both modes start from the SAME prefilled cache (prefill uses the stock path, L>8), so only decode-attention numerics differ.
Reports mean / p99 / max KL(ref||cand) in nats, top-1 agreement, PPL of the forced tokens under each mode.
Usage: python quality/kl_decode.py --ctx 32768 131072 --steps 256 --out quality/results/kl_<name>.json"""
import argparse, copy, json, os, sys, time
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "bench")); sys.path.insert(0, os.path.join(W, "bench", "kernels"))
import mlx.core as mx
import numpy as np
from mlx_vlm.utils import load
from mlx_vlm.models import cache as vcache
import prompts
from gqa_attn import gqa_attention


def log_softmax(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, nargs="+", default=[32768])
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--out", required=True)
ap.add_argument("--kv8", default=None, choices=["tq8", "affine32"], help="add mode 'kv8': stock attention on a fake-quantized 8-bit KV (DOSSIER_KV quality study)")
a = ap.parse_args()

from mlx_vlm.models.qwen3_5 import language as q35
ORIG = q35.scaled_dot_product_attention


def exact_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
    """fp32 reference attention for decode-shaped calls (L<=8, GQA 16:2, causal bottom-right)."""
    if queries.shape[2] <= 8 and queries.shape[1] == 16 and keys.shape[1] == 2 and not kw \
            and (mask is None or (isinstance(mask, str) and mask == "causal")):
        B, H, L, D = queries.shape; N = keys.shape[2]
        q = queries.astype(mx.float32).reshape(B, 2, 8 * L, D) if L == 1 else \
            queries.astype(mx.float32).reshape(B, 2, 8, L, D).reshape(B, 2, 8 * L, D)
        k = keys.astype(mx.float32); v = values.astype(mx.float32)
        s = (q * scale) @ k.transpose(0, 1, 3, 2)                        # [B, 2, 8L, N]
        if L > 1:
            rows = mx.tile(mx.arange(L), 8)[None, None, :, None]        # row index of each (head, row) pair
            s = mx.where(mx.arange(N)[None, None, None, :] <= N - L + rows, s, -mx.inf)
        o = mx.softmax(s, axis=-1) @ v                                   # [B, 2, 8L, D]
        return o.reshape(B, 2, 8, L, D).reshape(B, H, L, D).astype(queries.dtype)
    return ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


def fq(t):
    """quantize -> dequantize along head_dim (8 bit), returned in t.dtype."""
    if a.kv8 == "affine32":
        w, sc_, b = mx.quantize(t, group_size=32, bits=8)
        return mx.dequantize(w, sc_, b, group_size=32, bits=8).astype(t.dtype)
    from mlx_vlm.turboquant import _TurboQuantMSECodec
    codec = FQ_CODECS.setdefault(t.shape[-1], _TurboQuantMSECodec(t.shape[-1], 8, 0))
    return codec.dequantize(codec.quantize(t)).reshape(t.shape).astype(t.dtype)
FQ_CODECS = {}


def kv8_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
    """decode step: fake-quantize the L new K/V rows in the cache (the prefix was fake-quantized once after prefill)."""
    if queries.shape[2] <= 8 and keys.shape[1] == 2 and cache is not None and getattr(cache, "keys", None) is not None:
        L, n = queries.shape[2], cache.offset
        cache.keys[..., n - L:n, :] = fq(cache.keys[..., n - L:n, :]); cache.values[..., n - L:n, :] = fq(cache.values[..., n - L:n, :])
        keys, values = cache.keys[..., :n, :], cache.values[..., :n, :]
    return ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


def cand_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
    if queries.shape[2] <= 8 and queries.shape[1] == 16 and queries.shape[3] == 256 and keys.shape[1] == 2 and not kw \
            and (mask is None or (isinstance(mask, str) and mask == "causal")):
        return gqa_attention(queries, keys, values, scale)
    return ORIG(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


model, proc = load(prompts.MODEL_DIR)
lm = model.language_model
corpus = prompts.corpus_ids()
res = {"ctx": {}, "steps": a.steps}
for ctx in a.ctx:
    ids = prompts.prompt_ids(ctx, 0)
    # forced continuation: real corpus tokens following the prompt's corpus slice
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
    for mode in ("ref", "cand", "exact") + (("kv8",) if a.kv8 else ()):
        q35.scaled_dot_product_attention = {"ref": ORIG, "cand": cand_sdpa, "exact": exact_sdpa, "kv8": kv8_sdpa}[mode]
        c = copy.deepcopy(c0)
        if mode == "kv8":   # fake-quantize the prefilled K/V of the full-attention layers once
            for cc in c:
                if getattr(cc, "keys", None) is not None and cc.keys.ndim == 4:
                    n = cc.offset; cc.keys[..., :n, :] = fq(cc.keys[..., :n, :]); cc.values[..., :n, :] = fq(cc.values[..., :n, :])
        rows = [log_softmax(last_logits.astype(mx.float32))]
        for t in forced[:-1]:
            o = lm(mx.array([[t]]), cache=c)
            rows.append(log_softmax(o.logits[:, -1, :].astype(mx.float32)))
            mx.eval(rows[-1])
        lp[mode] = mx.concatenate(rows, axis=0)
        mx.eval(lp[mode]); del c
    q35.scaled_dot_product_attention = ORIG
    r, cnd, ex = lp["ref"], lp["cand"], lp["exact"]
    kl = np.array((mx.exp(r) * (r - cnd)).sum(axis=-1))
    kl_ex_ref = np.array((mx.exp(ex) * (ex - r)).sum(axis=-1))
    kl_ex_cand = np.array((mx.exp(ex) * (ex - cnd)).sum(axis=-1))
    top1 = float(np.mean(np.array(r.argmax(-1)) == np.array(cnd.argmax(-1))))
    f = mx.array(forced)
    nll_r = -np.array(mx.take_along_axis(r, f[:, None], axis=-1)).mean()
    nll_c = -np.array(mx.take_along_axis(cnd, f[:, None], axis=-1)).mean()
    d = dict(kl_exact_vs_stock=float(kl_ex_ref.mean()), kl_exact_vs_cand=float(kl_ex_cand.mean()),
             top1_exact_stock=float(np.mean(np.array(ex.argmax(-1)) == np.array(r.argmax(-1)))),
             top1_exact_cand=float(np.mean(np.array(ex.argmax(-1)) == np.array(cnd.argmax(-1)))),
             kl_mean=float(kl.mean()), kl_p99=float(np.percentile(kl, 99)), kl_max=float(kl.max()), top1_agree=top1,
             ppl_ref=float(np.exp(nll_r)), ppl_cand=float(np.exp(nll_c)), prefill_s=tpre)
    if a.kv8:
        k8 = lp["kv8"]
        kl8 = np.array((mx.exp(r) * (r - k8)).sum(axis=-1))
        d.update(kv8_format=a.kv8, kl_kv8_mean=float(kl8.mean()), kl_kv8_p99=float(np.percentile(kl8, 99)), kl_kv8_max=float(kl8.max()),
                 top1_agree_kv8=float(np.mean(np.array(r.argmax(-1)) == np.array(k8.argmax(-1)))),
                 ppl_kv8=float(np.exp(-np.array(mx.take_along_axis(k8, f[:, None], axis=-1)).mean())))
    res["ctx"][ctx] = d
    print(ctx, json.dumps(d), flush=True)
    del c0, lp, r, cnd, ex
    mx.clear_cache()
json.dump(res, open(a.out, "w"), indent=1)
