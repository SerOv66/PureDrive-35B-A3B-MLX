"""KV-cache 8-bit format study on the REAL model's K/V (no benchmark text): prefill a non-benchmark code context, then
compare the relative reconstruction error of the 10 full-attention layers' K and V under
  - TurboQuant 8-bit MSE as shipped in mlx-vlm (RHT rotation + quantile codebook),
  - TurboQuant 8-bit with a Lloyd-Max codebook for the same coordinate distribution,
  - MLX affine 8-bit, group 64 and group 32 (Q8_0-like).
Usage: python quality/kv_quant_err.py --tokens 16384 --out quality/results/kv_quant_err.json"""
import argparse, glob, json, math, os, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "bench"))
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load
from mlx_vlm.models import cache as vcache
from mlx_vlm.turboquant import _TurboQuantMSECodec, _beta_pdf
import prompts

ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int, default=16384); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = prompts.tokenizer()
files = sorted(glob.glob(f"{W}/runtimes/omlx/omlx/**/*.py", recursive=True))   # non-benchmark source
ids = []
for f in files:
    ids += tok.encode(open(f, encoding="utf-8", errors="ignore").read(), add_special_tokens=False)
    if len(ids) >= a.tokens: break
ids = ids[: a.tokens]
model, _ = load(prompts.MODEL_DIR)
lm = model.language_model
c = vcache.make_prompt_cache(lm)
x = mx.array(ids)[None]
for i in range(0, len(ids), 2048):
    lm(x[:, i:i + 2048], cache=c); mx.eval([cc.state for cc in c])

def lloyd_codebook(dim, bits, iters=60):
    grid = np.linspace(-1 + 1e-6, 1 - 1e-6, 65536); w = _beta_pdf(grid.astype(np.float32), dim).astype(np.float64); w /= w.sum()
    cdf = np.cumsum(w); levels = 1 << bits
    cb = np.interp((np.arange(levels) + 0.5) / levels, cdf, grid)
    for _ in range(iters):
        mid = (cb[:-1] + cb[1:]) / 2; idx = np.searchsorted(mid, grid)
        num = np.bincount(idx, weights=w * grid, minlength=levels); den = np.bincount(idx, weights=w, minlength=levels)
        cb = np.where(den > 0, num / np.maximum(den, 1e-30), cb)
    return mx.array(cb.astype(np.float32))

def rel(xh, x):
    x = x.astype(mx.float32); return (mx.linalg.norm((xh.astype(mx.float32) - x).reshape(-1)) / mx.linalg.norm(x.reshape(-1))).item()

res = {"tokens": len(ids), "layers": {}}
lloyd = None
for li, cc in enumerate(c):
    if not hasattr(cc, "keys") or cc.keys is None or getattr(cc, "offset", 0) == 0 or cc.keys.ndim != 4: continue
    k = cc.keys[:, :, :cc.offset]; v = cc.values[:, :, :cc.offset]
    r = {}
    for name, t in (("K", k), ("V", v)):
        codec = _TurboQuantMSECodec(t.shape[-1], 8, 0)
        r[f"{name}_tq8_quantile"] = rel(codec.dequantize(codec.quantize(t)).reshape(t.shape), t)
        if lloyd is None: lloyd = lloyd_codebook(t.shape[-1], 8)
        codec.codebook = lloyd; codec._midpoints = (lloyd[:-1] + lloyd[1:]) / 2
        r[f"{name}_tq8_lloyd"] = rel(codec.dequantize(codec.quantize(t)).reshape(t.shape), t)
        for gs in (64, 32):
            wq, s, b = mx.quantize(t, group_size=gs, bits=8)
            r[f"{name}_affine8_g{gs}"] = rel(mx.dequantize(wq, s, b, group_size=gs, bits=8), t)
        r[f"{name}_bf16_absmax_ch_ratio"] = (mx.abs(t).max(axis=(0, 1, 2)).max() / mx.abs(t).max(axis=(0, 1, 2)).mean()).item()
    res["layers"][li] = r
    print(li, {k_: round(v_, 5) for k_, v_ in r.items()}, flush=True)
agg = {k_: float(np.mean([r[k_] for r in res["layers"].values()])) for k_ in next(iter(res["layers"].values()))}
res["mean"] = agg
print("MEAN", {k_: round(v_, 5) for k_, v_ in agg.items()})
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
