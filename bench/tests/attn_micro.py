"""Decode-attention microbenchmark for this model's full-attention layers: Hq=16, Hkv=2 (GQA 8:1), D=256, bf16 KV.
Measures mx.fast.scaled_dot_product_attention at q_len 1/4/8 (decode / MTP verify rows) vs context L.
Reports ms per call, effective KV bandwidth (K+V bytes / time) and ms per token for the 10 full-attention layers.
Usage: python bench/attn_micro.py [--L 2048 32768 65536 131072] [--q 1 4 8]"""
import argparse, time, statistics, mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, nargs="+", default=[2048, 32768, 65536, 131072])
ap.add_argument("--q", type=int, nargs="+", default=[1, 4, 8])
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()
Hq, Hkv, D = 16, 2, 256
scale = D ** -0.5

def bench(fn):
    for _ in range(5): mx.eval(fn())
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

print(f"{'L':>7} {'q':>2} {'ms/call':>8} {'KV GB/s':>8} {'ms/token(10 layers)':>20}")
for L in a.L:
    k = mx.random.normal((1, Hkv, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hkv, L, D)).astype(mx.bfloat16)
    mx.eval(k, v)
    kvb = k.nbytes + v.nbytes
    for q in a.q:
        x = mx.random.normal((1, Hq, q, D)).astype(mx.bfloat16); mx.eval(x)
        mask = "causal" if q > 1 else None
        t = bench(lambda: mx.fast.scaled_dot_product_attention(x, k, v, scale=scale, mask=mask))
        print(f"{L:7d} {q:2d} {t*1e3:8.3f} {kvb/t/1e9:8.1f} {10*t*1e3:20.2f}", flush=True)
    del k, v
