"""H24 gate: chained microbench of the int8-g32 c35 kernel (gqa_attention_q8, dequant at staging) against bf16 v16,
L=1 and L=3, 16K..128K (synthetic KV). If KV8 does not save >=0.15 ms/layer at 128K L=3, the KV8 quality gate is not worth it.
Usage: C35_GQA_VERSION=16 C35_GQA_P2=3 python bench/q8_gate.py"""
import os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
import mlx.core as mx
import gqa_attn as g

def chain(fn, iters=30, layers=10):
    def step():
        out = fn(None)
        for _ in range(layers - 1): out = fn(out)
        return out
    for _ in range(3): mx.eval(step())
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(step()); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / layers * 1e3

sc = g.D ** -0.5
print(f"v{g.VERSION} P2={g.P2V}   ms/layer: bf16 | q8 | delta")
for N in (16384, 32768, 65536, 131072):
    k = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16); v = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16)
    kw, ks, kb = mx.quantize(k, group_size=32, bits=8); vw, vs, vb = mx.quantize(v, group_size=32, bits=8)
    mx.eval(k, v, kw, ks, kb, vw, vs, vb)
    for L in (1, 3):
        q = mx.random.normal((1, g.HQ, L, g.D)).astype(mx.bfloat16); mx.eval(q)
        tb = chain(lambda p: g.gqa_attention(q if p is None else q + 0 * p, k, v, sc))
        tq = chain(lambda p: g.gqa_attention_q8(q if p is None else q + 0 * p, kw, ks, kb, vw, vs, vb, sc))
        print(f"N={N:7d} L={L}: {tb:.3f} | {tq:.3f} | {tq - tb:+.3f}", flush=True)
