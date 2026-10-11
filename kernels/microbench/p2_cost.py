"""Split the c35 attention time into pass 1 (split-K partials) and pass 2 (merge), chained like real layers.
Usage: C35_GQA_VERSION=16 python bench/kernels/p2_cost.py [N ...]"""
import sys, time, statistics
import mlx.core as mx
import gqa_attn as g

def t_chain(fn, iters=30, layers=10):
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
for N in [int(x) for x in sys.argv[1:]] or [32768, 131072]:
    k = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16); v = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16)
    mx.eval(k, v)
    for L in (1, 3):
        q = mx.random.normal((1, g.HQ, L, g.D)).astype(mx.bfloat16); mx.eval(q)
        ns = (N + g.B - 1) // g.B; k1, k2 = g._kernels(L)
        nkv = mx.array([N], dtype=mx.int32); s = mx.array([sc], dtype=mx.float32)
        def p1(prev):
            qq = q if prev is None else q + 0 * prev[0][0, :1, :1, :1].astype(q.dtype)
            return k1(inputs=[qq, k, v, nkv, s], template=[("T", q.dtype)], grid=(32 * g.G, ns, g.HKV), threadgroup=(32 * g.G, 1, 1),
                      output_shapes=[(ns, g.HQ, L, g.D), (ns, g.HQ, L), (ns, g.HQ, L)], output_dtypes=[mx.float32] * 3)
        def full(prev):
            qq = q if prev is None else q + 0 * prev
            return g.gqa_attention(qq, k, v, sc)
        a, b = t_chain(p1), t_chain(full)
        print(f"v{g.VERSION} N={N:7d} L={L}: pass1 {a:.3f} ms  full {b:.3f} ms  pass2+glue {b - a:.3f} ms  partials {ns * g.HQ * L * g.D * 4 / 1e6:.1f} MB", flush=True)
