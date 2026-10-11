"""KV8 condition C3, synthetic part (no model): (1) the int8-g32 c35 kernel is row-exact up to L=16 through the same 8-row
blocking as kv8_patch._rows_attention_q8; (2) mx.quantize of a row block == the same rows of a whole-tensor quantize;
(3) an incremental shadow (prefill, decode L=1, verify L=3 with rollback, as kv8_patch._q8_shadow) == quantizing the final
cache at once. Bitwise checks (mx.array_equal). Run with the GPU free: python bench/kv8_c3_synth.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
import mlx.core as mx
import gqa_attn as g

sc = g.D ** -0.5
q8 = lambda x: mx.quantize(x, group_size=32, bits=8)

def rows_q8(q, arrs):   # transcription of kv8_patch._rows_attention_q8
    L, n = q.shape[2], arrs[0].shape[2]
    if L <= 8:
        return g.gqa_attention_q8(q, *arrs, sc)
    outs = []
    for c0 in range(0, L, 8):
        c1 = min(c0 + 8, L); m = n - (L - c1)
        outs.append(g.gqa_attention_q8(q[:, :, c0:c1, :], *[a[:, :, :m] for a in arrs], sc))
    return mx.concatenate(outs, axis=2)

mx.random.seed(7); bad = 0; cases = 0
for N in (17, 255, 257, 4133, 33001):
    k = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16); v = mx.random.normal((1, g.HKV, N, g.D)).astype(mx.bfloat16)
    kw, ks, kb = q8(k); vw, vs, vb = q8(v); arrs = [kw, ks, kb, vw, vs, vb]
    for L in (1, 2, 3, 8, 9, 12, 16):
        if L > N: continue
        q = (mx.random.normal((1, g.HQ, L, g.D)) * (6.0 if L % 2 else 1.0)).astype(mx.bfloat16)
        o = rows_q8(q, arrs)
        for r in range(L):
            m = N - L + r + 1
            one = g.gqa_attention_q8(q[:, :, r:r + 1, :], *[a[:, :, :m] for a in arrs], sc)
            cases += 1
            if not mx.array_equal(o[:, :, r:r + 1, :], one).item(): bad += 1; print("ROW-EXACT FAIL N", N, "L", L, "r", r)
print(f"(1) q8 row-exact L<=16: {cases - bad}/{cases} identical")

bad = 0; cases = 0
x = mx.random.normal((1, 2, 5000, 256)).astype(mx.bfloat16) * 3
whole = q8(x)
for lo, hi in ((0, 1), (0, 5000), (17, 18), (4997, 5000), (1234, 1300)):
    part = q8(x[:, :, lo:hi]); cases += 1
    if not all(mx.array_equal(p, w[:, :, lo:hi]).item() for p, w in zip(part, whole)): bad += 1; print("BLOCK FAIL", lo, hi)
print(f"(2) per-row == per-block quantize: {cases - bad}/{cases} identical")

# (3) shadow: prefill 3000 rows, then cycles of verify L=3 (rewrites rows [n-3, n)) with rollback of rejected rows
cap = 4096; K = mx.random.normal((1, 2, cap, 256)).astype(mx.bfloat16)
sh = [mx.zeros((1, 2, cap, 64), dtype=mx.uint32), mx.zeros((1, 2, cap, 8), dtype=mx.bfloat16), mx.zeros((1, 2, cap, 8), dtype=mx.bfloat16)]
valid = 0
def update(n, L):
    global valid
    lo = max(0, min(valid, n - L))
    w, s, b = q8(K[:, :, lo:n]); sh[0][:, :, lo:n] = w; sh[1][:, :, lo:n] = s; sh[2][:, :, lo:n] = b
    valid = n
n = 3000; update(n, n)
import random; random.seed(3)
for _ in range(200):
    K[:, :, n:n + 3] = mx.random.normal((1, 2, 3, 256)).astype(mx.bfloat16)   # verify window rows (new bf16 rows)
    update(n + 3, 3)
    n += random.choice((1, 2, 3))                                               # accepted rows; the rest is rolled back
    valid = min(valid, n)
final = q8(K[:, :, :n])
ok = all(mx.array_equal(a[:, :, :n], b).item() for a, b in zip(sh, final))
print(f"(3) incremental shadow == final quantize after 200 verify cycles: {'IDENTICAL' if ok else 'DIFFERENT'}")
