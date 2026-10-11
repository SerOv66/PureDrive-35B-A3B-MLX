"""H31 (EXP-110): (1) runtime-B v31 kernels == compiled-B v31 kernels, bitwise, for L 1..8 (KDEV/QREG auto variants), many N and B
(incl. partial tiles/splits); (2) with C35_GQA_BRULE=2 every verify row equals the 1-row decode at its position (row-exact),
also across a change of B inside the window. Usage: runtimes/venv-mlx/bin/python bench/tests/v31r_test.py [dir with gqa_attn.py]"""
import os, sys
os.environ.update(C35_GQA_VERSION="31", C35_GQA_P2="3", C35_GQA_BRULE="2")
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
import mlx.core as mx
import gqa_attn as g
assert g.BRULE and g.BRULE2
sc = g.D ** -0.5
mx.random.seed(11)
bad = n = 0
for N in (300, 2051, 9000, 33000, 70001):
    kb = mx.random.normal((1, g.HKV, N + 64, g.D)).astype(mx.bfloat16); vb = mx.random.normal((1, g.HKV, N + 64, g.D)).astype(mx.bfloat16)
    k, v = kb[:, :, :N], vb[:, :, :N]
    for L in range(1, 9):
        q = mx.random.normal((1, g.HQ, L, g.D)).astype(mx.bfloat16)
        for bs in (128, 208, 256, 416, 832, 1648, 2464):
            if (N + bs - 1) // bs < 1: continue
            a = g._gqa31(q, k, v, sc, bs, False); b = g._gqa31(q, k, v, sc, bs, True)
            n += 1
            if not mx.array_equal(a, b).item():
                bad += 1; print("DIFF compiled vs runtime B: N", N, "L", L, "B", bs)
print(f"runtime-B vs compiled-B: {n - bad}/{n} BITWISE IDENTICAL", flush=True)
# row-exactness under BRULE=2, including windows that straddle a change of B (p = 640*j boundaries and k changes)
rb = rn = 0
for N in (8195, 16384 + 3, 640 * 26 + 1, 640 * 26 + 2, 49152 + 2, 110592 + 1, 110592 + 2, 131072 + 3):
    kb = mx.random.normal((1, g.HKV, N + 8, g.D)).astype(mx.bfloat16); vb = mx.random.normal((1, g.HKV, N + 8, g.D)).astype(mx.bfloat16)
    for L in (1, 2, 3, 8):
        q = mx.random.normal((1, g.HQ, L, g.D)).astype(mx.bfloat16)
        o = g.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc)
        for r in range(L):
            one = g.gqa_attention(q[:, :, r:r + 1], kb[:, :, :N - L + r + 1], vb[:, :, :N - L + r + 1], sc)
            rn += 1
            if not mx.array_equal(o[:, :, r:r + 1], one).item():
                rb += 1; print("ROW-EXACT FAIL N", N, "L", L, "row", r, "B", [g._brule2(N - L + i) for i in range(L)])
ref_err = mx.abs(g.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc).astype(mx.float32) - g.reference(q, kb[:, :, :N], vb[:, :, :N], sc)).max().item()
print(f"BRULE=2 row-exact: {rn - rb}/{rn}; max|err| vs fp32 reference at N {N} L {L}: {ref_err:.2e}")
sys.exit(0 if bad == 0 and rb == 0 and ref_err < 2e-2 else 1)
