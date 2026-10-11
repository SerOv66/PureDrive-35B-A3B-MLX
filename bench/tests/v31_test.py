"""EXP-091 checks of attention v31 in bench/kernels/gqa_attn.py (run with C35_GQA_VERSION=31 C35_GQA_P2=3 [C35_GQA_BRULE=1]):
(1) accuracy vs fp32 reference; (2) row-exactness: every row of an L-row call == the 1-row call at its position, incl. rows
straddling the split-size thresholds of the BRULE (8192, 49152, 98304 keys); (3) KDEV/QREG variants bitwise identical;
(4) integration == standalone EXP-088 source (logs/exp088/gqa_v31.py) at B=256; (5) sensitivity: v16 differs."""
import os, sys, importlib
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "bench", "kernels"))
import mlx.core as mx
import gqa_attn as G
assert G.VERSION == 31, "run with C35_GQA_VERSION=31"
sc = G.D ** -0.5
mx.random.seed(1)
bad = n = 0
def rowexact(q, kb, vb, N, L):
    o = G.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc)
    return o, all(mx.array_equal(o[:, :, r:r + 1], G.gqa_attention(q[:, :, r:r + 1], kb[:, :, :N - L + r + 1], vb[:, :, :N - L + r + 1], sc)).item() for r in range(L))
cases = [(300, (1, 3, 8)), (4133, (1, 2, 3, 5, 8)), (8194, (3, 8)), (8199, (8,)), (33000, (3,)), (49153, (3, 8)), (98306, (3, 8))]
for N, Ls in cases:
    alloc = N + 256
    kb = mx.random.normal((1, G.HKV, alloc, G.D)).astype(mx.bfloat16); vb = mx.random.normal((1, G.HKV, alloc, G.D)).astype(mx.bfloat16)
    for L in Ls:
        q = mx.random.normal((1, G.HQ, L, G.D)).astype(mx.bfloat16)
        o, rex = rowexact(q, kb, vb, N, L)
        err = mx.abs(o.astype(mx.float32) - G.reference(q, kb[:, :, :N], vb[:, :, :N], sc)).max().item()
        outs = []
        for kd, qr in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if kd == 0 and G._v31_tg_bytes(L, 0, 0) > 32768: continue
            G.V31_KDEV, G.V31_QREG = kd, qr; outs.append(G.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc))
        G.V31_KDEV, G.V31_QREG = -1, -1
        same = all(mx.array_equal(x, outs[0]).item() for x in outs) and mx.array_equal(outs[0], o).item()
        n += 1; ok = rex and same and err < 2e-2
        bad += not ok
        print(f"N={N} L={L} brule={G.BRULE} B={[G._brule(N - L + r) for r in range(L)] if G.BRULE else G.B} err={err:.2e} rowexact={rex} variants_same={same}{'' if ok else '  <-- BAD'}", flush=True)
# (4)+(5) at the module split size, BRULE off
G.BRULE = False
sys.path.insert(0, os.path.join(W, "logs", "exp088")); S = importlib.import_module("gqa_v31")
N = 4133; kb = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16); vb = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
for L in (1, 3, 8):
    q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16)
    a = G.gqa_attention(q, kb, vb, sc); b = S.gqa_attention(q, kb, vb, sc, 1 if L > 6 else 0, 0)
    G.VERSION = 16; c = G.gqa_attention(q, kb, vb, sc); G.VERSION = 31
    eq, d16 = mx.array_equal(a, b).item(), not mx.array_equal(a, c).item()
    n += 1; bad += not (eq and d16)
    print(f"L={L}: integrated == EXP-088 source: {eq}; differs from v16 (sensitivity): {d16}")
print(f"TOTAL {n} cases, {bad} bad")
sys.exit(1 if bad else 0)
