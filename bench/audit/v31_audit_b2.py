"""EXP-110 copy of the AUDIT EXP-092 tool, adapted to C35_GQA_BRULE=2 (wave-exact B, runtime-B kernels): independent checks of attention v31 + BRULE (bench/kernels/gqa_attn.py), synthetic tensors only.
Run with: env C35_GQA_VERSION=31 C35_GQA_P2=3 C35_GQA_BRULE=2 runtimes/venv-mlx/bin/python bench/audit/v31_audit_b2.py
(1) row-exactness: every row of an L-row call (L=2..8, and L=16/24 through the runtime's _rows_attention) == the 1-row decode
    at its position, with N chosen so the rows straddle the BRULE thresholds 8192/49152/98304; 3 input distributions
    (flat N(0,1), peaky scores, channel outliers + sink); (2) KDEV/QREG(/DBUF) variants bitwise identical;
(3) accuracy vs fp32 against v16 and stock SDPA; (4) determinism; (5) sensitivity: kernel mutation whose partial-sum order
    depends on LROWS, and a split-size rule taken from the call (N-1) instead of the row position -> must be detected."""
import importlib.util, os, sys
W = __import__("os").environ.get("REPO_DIR", ".")
sys.path.insert(0, os.path.join(W, "bench", "kernels"))
import mlx.core as mx
import gqa_attn as G
assert G.VERSION == 31 and G.BRULE and G.BRULE2 and G.P2V == 3, (G.VERSION, G.BRULE, G.P2V)
spec = importlib.util.spec_from_file_location("c35p", os.path.join(W, "runtimes/omlx/omlx/patches/c35_gqa_attention.py"))
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
sc = G.D ** -0.5
mx.random.seed(92)


def mk(alloc, kind):
    k = mx.random.normal((1, G.HKV, alloc, G.D)); v = mx.random.normal((1, G.HKV, alloc, G.D))
    if kind == "outlier":   # a few huge channels in K and V, attention sink at key 0
        ch = mx.zeros((G.D,)); ch = mx.where(mx.arange(G.D) % 61 == 3, 12.0, 1.0)
        k = k * ch; v = v * mx.where(mx.arange(G.D) % 37 == 5, 20.0, 1.0)
        k = mx.concatenate([k[:, :, :1] * 0 + 1.0, k[:, :, 1:]], axis=2)
    return k.astype(mx.bfloat16), v.astype(mx.bfloat16)


def mkq(L, kind):
    q = mx.random.normal((1, G.HQ, L, G.D))
    if kind == "peaky": q = q * 5.0
    if kind == "outlier": q = q * 2.0 + 1.0
    return q.astype(mx.bfloat16)


def call(q, k, v):
    return RP._rows_attention(G.gqa_attention, q, k, v, sc)   # exactly the runtime entry (L<=8 -> gqa_attention)


def rowexact(q, kb, vb, N, L, fn=call):
    o = fn(q, kb[:, :, :N], vb[:, :, :N])
    ok = 0
    for r in range(L):
        o1 = fn(q[:, :, r:r + 1], kb[:, :, :N - L + r + 1], vb[:, :, :N - L + r + 1])
        ok += mx.array_equal(o[:, :, r:r + 1], o1).item()
    return o, ok


tot_rows = ok_rows = 0
bad = []
cases = [(3, (300, 2051, 4133, 8192, 8193, 8194, 8195, 16641, 16642, 33003, 49281, 49282, 65539, 110593, 110594, 110595, 129281, 129282, 131072)),
         (2, (8193, 16641, 110593)), (4, (8195, 16643, 110595)), (5, (16644, 110596)), (6, (8197, 110597)), (7, (16646, 110598)),
         (8, (300, 8193, 8196, 8199, 16641, 16647, 49287, 110593, 110599, 129287, 131072)),
         (16, (8200, 8207, 16648, 16655, 110600, 110607, 131072)), (24, (8200, 16660, 110610))]
for kind in ("normal", "peaky", "outlier"):
    kb, vb = mk(131072 + 64, kind)
    for L, Ns in cases:
        for N in Ns:
            q = mkq(L, kind)
            o, ok = rowexact(q, kb, vb, N, L)
            tot_rows += L; ok_rows += ok
            Bs = sorted(set(G._brule2(N - L + r) for r in range(L)))
            if ok != L: bad.append((kind, L, N, ok))
            print(f"rowexact {kind:7s} L={L:2d} N={N:6d} B={Bs} rows {ok}/{L}", flush=True)
print(f"ROWEXACT TOTAL {ok_rows}/{tot_rows} bad={bad}", flush=True)

# (2) staging variants
vb_ok = vb_n = 0
kb, vb = mk(131072 + 64, "peaky")
for L in (1, 2, 3, 4, 5, 6, 7, 8):
    for N in (8195, 49155, 98306, 131072, 2051):
        q = mkq(L, "peaky"); outs = []
        for kd in (0, 1):
            for qr in (0, 1):
                if kd == 0 and G._v31_tg_bytes(L, 0, 0) > 32768: continue
                G.V31_KDEV, G.V31_QREG = kd, qr
                outs.append(G.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc))
        G.V31_KDEV, G.V31_QREG = -1, -1
        auto = G.gqa_attention(q, kb[:, :, :N], vb[:, :, :N], sc)
        same = all(mx.array_equal(x, auto).item() for x in outs)
        vb_n += 1; vb_ok += same
        if not same: print("VARIANT MISMATCH", L, N)
print(f"VARIANTS bitwise identical: {vb_ok}/{vb_n} (variants per case incl. DBUF switch at L=4,5)", flush=True)

# (3) accuracy vs fp32 (max |err| and mean |err| / mean |ref|) -- v31+BRULE, v16 (B=256), stock SDPA
print("accuracy: kind N L | v31 max mean_rel | v16 max mean_rel | sdpa max mean_rel")
acc = []
for kind in ("normal", "peaky", "outlier"):
    kb, vb = mk(131072, kind)
    for N in (2048, 32768, 65536, 131072):
        for L in (1, 3):
            q = mkq(L, kind); k, v = kb[:, :, :N], vb[:, :, :N]
            ref = G.reference(q, k, v, sc); den = mx.abs(ref).mean().item()
            row = []
            for name in ("v31", "v16", "sdpa"):
                if name == "v31": o = G.gqa_attention(q, k, v, sc)
                elif name == "v16":
                    G.VERSION = 16; o = G.gqa_attention(q, k, v, sc); G.VERSION = 31
                else: o = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal" if L > 1 else None)
                e = mx.abs(o.astype(mx.float32) - ref)
                row.append((e.max().item(), e.mean().item() / den))
            acc.append((kind, N, L, row))
            print(f"acc {kind:7s} {N:6d} {L} | " + " | ".join(f"{a:.2e} {b:.2e}" for a, b in row), flush=True)
import statistics
for i, name in enumerate(("v31", "v16", "sdpa")):
    print(f"ACC mean_rel geo-avg {name}: {statistics.geometric_mean([r[3][i][1] for r in acc]):.3e}  worst max {max(r[3][i][0] for r in acc):.2e}")

# (4) determinism
kb, vb = mk(98306 + 8, "peaky"); q = mkq(3, "peaky")
d = all(mx.array_equal(G.gqa_attention(q, kb[:, :, :98306], vb[:, :, :98306], sc), G.gqa_attention(q, kb[:, :, :98306], vb[:, :, :98306], sc)).item() for _ in range(5))
print(f"DETERMINISM 5/5 repeated calls identical: {d}")

# (5) sensitivity
# 5a: kernel mutation, partial-sum order depends on LROWS (not row-exact by construction)
orig_src = G._P1V31
needle = "const float4 s4 = ((*((threadgroup float4*)(Sp + o4)) + *((threadgroup float4*)(Sp + SPN + o4)))"
assert needle in orig_src
mut = orig_src.replace(needle, "const float4 s4 = (LROWS > 1) ? ((*((threadgroup float4*)(Sp + o4)) + *((threadgroup float4*)(Sp + 2 * SPN + o4))) + *((threadgroup float4*)(Sp + SPN + o4))) + *((threadgroup float4*)(Sp + 3 * SPN + o4)) : ((*((threadgroup float4*)(Sp + o4)) + *((threadgroup float4*)(Sp + SPN + o4)))")
G._P1V31 = mut; G._cache31.clear(); G._cache31r.clear()
_orig_mk = mx.fast.metal_kernel
def _mk_renamed(**kw):   # distinct kernel names so no compiled library of the original is reused
    kw["name"] = kw["name"] + "_audmut"; return _orig_mk(**kw)
mx.fast.metal_kernel = _mk_renamed
kb, vb = mk(131072 + 64, "peaky")
det = tot = 0
for L, N in ((3, 8194), (3, 16642), (3, 110594), (8, 8199), (8, 131072), (3, 33003)):
    q = mkq(L, "peaky"); _, ok = rowexact(q, kb, vb, N, L); det += L - ok; tot += L
print(f"MUTATION 5a (sum order depends on LROWS): rows detected non-exact {det}/{tot}")
G._P1V31 = orig_src; G._cache31.clear(); G._cache31r.clear(); mx.fast.metal_kernel = _orig_mk
# 5b: split size from the call (N-1) instead of each row position
orig_rows = G._gqa31_rows
def bad_rows(q, k, v, scale):
    return G._gqa31(q, k, v, scale, G._brule2(k.shape[2] - 1), True)
G._gqa31_rows = bad_rows
det = tot = 0
for L, N in ((3, 8193), (3, 8194), (3, 16642), (3, 110594), (8, 8199), (8, 110599), (3, 33003)):
    q = mkq(L, "peaky"); _, ok = rowexact(q, kb, vb, N, L); det += L - ok; tot += L
    print(f"  5b L={L} N={N} rows non-exact {L - ok}/{L}")
print(f"MUTATION 5b (B from call length, not row position): rows detected non-exact {det}/{tot} (straddling cases only expected)")
G._gqa31_rows = orig_rows
# 5c: B differs between two sides at the same position -> outputs differ (B is load-bearing for the numerics)
q = mkq(1, "peaky"); a = G._gqa31(q, kb[:, :, :131072], vb[:, :, :131072], sc, 1024); b = G._gqa31(q, kb[:, :, :131072], vb[:, :, :131072], sc, 512)
print(f"SENS 5c B=1024 vs B=512 same row differ: {not mx.array_equal(a, b).item()}  max|d|={mx.abs(a.astype(mx.float32)-b.astype(mx.float32)).max().item():.2e}")
