# Error of the v31 attention vs fp32 exact as a function of the split size B (H31 numerics question): same q/k/v, L=3,
# synthetic logits with realistic spread (a few spiky keys). Reports mean / p99 / max |err| over all outputs.
import sys, numpy as np
sys.path.insert(0, __import__("os").environ.get("REPO_DIR", ".") + "/kernels")
import mlx.core as mx
import gqa_attn as g
L, D = 3, 256
for N in (32768, 131072):
    mx.random.seed(N)
    q = (mx.random.normal((1, 16, L, D)) * 1.0).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, D)) * 1.0).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, D)).astype(mx.bfloat16)
    spikes = mx.random.randint(0, N, (64,))
    k[:, :, spikes, :] = (k[:, :, spikes, :].astype(mx.float32) * 3).astype(mx.bfloat16)
    sc = 1 / 16.0   # 1/sqrt(256)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    qg = qf.reshape(1, 2, 8 * L, D)
    s = (qg @ kf.transpose(0, 1, 3, 2)) * sc
    # causal: row r sees keys < N - L + r + 1
    pos = mx.arange(N)[None, None, None, :]
    lim = (N - L + mx.tile(mx.arange(L), 8))[None, None, :, None]
    s = mx.where(pos <= lim, s, -mx.inf)
    ref = (mx.softmax(s, axis=-1) @ vf).reshape(1, 16, L, D)
    mx.eval(ref)
    for B in (256, 512, 832, 1024, 1648, 2048):
        out = g._gqa31(q, k, v, sc, B, True).astype(mx.float32)
        e = np.abs(np.array(out - ref)).ravel()
        r = np.abs(np.array(ref)).ravel()
        print(f"N {N:6d} B {B:4d}  mean|err| {e.mean():.3e}  p99 {np.percentile(e, 99):.3e}  max {e.max():.3e}  mean|ref| {r.mean():.3e}", flush=True)
