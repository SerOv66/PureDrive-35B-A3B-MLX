"""Bitwise identity of the fused MoE (verify window M=3 and one-token M=1) across kernel geometries (C35_MOE_GU_ROWS,
C35_MOE_GU_NSG, C35_MOE_DOWN_ROWS): run once per geometry with the same seed, then --compare. Also checks that each row
of the M=3 window equals the one-token call on that row. Usage: [env] python bench/moe_geom_identity.py out.npz | --compare a b"""
import sys, os, argparse
import numpy as np
if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
    print(f"{len(a.files)} outputs: " + ("BITWISE IDENTICAL" if not bad else f"{len(bad)} DIFFER {bad[:6]}")); sys.exit(1 if bad else 0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h9_ops as h
import mlx.core as mx
a = argparse.Namespace(rows=3, union=16, moe_layers=2, reps=1, cmp_rows=0, bw=248.0, target_ms=19.7, dispatch=False, json=None, only=None, list=False)
c = h.make_ctx(a); h.arm(c.vq, True); out = {}; rowexact_bad = 0
for li in range(2):
    blk = h.build_moe(c, li)
    for seed in range(6):
        for union in (8, 14, 18, 24):
            c.union = union
            x = h.spikes(c, 3, union, 500 + seed * 10 + li)
            y = c.v._feed_forward(blk, x)
            out[f"l{li}_s{seed}_u{union}_m3"] = np.array(y.astype(mx.float32))
            for r in range(3):
                y1 = c.v._feed_forward(blk, x[:, r:r + 1])
                out[f"l{li}_s{seed}_u{union}_m1r{r}"] = np.array(y1.astype(mx.float32))
                if not mx.array_equal(y[:, r:r + 1], y1).item(): rowexact_bad += 1
np.savez(sys.argv[1], **out)
print(f"saved {len(out)}; row-exact violations (M=3 row vs one-token): {rowexact_bad}")
