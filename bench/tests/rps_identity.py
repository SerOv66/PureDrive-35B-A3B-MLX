"""Bitwise identity of row_exact_qmv across tile widths (C35_RPS): run once per RPS value, then --compare.
Covers quantized_linear (rows 1..8) and quantized_linears groups for the checkpoint's formats.
Usage: C35_RPS=4 python bench/rps_identity.py <out.npz> ; python bench/rps_identity.py --compare a.npz b.npz"""
import sys, os
import numpy as np
if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
    print(f"{len(a.files)} outputs compared: " + ("BITWISE IDENTICAL" if not bad else f"{len(bad)} DIFFER: {bad[:8]}"))
    sys.exit(1 if bad else 0)
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "runtimes", "omlx"))
import mlx.core as mx
import mlx.nn as nn
from omlx.patches import row_exact_qmv as rq

mx.random.seed(0); out = {}
def qlin(k, n, bits, gs):
    lin = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    w = mx.random.normal((n, k)).astype(mx.bfloat16) * 0.05
    lin.weight, lin.scales, lin.biases = mx.quantize(w, group_size=gs, bits=bits)
    return lin
for (k, n, bits, gs) in ((2048, 8192, 8, 64), (2048, 4096, 6, 64), (4096, 2048, 6, 128), (4096, 2048, 8, 64), (2048, 512, 8, 64), (2048, 32, 8, 64)):
    lin = qlin(k, n, bits, gs)
    for rows in range(1, 9):
        x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
        y = rq.quantized_linear(lin, x)   # rows == 1 -> lin(x), the serial decode call
        out[f"lin_{k}_{n}_{bits}_{gs}_r{rows}"] = np.array(y.astype(mx.float32))
grp = [qlin(2048, n, 8, 64) for n in (8192, 4096, 32, 32)]
for rows in (2, 3, 4, 8):
    x = mx.random.normal((1, rows, 2048)).astype(mx.bfloat16)
    for i, y in enumerate(rq.quantized_linears(grp, x)): out[f"grp_r{rows}_{i}"] = np.array(y.astype(mx.float32))
np.savez(sys.argv[1], **out); print("saved", len(out), "RPS", rq._RPS)
