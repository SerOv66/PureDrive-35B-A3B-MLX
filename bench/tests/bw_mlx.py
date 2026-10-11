"""Effective memory bandwidth on this Mac as MLX sees it (roofline input, not a model benchmark).
- reduce: sum over a 2 GiB bf16 array (pure streaming read)
- qmv: quantized mat-vec (batch 1) at 6 and 8 bits, gs 64, i.e. the decode access pattern of the weights
- gather_qmm: MoE expert mat-vec (8 of 256 experts, 512x2048, 6-bit) like one MoE projection at decode
Median of 15 timed iterations after warm-up. Usage: python bench/bw_mlx.py"""
import time, statistics, mlx.core as mx

def timeit(fn, nbytes, iters=15):
    for _ in range(3): mx.eval(fn())
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    m = statistics.median(ts)
    return nbytes / m / 1e9, m * 1e3

a = mx.random.normal((1 << 30,)).astype(mx.bfloat16); mx.eval(a)
print("reduce bf16 2GiB      %.1f GB/s  (%.2f ms)" % timeit(lambda: a.sum(), a.nbytes))
del a

for bits in (6, 8):
    w = mx.random.normal((32768, 16384)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits); del w
    x = mx.random.normal((1, 16384)).astype(mx.bfloat16); mx.eval(wq, s, b, x)
    nb = wq.nbytes + s.nbytes + b.nbytes
    print("qmv %d-bit gs64 %.2fGB %.1f GB/s  (%.2f ms)" % ((bits, nb / 1e9) + timeit(
        lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits), nb)))
    del wq, s, b

# MoE: 40 layers x 256 experts x [512 x 2048] gate proj, 6-bit; route 8 random experts per layer
L, E, K = 40, 256, 8
w = mx.random.normal((E, 512, 2048)).astype(mx.bfloat16)
ws = [mx.quantize(w, group_size=64, bits=6) for _ in range(L)]; del w
x = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)
idx = [mx.random.randint(0, E, (1, K)).astype(mx.uint32) for _ in range(L)]
mx.eval(ws, x, idx)
per_exp = sum(t.nbytes for t in ws[0]) / E
nb = L * K * per_exp
def moe():
    return [mx.gather_qmm(x, q, s, b, rhs_indices=i, transpose=True, group_size=64, bits=6) for (q, s, b), i in zip(ws, idx)]
print("gather_qmm 6-bit 40x8 experts %.1f MB  %.1f GB/s  (%.3f ms)" % ((nb / 1e6,) + timeit(moe, nb)))
