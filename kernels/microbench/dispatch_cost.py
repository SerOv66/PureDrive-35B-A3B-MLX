"""Per-dispatch cost on this GPU: chains of N dependent tiny ops in one graph, and chains of dependent 8-bit/6-bit
quantized GEMVs of decode-sized matrices (M=1 and M=3) -> us per op and effective GB/s. Usage: python dispatch_cost.py"""
import time, statistics
import mlx.core as mx

def t(fn, n, iters=20):
    for _ in range(3): mx.eval(fn())
    ts = []
    for _ in range(iters):
        s = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - s)
    return statistics.median(ts) / n * 1e6

x = mx.zeros((1,), dtype=mx.float32); mx.eval(x)
for n in (100, 400):
    def chain():
        y = x
        for _ in range(n): y = y + 1
        return y
    print(f"tiny add chain n={n}: {t(chain, n):.2f} us/op", flush=True)

for (K, N, bits, label) in ((2048, 512, 6, "expert gate/up 2048->512 6b"), (512, 2048, 6, "expert down 512->2048 6b"),
                            (2048, 12288, 8, "GDN in_proj 2048->12288 8b"), (4096, 2048, 8, "out_proj 4096->2048 8b"),
                            (2048, 1024, 8, "2048->1024 8b")):
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits); mx.eval(wq, s, b)
    nbytes = wq.nbytes + s.nbytes + b.nbytes
    for M in (1, 3):
        h = mx.random.normal((M, K)).astype(mx.bfloat16); mx.eval(h)
        n = 64
        ws = [(wq, s, b)] * n
        def chain():
            y = h
            for (a, sc, bi) in ws:
                o = mx.quantized_matmul(y, a, sc, bi, transpose=True, group_size=64, bits=bits)
                y = h + 0 * o[:, :1]          # dependent chain, same input size
            return y
        us = t(chain, n)
        print(f"{label:30s} M={M}: {us:7.2f} us/op  {nbytes / us / 1e3:6.1f} GB/s  ({nbytes/1e6:.2f} MB)", flush=True)
