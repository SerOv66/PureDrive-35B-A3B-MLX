"""EXP-076 probes (M4 Pro), no model:
(1) resident threadgroups per core of a pass-1 attention kernel: atomic counter of running TGs (max over 5 launches / 20 cores);
(2) bf16 simdgroup MMA 8x8x8 latency vs throughput: dependent chains per simdgroup and simdgroups per core.
Usage: python occ_probe.py [N=131072] [L=3] [versions=16,28,29]"""
import sys, time
import mlx.core as mx
import gqa_attn as g

def timeit(run, reps=5):
    mx.eval(run()); ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t)
    return min(ts)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
vers = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "16,28,29").split(",")]
PRE = "    if (thread_index_in_threadgroup == 0) { device atomic_uint* cc = (device atomic_uint*)(cnt); uint now = atomic_fetch_add_explicit(cc, 1u, memory_order_relaxed) + 1u; atomic_fetch_max_explicit(cc + 1, now, memory_order_relaxed); }\n"
POST = "\n    if (thread_index_in_threadgroup == 0) { device atomic_uint* cc = (device atomic_uint*)(cnt); atomic_fetch_sub_explicit(cc, 1u, memory_order_relaxed); }\n"
HDR = f"#define ABL 0\n#define LROWS {L}\n#define B_SPLIT {g.B}\n#define T_TILE {g.T}\n#define D_DIM {g.D}\n#define HQ_N {g.HQ}\n#define G {g.G}\n#define KS (D_DIM + {g.PAD})\n#define SS (LROWS <= 4 ? (T_TILE + 8) : T_TILE)\n"
q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16)
k = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
nkv = mx.array([N], dtype=mx.int32); sc = mx.array([256 ** -0.5], dtype=mx.float32); ns = (N + g.B - 1) // g.B
print(f"(1) resident TGs/core, pass 1, N={N} L={L} ({2 * ns} TGs)")
for ver in vers:
    kern = mx.fast.metal_kernel(name=f"occ_{ver}_{L}", input_names=["q", "k", "v", "nkv", "sc", "cnt"], output_names=["po", "pm", "pl"],
                                header=HDR, source=PRE + getattr(g, f"_P1V{ver}") + POST, ensure_row_contiguous=False)
    best = 0
    for _ in range(5):
        cnt = mx.zeros((4096,), dtype=mx.uint32)   # >= 4 KB so MLX binds it as device memory (small inputs go to constant space)
        mx.eval(kern(inputs=[q, k, v, nkv, sc, cnt], template=[("T", mx.bfloat16)], grid=(256, ns, 2), threadgroup=(256, 1, 1),
                     output_shapes=[(ns, 16, L, 256), (ns, 16, L), (ns, 16, L)], output_dtypes=[mx.float32] * 3))
        best = max(best, cnt[1].item())
    print(f"  v{ver}: max {best} running TGs -> {best / 20:.1f} per core", flush=True)

MMA = r"""
    simdgroup_matrix<bfloat, 8, 8> a, b;
    simdgroup_load(a, x + (thread_index_in_simdgroup % 2) * 64, 8);
    simdgroup_load(b, x + 128, 8);
    simdgroup_float8x8 c[NCH];
    for (int j = 0; j < NCH; ++j) c[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (simdgroup_index_in_threadgroup < ACTIVE) {
        for (int i = 0; i < ITERS; ++i) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < NCH; ++j) simdgroup_multiply_accumulate(c[j], a, b, c[j]);
        }
    }
    float s = 0.0f;
    for (int j = 0; j < NCH; ++j) s += c[j].thread_elements()[0];
    if (thread_index_in_simdgroup == 0) out[threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup] = s;
"""
print("(2) MMA bf16 8x8x8, 256-thread TGs, ITERS=4096 dependent steps per chain")
x = (mx.random.normal((256,)) * 0.01).astype(mx.bfloat16)
for tgs, active, nch in ((20, 8, 1), (20, 8, 2), (20, 8, 4), (20, 6, 1), (40, 8, 1), (80, 8, 1), (160, 8, 1), (2048, 8, 1), (2048, 8, 4)):
    it = 4096
    kk = mx.fast.metal_kernel(name=f"mmal_{nch}_{active}", input_names=["x"], output_names=["out"], source=MMA, header=f"#define ITERS {it}\n#define NCH {nch}\n#define ACTIVE {active}\n")
    t = timeit(lambda: kk(inputs=[x], grid=(256 * tgs, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(tgs * 8,)], output_dtypes=[mx.float32])[0])
    print(f"  TGs {tgs:5d} ({tgs / 20:g}/core) active sg/TG {active} chains/sg {nch}: {t / it * 1e9:7.1f} ns per chain step, "
          f"{tgs * active * nch * it * 1024 / t / 1e12:5.2f} TFLOPS", flush=True)
