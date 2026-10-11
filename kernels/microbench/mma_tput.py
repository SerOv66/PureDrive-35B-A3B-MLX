"""simdgroup_matrix MMA throughput on this GPU by operand type (bfloat / half / float), fp32 accumulators.
4 independent accumulator chains per simdgroup, 8 simdgroups per threadgroup, many threadgroups. Usage: python mma_tput.py"""
import time
import mlx.core as mx

SRC = r"""
    simdgroup_matrix<T, 8, 8> a, b;
    simdgroup_load(a, x + (thread_index_in_simdgroup % 2) * 64, 8);
    simdgroup_load(b, x + 128, 8);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < ITERS; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    simdgroup_float8x8 s;
    simdgroup_multiply_accumulate(s, a, b, c0);
    s.thread_elements()[0] += c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0];
    if (thread_index_in_simdgroup == 0) out[threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup] = s.thread_elements()[0];
"""
ITERS, TGS = 4096, 2048
k = mx.fast.metal_kernel(name="mma_tput", input_names=["x"], output_names=["out"], source=SRC, header=f"#define ITERS {ITERS}\n")
for dt in (mx.bfloat16, mx.float16, mx.float32):
    x = (mx.random.normal((256,)) * 0.01).astype(dt)
    run = lambda: k(inputs=[x], template=[("T", dt)], grid=(256 * TGS, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(TGS * 8,)], output_dtypes=[mx.float32])[0]
    mx.eval(run()); ts = []
    for _ in range(5):
        t = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t)
    flops = TGS * 8 * ITERS * 4 * 8 * 8 * 8 * 2
    print(f"{str(dt):16s} {flops / min(ts) / 1e12:6.2f} TFLOPS (MMA 8x8x8, fp32 acc)", flush=True)
