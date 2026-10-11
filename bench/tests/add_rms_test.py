"""H16: fused residual add + RMSNorm for D <= 4096 (MLX rms_single_row transcription, 512 threads/row at D=2048) —
bitwise test against mx.add + mx.fast.rms_norm (bf16), rows 1..16. Usage: python bench/add_rms_test.py"""
import mlx.core as mx

SRC = r"""
    // (a + b, rms_norm(a + b) * w): MLX 0.32.3 rms_single_row arithmetic with the add folded into the load
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    float acc = 0;
    float tx[NR];
    const size_t base = size_t(gid) * D + lid * NR;
    for (int i = 0; i < NR; i++) {
        const T v = a[base + i] + b[base + i];
        s_out[base + i] = v;
        tx[i] = v;
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sgi == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sgi] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size[0] + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < NR; i++) n_out[base + i] = w[lid * NR + i] * static_cast<T>(tx[i] * local_inv_mean[0]);
"""
_K = {}

def add_rms(a, b, w, eps):
    D = a.shape[-1]; rows = a.size // D
    assert D % 128 == 0 and D <= 4096
    if "k" not in _K:
        _K["k"] = mx.fast.metal_kernel(name="c35_add_rms_single", input_names=["a", "b", "w", "eps", "axis_size"],
                                       output_names=["s_out", "n_out"], source=SRC)
    return _K["k"](inputs=[a, b, w, mx.array([eps], dtype=mx.float32), mx.array([D], dtype=mx.uint32)],
                   template=[("T", a.dtype), ("D", D)], grid=(D // 4 * rows, 1, 1), threadgroup=(D // 4, 1, 1),
                   output_shapes=[a.shape, a.shape], output_dtypes=[a.dtype, a.dtype])

if __name__ == "__main__":
    mx.random.seed(0); bad = 0; n = 0
    for trial in range(30):
        for rows in (1, 2, 3, 4, 8, 16):
            scale = [1e-3, 1.0, 30.0][trial % 3]
            a = (mx.random.normal((1, rows, 2048)) * scale).astype(mx.bfloat16)
            b = (mx.random.normal((1, rows, 2048)) * scale * 0.5).astype(mx.bfloat16)
            w = (mx.random.normal((2048,)) * 0.3 + 1.0).astype(mx.bfloat16)
            s_ref = a + b; n_ref = mx.fast.rms_norm(s_ref, w, 1e-6)
            s, nn = add_rms(a, b, w, 1e-6)
            n += 1
            if not (mx.array_equal(s, s_ref).item() and mx.array_equal(nn, n_ref).item()):
                bad += 1; print("DIFF rows", rows, "trial", trial, mx.abs(nn.astype(mx.float32) - n_ref.astype(mx.float32)).max().item())
    print(f"add_rms vs add+rms_norm: {n - bad}/{n} BITWISE IDENTICAL")
