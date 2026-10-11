"""Row-exact GQA decode/verify attention for Qwen3.5-35B-A3B full-attention layers (Hq=16, Hkv=2, D=256), MLX custom Metal.

Flash-decoding with FIXED absolute key blocks (B keys per split, independent of L and of the KV length), so every
query row r is computed with exactly the same arithmetic whether it is a 1-row decode step or a row of a verify
window: row-exact by construction (greedy speculative output == non-speculative output).
Each threadgroup = (split, kv head): stages K/V tiles once in threadgroup memory and serves all 8 query heads of that
KV head and all L <= 8 rows -> K/V are read from DRAM once per step instead of once per (row, q-head).
Pass 2 merges the splits of each (head, row) in ascending order.

API: gqa_attention(q, k, v, scale) with q [1,16,L,256], k/v [1,2,N,256] (row-contiguous or a prefix slice of a
contiguous cache buffer); causal, bottom-right aligned (row r sees keys [0, N-L+r]). Returns [1,16,L,256] in q.dtype."""
import os
import mlx.core as mx

B = int(os.environ.get("C35_GQA_B", "256"))  # keys per split: FIXED absolute blocks (never derived from N or L)
T = int(os.environ.get("C35_GQA_T", "16"))   # keys per threadgroup-memory tile (fixed globally: numerics depend on it)
D = 256
HQ, HKV = 16, 2
G = HQ // HKV  # 8 query heads per kv head = 8 simdgroups per threadgroup
PAD = int(os.environ.get("C35_GQA_PAD", "8"))  # v21+: threadgroup row padding (elements) of the K/V tiles
P2V = int(os.environ.get("C35_GQA_P2", "1"))  # pass-2 merge kernel: 1 = _P2, 2 = _P2B, 3 = _P2C (same arithmetic, batched loads)
P2U = int(os.environ.get("C35_GQA_P2U", "16"))  # splits per batch in _P2C

_P1 = r"""
    // grid: threads (32*G, NSPLIT, HKV) ; threadgroup (32*G, 1, 1)
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;     // query head within the kv group
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const uint tid = thread_index_in_threadgroup;
    const int N = nkv[0];
    const int Lq = LROWS;
    const uint h = kvh * G + sg;

    threadgroup T_IN kt[T_TILE * D_DIM];
    threadgroup T_IN vt[T_TILE * D_DIM];

    // q rows for this head: lane owns dims [lane*8, lane*8+8)
    float qr[LROWS][8];
    for (int r = 0; r < Lq; ++r)
        for (int i = 0; i < 8; ++i)
            qr[r][i] = float(q[((h * Lq) + r) * D_DIM + lane * 8 + i]) * sc[0];

    float m[LROWS], l[LROWS], acc[LROWS][8];
    for (int r = 0; r < Lq; ++r) { m[r] = -INFINITY; l[r] = 0.0f; for (int i = 0; i < 8; ++i) acc[r][i] = 0.0f; }

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    // strides (elements) of k/v: [1, HKV, alloc, D]
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        // cooperative tile load: 16-byte vectors (8 x bf16), T_TILE*D/8 vectors per tensor, 256 threads
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (key < kend) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int tn = min(T_TILE, kend - t0);
        for (int kk = 0; kk < tn; ++kk) {
            const int key = t0 + kk;
            float kv8[8], vv8[8];
            for (int i = 0; i < 8; ++i) { kv8[i] = float(kt[kk * D_DIM + lane * 8 + i]); vv8[i] = float(vt[kk * D_DIM + lane * 8 + i]); }
            for (int r = 0; r < Lq; ++r) {
                if (key > N - Lq + r) continue;                       // causal, bottom-right aligned
                float s = 0.0f;
                for (int i = 0; i < 8; ++i) s += qr[r][i] * kv8[i];
                s = simd_sum(s);
                const float mn = max(m[r], s);
                const float a = metal::exp(m[r] - mn);
                const float p = metal::exp(s - mn);
                l[r] = l[r] * a + p;
                for (int i = 0; i < 8; ++i) acc[r][i] = acc[r][i] * a + p * vv8[i];
                m[r] = mn;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // partials: po [NSPLIT, HQ, L, D], pm/pl [NSPLIT, HQ, L]
    for (int r = 0; r < Lq; ++r) {
        const int64_t base = ((int64_t(split) * HQ_N + h) * Lq + r);
        for (int i = 0; i < 8; ++i) po[base * D_DIM + lane * 8 + i] = acc[r][i];
        if (lane == 0) { pm[base] = m[r]; pl[base] = l[r]; }
    }
"""


_P1V2 = r"""
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    P[tid * T_TILE + j] = T(pr);
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) P[tid * T_TILE + j] = T(0.0f);
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V3 = r"""
    // v3: 32-key tiles, K staged in threadgroup memory, V read with simdgroup_load from device, Q of one row kept in
    // registers per simdgroup, softmax one simdgroup per query row (lane = key) with simd_max/simd_sum.
    // Pairs (row r, key block kb of 8) : 4*L per tile; simdgroup sg owns rows r = sg % L (and its key blocks).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int TT = 32;

    threadgroup T kt[TT * D_DIM];                  // 16 KB
    threadgroup float S[8 * LROWS * TT];            // <= 8 KB
    threadgroup T P[8 * LROWS * TT];                // <= 4 KB
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    // pair assignment: total pairs NP = 4*L; simdgroup sg handles pairs p = sg, sg+8, ... ; with row r = p % L
    // Q tiles of the rows this simdgroup touches are reloaded per pair from device (L1-resident, 4 KB per row)
    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    const device T* vg = v + kvh * vs_h;

    for (int t0 = k0; t0 < kend; t0 += TT) {
        const int tn = min(TT, kend - t0);
        for (int e = tid; e < TT * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = (kk < tn)
                ? *((const device uint4*)(k + kvh * ks_h + (t0 + kk) * ks_n + d8)) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int pp = sg; pp < 4 * Lq; pp += G) {
            const int r = pp % Lq, kb = pp / Lq;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * TT + kb * 8, TT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool partial = (t0 + TT > kend);
        if (partial) {   // last partial tile: stage V in kt (free after the S phase), zero padded -> 0*finite = 0 exactly
            for (int e = tid; e < TT * D_DIM / 8; e += 32 * G) {
                const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = (kk < tn)
                    ? *((const device uint4*)(vg + (t0 + kk) * vs_n + d8)) : uint4(0);
            }
        }
        // softmax: simdgroup sg handles query rows qi = sg, sg+8, ...; lane j = key j of the tile
        for (int qi = sg; qi < NQ; qi += G) {
            const int r = qi >> 3;
            const int lim = N - Lq + r;
            const bool vis = (int(lane) < tn) && (t0 + int(lane) <= lim);
            const float sv = vis ? S[qi * TT + lane] * sc0 : -INFINITY;
            const float mold = ms[qi];
            const float mn = max(mold, simd_max(sv));
            float a = 1.0f, pr = 0.0f;
            if (mn != -INFINITY) { a = metal::exp(mold - mn); pr = vis ? metal::exp(sv - mn) : 0.0f; }
            P[qi * TT + lane] = T(pr);
            const float sum = simd_sum(pr);
            if (lane == 0) { ls[qi] = ls[qi] * a + sum; ms[qi] = mn; as_[qi] = a; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < 4; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * TT + kb * 8, TT);
                    if (partial) simdgroup_load(vm, kt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    else simdgroup_load(vm, vg + int64_t(t0 + kb * 8) * vs_n + sg * 32 + dt * 8, vs_n);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V4 = r"""
    // v4 = v2 + Q of a fixed row kept in registers per simdgroup + softmax on half-simdgroups (lane%16 = key).
    // S pairs per 16-key tile: (row r, key block kb in {0,1}) -> 2L pairs. Fixed assignment over the whole split:
    //   L>=4 : sg -> (r = sg % L', kb) where each simdgroup owns ONE row; L=8: sg owns row sg, both kb.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    // fixed S work of this simdgroup: row my_r, key blocks [kb0, kb1)
    int my_r, kb0, kb1;
    if (2 * LROWS <= G) {            // L <= 4: one (row, key block) pair per simdgroup, extra simdgroups idle
        my_r = int(sg) % LROWS; kb0 = int(sg) / LROWS; kb1 = (int(sg) < 2 * LROWS) ? kb0 + 1 : kb0;
    } else {                         // L = 5..8: simdgroup sg owns row sg (both key blocks), sg >= L idle
        my_r = min(int(sg), LROWS - 1); kb0 = 0; kb1 = (int(sg) < LROWS) ? 2 : 0;
    }
    simdgroup_matrix<T, 8, 8> qreg[D_DIM / 8];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + my_r * D_DIM + c * 8, Lq * D_DIM);

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kb = kb0; kb < kb1; ++kb) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kbt;
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qreg[c], kbt, acc);
            }
            simdgroup_store(acc, S + (my_r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // softmax: half-simdgroups, lane%16 = key, rows qi = 2*sg + (lane/16) + 16*j
        {
            const int hh = int(lane) >> 4, j16 = int(lane) & 15;
            for (int qi = 2 * int(sg) + hh; qi < NQ; qi += 2 * G) {
                const int r = qi >> 3;
                const int lim = N - Lq + r;
                const bool vis = (j16 < tn) && (t0 + j16 <= lim);
                const float sv = vis ? S[qi * T_TILE + j16] * sc0 : -INFINITY;
                float mx_ = sv;
                mx_ = max(mx_, simd_shuffle_xor(mx_, 8)); mx_ = max(mx_, simd_shuffle_xor(mx_, 4));
                mx_ = max(mx_, simd_shuffle_xor(mx_, 2)); mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
                const float mold = ms[qi];
                const float mn = max(mold, mx_);
                float a = 1.0f, pr = 0.0f;
                if (mn != -INFINITY) { a = metal::exp(mold - mn); pr = vis ? metal::exp(sv - mn) : 0.0f; }
                P[qi * T_TILE + j16] = T(pr);
                float sm = pr;
                sm += simd_shuffle_xor(sm, 8); sm += simd_shuffle_xor(sm, 4);
                sm += simd_shuffle_xor(sm, 2); sm += simd_shuffle_xor(sm, 1);
                if (j16 == 0) { ls[qi] = ls[qi] * a + sm; ms[qi] = mn; as_[qi] = a; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < 2; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V5 = r"""
    // v5 = v2 with fp32 probabilities and fp32 V in the P*V product (MLX sdpa_vector keeps p in fp32).
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup float vt[T_TILE * D_DIM];          // V tile converted to fp32 (16 KB)
    threadgroup float S[8 * LROWS * T_TILE];       // scores, then probabilities in place (fp32)
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                // bf16 -> fp32 is exact: the bf16 bits are the high half of the fp32 word
                const uint4 u = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
                threadgroup float4* vd = (threadgroup float4*)(vt + kk * D_DIM + d8);
                vd[0] = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u), as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
                vd[1] = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xFFFF0000u), as_type<float>(u.w << 16), as_type<float>(u.w & 0xFFFF0000u));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                ((threadgroup float4*)(vt + kk * D_DIM + d8))[0] = float4(0.0f); ((threadgroup float4*)(vt + kk * D_DIM + d8))[1] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < 2 * Lq; pp += G) {
            const int r = pp >> 1, kb = pp & 1;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    S[tid * T_TILE + j] = pr;
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) S[tid * T_TILE + j] = 0.0f;
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < 2; ++kb) {
                    simdgroup_float8x8 pm, vm;
                    simdgroup_load(pm, S + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V6 = r"""
    // v6 = v2 with probabilities split as P = P_hi + P_lo (two bf16 MMAs): ~fp32-accurate P*V, V stays bf16.
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup T Plo[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < 2 * Lq; pp += G) {
            const int r = pp >> 1, kb = pp & 1;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    const T ph = T(pr);
                    P[tid * T_TILE + j] = ph;
                    Plo[tid * T_TILE + j] = T(pr - float(ph));
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) { P[tid * T_TILE + j] = T(0.0f); Plo[tid * T_TILE + j] = T(0.0f); }
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < 2; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, pl2, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(pl2, Plo + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pl2, vm, O[r][dt]);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V7 = r"""
    // v7: v2 arithmetic (same S MMA, online softmax per query over 16-key tiles, bf16 P, same split merge) with
    //   - Q of the group staged ONCE per split in threadgroup memory when 8*L*D*2 bytes fit (L <= 4), else read from
    //     device (identical values -> identical results), removing the per-tile Q re-reads of v2;
    //   - V read directly from device in P*V with loop order (key block, dim tile) outer / rows inner (each V fragment
    //     loaded once per simdgroup); the last partial tile stages V into the K tile buffer (zero padded).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr bool QTG = (LROWS <= 4);

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T qt[QTG ? 8 * LROWS * D_DIM : 8];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;     // [8 heads][L][D] slice of this kv group
    const device T* vg = v + kvh * vs_h;
    if (QTG) {
        for (int e = tid; e < 8 * LROWS * D_DIM / 8; e += 32 * G)
            *((threadgroup uint4*)(qt + e * 8)) = *((const device uint4*)(qg + e * 8));
    }
    // Q rows (head i, row r) at qbase + (i*L + r)*D ; tile for row r = 8 heads with stride L*D
    const threadgroup T* qtg = qt;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = (kk < tn)
                ? *((const device uint4*)(k + kvh * ks_h + (t0 + kk) * ks_n + d8)) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int pp = sg; pp < 2 * Lq; pp += G) {
            const int r = pp >> 1, kb = pp & 1;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                if (QTG) simdgroup_load(qa, qtg + r * D_DIM + c * 8, Lq * D_DIM);
                else     simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool partial = (t0 + T_TILE > kend);
        if (partial) {   // stage V of the partial tile into kt (K no longer needed), zero padded
            for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
                const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = (kk < tn)
                    ? *((const device uint4*)(vg + (t0 + kk) * vs_n + d8)) : uint4(0);
            }
        }
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    P[tid * T_TILE + j] = T(pr);
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) P[tid * T_TILE + j] = T(0.0f);
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) { thread auto& el = O[r][dt].thread_elements(); el[0] *= ar; el[1] *= ar; }
        }
        for (int kb = 0; kb < 2; ++kb) {
            for (int dt = 0; dt < 4; ++dt) {
                simdgroup_matrix<T, 8, 8> vm;
                if (partial) simdgroup_load(vm, kt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                else         simdgroup_load(vm, vg + int64_t(t0 + kb * 8) * vs_n + sg * 32 + dt * 8, vs_n);
                for (int r = 0; r < LROWS; ++r) {
                    simdgroup_matrix<T, 8, 8> pm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V8 = r"""
    // v8 = v2 + the group's Q staged once per split in threadgroup memory when L <= 3 (identical values -> identical results).
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    constexpr bool QTG = (LROWS <= 3);
    threadgroup T qt[QTG ? 8 * LROWS * D_DIM : 8];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    if (QTG) {
        for (int e = tid; e < 8 * LROWS * D_DIM / 8; e += 32 * G)
            *((threadgroup uint4*)(qt + e * 8)) = *((const device uint4*)(qg + e * 8));
    }

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < 2 * Lq; pp += G) {
            const int r = pp >> 1, kb = pp & 1;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                if (QTG) simdgroup_load(qa, qt + r * D_DIM + c * 8, Lq * D_DIM);
                else     simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    P[tid * T_TILE + j] = T(pr);
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) P[tid * T_TILE + j] = T(0.0f);
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < 2; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V9 = r"""
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    // v9: the next tile's K/V are fetched into registers while the current tile is computed (same values, same math)
    constexpr int NV = T_TILE * D_DIM / 8 / (32 * G);
    uint4 rk[NV], rv[NV];
#define C35_FETCH(tt) for (int j = 0; j < NV; ++j) { const int e = int(tid) + j * 32 * G; const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8; const int key = (tt) + kk; \
        if (key < kend) { rk[j] = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8)); rv[j] = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8)); } \
        else { rk[j] = uint4(0); rv[j] = uint4(0); } }
    C35_FETCH(k0)
    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int j = 0; j < NV; ++j) {
            const int e = int(tid) + j * 32 * G; const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = rk[j];
            *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = rv[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + T_TILE < kend) C35_FETCH(t0 + T_TILE)
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    P[tid * T_TILE + j] = T(pr);
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) P[tid * T_TILE + j] = T(0.0f);
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V10 = r"""
    // v10: 64-key tiles, K and V read straight from device memory by the simdgroup that uses them (no staging),
    // simdgroup sg owns key block sg (8 keys) for S = Q K^T over all rows and dim slice [32 sg, 32 sg + 32) for O += P V;
    // online softmax per query by 4 threads (16 keys each, fixed reduction order); 2 barriers per tile.
    // Every per-row operation is independent of L -> row-exact by construction (same as v2, different arithmetic).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int TT = 64;

    threadgroup float S[8 * LROWS * TT];
    threadgroup T P[8 * LROWS * TT];
    threadgroup float as_[8 * LROWS];

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_n = k_strides[2], vs_n = v_strides[2];
    const device T* kg = k + int64_t(kvh) * k_strides[1];
    const device T* vg = v + int64_t(kvh) * v_strides[1];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    // softmax ownership: query sq = tid / 4 (row sq / 8), quarter sub = tid % 4 (keys [16 sub, 16 sub + 16) of a tile)
    const int sq = int(tid) >> 2, sub = int(tid) & 3;
    const bool sact = sq < NQ;
    const int lim = N - Lq + (sq >> 3);          // last key visible to this query's row
    float m_run = -INFINITY, l_run = 0.0f;

    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int t0 = k0; t0 < kend; t0 += TT) {
        const bool full = t0 + TT <= kend;
        const int kb0 = t0 + int(sg) * 8;         // this simdgroup's key block
        // ---- S = Q K^T for key block sg, all rows (dims in ascending order per accumulator)
        {
            simdgroup_float8x8 acc[LROWS];
            for (int r = 0; r < LROWS; ++r) acc[r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kbt;
                if (full) {
                    simdgroup_load(kbt, kg + int64_t(kb0) * ks_n + c * 8, ks_n, ulong2(0, 0), true);
                } else {
                    thread auto& ke = kbt.thread_elements();
                    const int ka = kb0 + fn, kc = kb0 + fn + 1;
                    ke[0] = ka < kend ? kg[int64_t(ka) * ks_n + c * 8 + fm] : T(0);
                    ke[1] = kc < kend ? kg[int64_t(kc) * ks_n + c * 8 + fm] : T(0);
                }
                for (int r = 0; r < LROWS; ++r) {
                    simdgroup_matrix<T, 8, 8> qa;
                    simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                    simdgroup_multiply_accumulate(acc[r], qa, kbt, acc[r]);
                }
            }
            for (int r = 0; r < LROWS; ++r) simdgroup_store(acc[r], S + (r * 8) * TT + sg * 8, TT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- online softmax (per query, 4 threads x 16 keys, fixed shuffle-reduction order)
        if (sact) {
            const threadgroup float* srow = S + sq * TT + sub * 16;
            const int jb = t0 + sub * 16;
            float mx_ = -INFINITY;
            for (int j = 0; j < 16; ++j) if (jb + j <= lim && jb + j < kend) mx_ = max(mx_, srow[j] * sc0);
            mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
            mx_ = max(mx_, simd_shuffle_xor(mx_, 2));
            const float mn = max(m_run, mx_);
            float sum = 0.0f;
            threadgroup T* prow = P + sq * TT + sub * 16;
            if (mn == -INFINITY) {
                for (int j = 0; j < 16; ++j) prow[j] = T(0.0f);
                if (sub == 0) as_[sq] = 1.0f;
            } else {
                for (int j = 0; j < 16; ++j) {
                    float pr = 0.0f;
                    if (jb + j <= lim && jb + j < kend) pr = metal::exp(srow[j] * sc0 - mn);
                    prow[j] = T(pr);
                    sum += pr;
                }
                const float a = metal::exp(m_run - mn);
                sum += simd_shuffle_xor(sum, 1);
                sum += simd_shuffle_xor(sum, 2);
                l_run = l_run * a + sum; m_run = mn;
                if (sub == 0) as_[sq] = a;
            }
        } else {
            // keep the shuffles convergent: inactive threads of an active simdgroup still participate
            float z = 0.0f;
            z = max(z, simd_shuffle_xor(z, 1)); z = max(z, simd_shuffle_xor(z, 2));
            z += simd_shuffle_xor(z, 1); z += simd_shuffle_xor(z, 2);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- O = diag(a) O + P V on this simdgroup's 32-dim slice (key blocks in ascending order per accumulator)
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) { thread auto& el = O[r][dt].thread_elements(); el[0] *= ar; el[1] *= ar; }
        }
        for (int kb = 0; kb < TT / 8; ++kb) {
            const int kv0 = t0 + kb * 8;
            if (!full && kv0 >= kend) break;
            simdgroup_matrix<T, 8, 8> pm[LROWS];
            for (int r = 0; r < LROWS; ++r) simdgroup_load(pm[r], P + (r * 8) * TT + kb * 8, TT);
            for (int dt = 0; dt < 4; ++dt) {
                simdgroup_matrix<T, 8, 8> vm;
                const int d0 = int(sg) * 32 + dt * 8;
                if (full || kv0 + 8 <= kend) {
                    simdgroup_load(vm, vg + int64_t(kv0) * vs_n + d0, vs_n);
                } else {
                    thread auto& ve = vm.thread_elements();
                    const bool ok = kv0 + fm < kend;
                    ve[0] = ok ? vg[int64_t(kv0 + fm) * vs_n + d0 + fn] : T(0);
                    ve[1] = ok ? vg[int64_t(kv0 + fm) * vs_n + d0 + fn + 1] : T(0);
                }
                for (int r = 0; r < LROWS; ++r) simdgroup_multiply_accumulate(O[r][dt], pm[r], vm, O[r][dt]);
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (sact && sub == 0) {
        const int h = int(kvh) * 8 + (sq & 7), r = sq >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = m_run; pl[base] = l_run;
    }
"""

_P1V11 = r"""
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, one thread per query, keys in ascending order
        if (tid < uint(NQ)) {
            const int r = int(tid) >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[tid];
            float mn = mold;
            for (int j = 0; j < tn; ++j) if (t0 + j <= lim) mn = max(mn, S[tid * T_TILE + j] * sc0);
            float a = 1.0f, sum = 0.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int j = 0; j < T_TILE; ++j) {
                    float pr = 0.0f;
                    if (j < tn && t0 + j <= lim) pr = metal::exp(S[tid * T_TILE + j] * sc0 - mn);
                    P[tid * T_TILE + j] = T(pr);
                    sum += pr;
                }
            } else {
                for (int j = 0; j < T_TILE; ++j) P[tid * T_TILE + j] = T(0.0f);
            }
            ls[tid] = ls[tid] * a + sum; ms[tid] = mn; as_[tid] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        // v11: same per-accumulator order as v2 (scale, then kb ascending); V block loaded once for all rows
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) { thread auto& el = O[r][dt].thread_elements(); el[0] *= ar; el[1] *= ar; }
        }
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < KB; ++kb) {
            simdgroup_matrix<T, 8, 8> pm[LROWS];
            #pragma clang loop unroll(full)
            for (int r = 0; r < LROWS; ++r) simdgroup_load(pm[r], P + (r * 8) * T_TILE + kb * 8, T_TILE);
            #pragma clang loop unroll(full)
            for (int dt = 0; dt < 4; ++dt) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                #pragma clang loop unroll(full)
                for (int r = 0; r < LROWS; ++r) simdgroup_multiply_accumulate(O[r][dt], pm[r], vm, O[r][dt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V12 = r"""
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V13 = r"""
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V14 = r"""
    // v14 = v13 + the S-phase Q block of each simdgroup kept in registers when every simdgroup owns <= 1 (row, key block) pair (KB*L <= 8)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    constexpr bool QREG = KB * LROWS <= G;
    simdgroup_matrix<T, 8, 8> qreg[QREG ? D_DIM / 8 : 1];
    if (QREG && int(sg) < KB * Lq) {
        const int r = int(sg) / KB;
        for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + r * D_DIM + c * 8, Lq * D_DIM);
    }
    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        if (QREG) {
            if (int(sg) < KB * Lq) {
                const int r = int(sg) / KB, kb = int(sg) % KB;
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (int c = 0; c < D_DIM / 8; ++c) {
                    simdgroup_matrix<T, 8, 8> kbt;
                    simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, qreg[c], kbt, acc);
                }
                simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
            }
        } else
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V15 = r"""
    // v15 = v13 with phase ablations for profiling only (C35_GQA_ABL: 1 no S MMA, 2 no softmax, 3 no PV MMA, 4 no K/V load). NOT exact.
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        if (ABL != 4) for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        if (ABL != 1) for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (ABL != 2 && tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                if (ABL != 3) for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V16 = r"""
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V31 = r"""
    // v31 (H6-ter, EXP-088/091): S phase re-tiled for operand reuse. sg -> (8-key block sg&1, D quarter sg>>1): one K-block
    // load feeds the LROWS chains; quarter partials summed by the softmax in fixed order ((p0+p1)+p2)+p3 -> row-exact, new
    // numerics vs v16. KDEV/QREG/DBUF change only staging/registers (bitwise identical to each other).
    static_assert(T_TILE == 16, "v31 maps 2 key blocks x 4 D-quarters onto the 8 simdgroups");
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int SPN = 8 * LROWS * T_TILE;   // one D-quarter partial of S (fp32)

#if DBUF
    threadgroup T vt2[2 * T_TILE * D_DIM];
#else
    threadgroup T vt2[T_TILE * D_DIM];
#endif
#if KDEV
    threadgroup T ktl[8 * D_DIM];             // the (single) partial 8-key block of the last tile, zero-filled
#else
    threadgroup T kt[T_TILE * D_DIM];
#endif
    threadgroup float Sp[4 * SPN];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    #pragma clang loop unroll(full)
    for (int r = 0; r < LROWS; ++r)
        #pragma clang loop unroll(full)
        for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    const int kbm = int(sg) & 1, dq = int(sg) >> 1;

#if QREG
    simdgroup_matrix<T, 8, 8> qf[LROWS][8];
    #pragma clang loop unroll(full)
    for (int r = 0; r < LROWS; ++r)
        #pragma clang loop unroll(full)
        for (int c = 0; c < 8; ++c) simdgroup_load(qf[r][c], qg + r * D_DIM + (dq * 8 + c) * 8, Lq * D_DIM);
#define QFRAG(r, c, cc) qf[r][c]
#else
#define QFRAG(r, c, cc) qa
#endif

// one D-quarter of S for key block kbm: one K load per chunk feeds the LROWS chains
#define S_CHAIN(KPTR, KSTRIDE)                                                                       \
    {                                                                                                \
        simdgroup_float8x8 acc[LROWS];                                                               \
        _Pragma("clang loop unroll(full)")                                                           \
        for (int r = 0; r < LROWS; ++r) acc[r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);    \
        _Pragma("clang loop unroll(full)")                                                           \
        for (int c = 0; c < 8; ++c) {                                                                \
            const int cc = dq * 8 + c;                                                               \
            simdgroup_matrix<T, 8, 8> kbt;                                                           \
            simdgroup_load(kbt, (KPTR) + cc * 8, (KSTRIDE), ulong2(0, 0), true);                     \
            _Pragma("clang loop unroll(full)")                                                       \
            for (int r = 0; r < LROWS; ++r) {                                                        \
                simdgroup_matrix<T, 8, 8> qa;                                                        \
                if (!QREG) simdgroup_load(qa, qg + r * D_DIM + cc * 8, Lq * D_DIM);                  \
                simdgroup_multiply_accumulate(acc[r], QFRAG(r, c, cc), kbt, acc[r]);                 \
            }                                                                                        \
        }                                                                                            \
        _Pragma("clang loop unroll(full)")                                                           \
        for (int r = 0; r < LROWS; ++r) simdgroup_store(acc[r], Sp + dq * SPN + (r * 8) * T_TILE + kbm * 8, T_TILE); \
    }

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
#if DBUF
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
#else
        threadgroup T* vt = vt2;
#endif
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
#if !KDEV
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
#endif
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
#if !KDEV
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
#endif
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
#if KDEV
        if (tn & 7) {   // only the last tile of the last split can be partial
            const int kp = tn & ~7;
            for (int e = tid; e < D_DIM; e += 32 * G) {
                const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
                *((threadgroup uint4*)(ktl + kk * D_DIM + d8)) =
                    (kp + kk < tn) ? *((const device uint4*)(k + kvh * ks_h + int64_t(t0 + kp + kk) * ks_n + d8)) : uint4(0);
            }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S quarters (blocks with no valid key are skipped: the softmax never reads them)
        {
            const int nv = tn - kbm * 8;
#if KDEV
            if (nv >= 8) S_CHAIN(k + kvh * ks_h + int64_t(t0 + kbm * 8) * ks_n, ulong(ks_n))
            else if (nv > 0) S_CHAIN(ktl, D_DIM)
#else
            if (nv > 0) S_CHAIN(kt + (kbm * 8) * D_DIM, D_DIM)
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // softmax as v16, on S = ((p0 + p1) + p2) + p3
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float sv[KPT];
            float mloc = -INFINITY;
            static_assert(KPT == 4, "float4 partial reads");
            const int o4 = qq * T_TILE + sub * KPT;
            const float4 s4 = ((*((threadgroup float4*)(Sp + o4)) + *((threadgroup float4*)(Sp + SPN + o4)))
                               + *((threadgroup float4*)(Sp + 2 * SPN + o4))) + *((threadgroup float4*)(Sp + 3 * SPN + o4));
            #pragma clang loop unroll(full)
            for (int i = 0; i < KPT; ++i) {
                const int jj = sub * KPT + i;
                sv[i] = 0.0f;
                if (jj < tn && t0 + jj <= lim) { sv[i] = s4[i]; mloc = max(mloc, sv[i] * sc0); }
            }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                #pragma clang loop unroll(full)
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(sv[i] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    Sp[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); Sp[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += Sp[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V (as v16)
        #pragma clang loop unroll(full)
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            #pragma clang loop unroll(full)
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
#if !DBUF
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }
    #pragma clang loop unroll(full)
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        #pragma clang loop unroll(full)
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V27 = r"""
    // v27 = v16 software-pipelined: tile t+1 is staged (K into the K tile, V into the other V buffer) by the threads that
    //       have no softmax work (tid >= 4*NQ; all threads after their softmax when 4*NQ >= 32*G) during the softmax of
    //       tile t -> 2 barriers per tile, loads overlap the softmax. Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    // stage tile at key u0 (K -> kt, V -> vt2[parity]) with threads [sbase, 32*G)
    #define C35_STAGE(u0, sbase) { \
        const int un = min(T_TILE, kend - (u0)); \
        threadgroup T* vdst = vt2 + ((((u0) - k0) / T_TILE) & 1) * (T_TILE * D_DIM); \
        for (int e = int(tid) - (sbase); e < T_TILE * D_DIM / 8; e += 32 * G - (sbase)) { \
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8; \
            const int key = (u0) + kk; \
            if (kk < un) { \
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8)); \
                *((threadgroup uint4*)(vdst + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8)); \
            } else { \
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0); \
                *((threadgroup uint4*)(vdst + kk * D_DIM + d8)) = uint4(0); \
            } \
        } }
    if (k0 < kend) C35_STAGE(k0, 0)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        if (t0 + T_TILE < kend) {   // stage the next tile while the softmax runs (kt is free: the S phase is done)
            if (4 * NQ < 32 * G) { if (tid >= uint(4 * NQ)) C35_STAGE(t0 + T_TILE, 4 * NQ) }
            else C35_STAGE(t0 + T_TILE, 0)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    #undef C35_STAGE
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V28 = r"""
    // v28 = v16 software-pipelined across tiles (EXP-076). The S chain (32 dependent MMAs) is latency-bound and at L=3
    //       ~3 TGs/core give too few chains to fill the MMA pipes, so phase X(i) runs S(i+1) with PV(i) interleaved in the
    //       same unrolled block (PV MMAs fill the S-chain bubbles); phase Y(i) runs softmax(i+1) while the K(i+2)/V(i+1)
    //       tiles, loaded into registers at the start of X(i), are stored to threadgroup memory. 2 barriers per tile, single
    //       K/V tiles (18 KB). Every accumulator sees the same operations in the same order as v16 -> bitwise identical.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;                    // 8-key blocks per tile
    constexpr int NPV = LROWS * 4 * KB;               // PV MMAs per simdgroup per tile
    constexpr int NST = T_TILE * D_DIM / 8 / (32 * G); // uint4 per thread per tile (K or V)

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int ntile = (kend - k0 + T_TILE - 1) / T_TILE;
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    const device T* kg = k + kvh * ks_h;
    const device T* vg = v + kvh * vs_h;

    // register prefetch of one K or V tile (zero rows past the split end, as v16)
    uint4 kr[NST], vr[NST];
    #define C35_FETCH(dst, base, str, ti) { const int f0 = k0 + (ti) * T_TILE; const int fn_ = min(T_TILE, kend - f0); \
        for (int j = 0; j < NST; ++j) { const int e = int(tid) + j * 32 * G; const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8; \
            dst[j] = kk < fn_ ? *((const device uint4*)(base + (f0 + kk) * str + d8)) : uint4(0); } }
    #define C35_STORE(tile, src) { for (int j = 0; j < NST; ++j) { const int e = int(tid) + j * 32 * G; \
            *((threadgroup uint4*)(tile + (e / (D_DIM / 8)) * D_DIM + (e % (D_DIM / 8)) * 8)) = src[j]; } }

    // S chain(s) of tile ti into S (plain: used in the prologue and for the extra chains when KB*Lq > G)
    #define C35_S_PLAIN(pp0) { for (int pp = (pp0); pp < KB * Lq; pp += G) { const int r = pp / KB, kb = pp % KB; \
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
            _Pragma("clang loop unroll(full)") for (int c = 0; c < D_DIM / 8; ++c) { simdgroup_matrix<T, 8, 8> qa, kbt; \
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM); simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true); \
                simdgroup_multiply_accumulate(acc, qa, kbt, acc); } \
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE); } }

    // online softmax of tile ti (v12 scheme, bitwise = v2): 4 threads per query
    #define C35_SOFTMAX(ti) if (tid < uint(4 * NQ)) { const int t0 = k0 + (ti) * T_TILE; const int tn = min(T_TILE, kend - t0); \
            const int qq = int(tid) >> 2, sub = int(tid) & 3; constexpr int KPT = T_TILE / 4; const int r = qq >> 3; const int lim = N - Lq + r; \
            const float mold = ms[qq]; float mloc = -INFINITY; \
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); } \
            mloc = max(mloc, simd_shuffle_xor(mloc, 1)); mloc = max(mloc, simd_shuffle_xor(mloc, 2)); \
            const float mn = max(mold, mloc); float a = 1.0f; \
            if (mn != -INFINITY) { a = metal::exp(mold - mn); \
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; float pr = 0.0f; \
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn); \
                    P[qq * T_TILE + jj] = T(pr); S[qq * T_TILE + jj] = pr; } \
            } else { for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; } } \
            simdgroup_barrier(mem_flags::mem_threadgroup); \
            if (sub == 0) { float sum = 0.0f; for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj]; \
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a; } }

    // prologue: K(0), V(0) staged; S(0); softmax(0) while K(1) is stored
    C35_FETCH(kr, kg, ks_n, 0); C35_FETCH(vr, vg, vs_n, 0);
    C35_STORE(kt, kr); C35_STORE(vt, vr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ntile > 1) C35_FETCH(kr, kg, ks_n, 1);
    C35_S_PLAIN(int(sg));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    C35_SOFTMAX(0);
    if (ntile > 1) C35_STORE(kt, kr);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < ntile; ++i) {
        const bool nxt = i + 1 < ntile;
        if (i + 2 < ntile) C35_FETCH(kr, kg, ks_n, i + 2);
        if (nxt) C35_FETCH(vr, vg, vs_n, i + 1);
        // X(i): PV(i) for this simdgroup's 32-dim slice, interleaved with its S(i+1) chain (if any)
        float ar[LROWS];   // all loops touching O[][] fully unrolled: a dynamic index sends O to the stack (2x slower)
        #pragma clang loop unroll(full)
        for (int r = 0; r < LROWS; ++r) ar[r] = as_[r * 8 + fm];
        #pragma clang loop unroll(full)
        for (int r = 0; r < LROWS; ++r)
            #pragma clang loop unroll(full)
            for (int dt = 0; dt < 4; ++dt) { thread auto& el = O[r][dt].thread_elements(); el[0] *= ar[r]; el[1] *= ar[r]; }
        if (nxt && int(sg) < KB * Lq) {
            const int r0 = int(sg) / KB, kb0 = int(sg) % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r0 * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb0 * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
                #pragma clang loop unroll(full)
                for (int j = c * NPV / (D_DIM / 8); j < (c + 1) * NPV / (D_DIM / 8); ++j) {   // PV MMA j: kb-major, so consecutive MMAs are independent
                    const int kb = j / (LROWS * 4), r = (j / 4) % LROWS, dt = j % 4;
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
            simdgroup_store(acc, S + (r0 * 8) * T_TILE + kb0 * 8, T_TILE);
        } else {
            #pragma clang loop unroll(full)
            for (int j = 0; j < NPV; ++j) {
                const int kb = j / (LROWS * 4), r = (j / 4) % LROWS, dt = j % 4;
                simdgroup_matrix<T, 8, 8> pm, vm;
                simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
            }
        }
        if (nxt) C35_S_PLAIN(int(sg) + G);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Y(i): softmax(i+1); K(i+2), V(i+1) to threadgroup memory
        if (nxt) {
            C35_SOFTMAX(i + 1);
            if (i + 2 < ntile) C35_STORE(kt, kr);
            C35_STORE(vt, vr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #undef C35_FETCH
    #undef C35_STORE
    #undef C35_S_PLAIN
    #undef C35_SOFTMAX
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V29 = r"""
    // v29 = v16 with the Q fragments of the S chain kept in registers across tiles (EXP-076); same math -> bitwise identical
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
    // v29: the (row, key block) pair of a simdgroup's first S chain is the same for every tile -> its 32 Q fragments
    //      are loaded once into registers (fully unrolled: a dynamic index would put them on the stack, cf. v14)
    simdgroup_matrix<T, 8, 8> qreg[D_DIM / 8];
    {
        const int r = min(int(sg) / KB, Lq - 1);
        #pragma clang loop unroll(full)
        for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + r * D_DIM + c * 8, Lq * D_DIM);
    }

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        if (int(sg) < KB * Lq) {   // first chain of this simdgroup: Q fragments from registers
            const int r = int(sg) / KB, kb = int(sg) % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kbt;
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qreg[c], kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        for (int pp = int(sg) + G; pp < KB * Lq; pp += G) {   // extra chains (L > 4): as v16
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        #pragma clang loop unroll(full)
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            #pragma clang loop unroll(full)
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V30 = r"""
    // v30 = v16 with the S phase organised by key block (EXP-076); same MMA sequence per accumulator -> bitwise identical
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        // v30: simdgroup kb runs the chains of all rows for key block kb, interleaved: one transposed K load feeds LROWS MMAs
        for (int kb = sg; kb < KB; kb += G) {
            simdgroup_float8x8 acc[LROWS];
            #pragma clang loop unroll(full)
            for (int r = 0; r < LROWS; ++r) acc[r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kbt;
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int r = 0; r < LROWS; ++r) {
                    simdgroup_matrix<T, 8, 8> qa;
                    simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                    simdgroup_multiply_accumulate(acc[r], qa, kbt, acc[r]);
                }
            }
            #pragma clang loop unroll(full)
            for (int r = 0; r < LROWS; ++r) simdgroup_store(acc[r], S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V26 = r"""
    // v26 = v16 with V NOT staged: the P.V phase loads each simdgroup's 32-dim slice of V straight from device memory
    //       (every V element is read by exactly one simdgroup), threadgroup memory 27 KB -> ~10.5 KB. A partial last
    //       tile stages V (zero padded) into the free K tile after the S phase. Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        const bool partial = tn < T_TILE;
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (partial) {   // last tile only: K no longer needed, stage V zero padded (0 * finite = 0 exactly)
            for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
                const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = kk < tn ? *((const device uint4*)(v + kvh * vs_h + (t0 + kk) * vs_n + d8)) : uint4(0);
            }
        }
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    if (partial) simdgroup_load(vm, kt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    else simdgroup_load(vm, v + kvh * vs_h + int64_t(t0 + kb * 8) * vs_n + sg * 32 + dt * 8, vs_n);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V21 = r"""
    // v21 = v16 with K/V threadgroup rows padded to KS = D_DIM + PAD elements (row stride 512 B -> 528 B: the 8 rows of a
    //       transposed K / V simdgroup_load hit different banks). Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * KS];
    threadgroup T vt2[2 * T_TILE * KS];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * KS);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * KS + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * KS + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * KS + sg * 32 + dt * 8, KS);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V22 = r"""
    // v22 = v21 with the S (fp32) and P (T) threadgroup rows padded to T_TILE + 8 (stride 64 B -> 96 B, 32 B -> 48 B):
    //       fewer bank conflicts in the S store, softmax reads/sum and P loads. Same math as v16 -> bitwise identical.
    // v21 = v16 with K/V threadgroup rows padded to KS = D_DIM + PAD elements (row stride 512 B -> 528 B: the 8 rows of a
    //       transposed K / V simdgroup_load hit different banks). Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * KS];
    threadgroup T vt2[2 * T_TILE * KS];
    threadgroup float S[8 * LROWS * SS];
    threadgroup T P[8 * LROWS * SS];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * KS);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * KS + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * KS + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * SS + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * SS + jj] * sc0 - mn);
                    P[qq * SS + jj] = T(pr);
                    S[qq * SS + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * SS + jj] = T(0.0f); S[qq * SS + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * SS + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * SS + kb * 8, SS);
                    simdgroup_load(vm, vt + (kb * 8) * KS + sg * 32 + dt * 8, KS);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V23 = r"""
    // v23 = v22 with the Q tiles of the simdgroup's fixed S row kept in registers across all key tiles when L*KB <= G
    //       (one (row, key block) pair per simdgroup; loop fully unrolled -> static indices). Same math -> bitwise identical.
    // v22 = v21 with the S (fp32) and P (T) threadgroup rows padded to T_TILE + 8 (stride 64 B -> 96 B, 32 B -> 48 B):
    //       fewer bank conflicts in the S store, softmax reads/sum and P loads. Same math as v16 -> bitwise identical.
    // v21 = v16 with K/V threadgroup rows padded to KS = D_DIM + PAD elements (row stride 512 B -> 528 B: the 8 rows of a
    //       transposed K / V simdgroup_load hit different banks). Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * KS];
    threadgroup T vt2[2 * T_TILE * KS];
    threadgroup float S[8 * LROWS * SS];
    threadgroup T P[8 * LROWS * SS];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
#if LROWS * (T_TILE / 8) <= G
    simdgroup_matrix<T, 8, 8> qreg[D_DIM / 8];
    if (int(sg) < KB * Lq) {
        const int rq = int(sg) / KB;
        #pragma clang loop unroll(full)
        for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + rq * D_DIM + c * 8, Lq * D_DIM);
    }
#endif

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * KS);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * KS + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * KS + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
#if LROWS * (T_TILE / 8) <= G
                qa = qreg[c];
#else
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
#endif
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * SS + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * SS + jj] * sc0 - mn);
                    P[qq * SS + jj] = T(pr);
                    S[qq * SS + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * SS + jj] = T(0.0f); S[qq * SS + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * SS + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * SS + kb * 8, SS);
                    simdgroup_load(vm, vt + (kb * 8) * KS + sg * 32 + dt * 8, KS);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V24 = r"""
    // v24 = v23 computing S^T = K Q^T (K tile loaded NOT transposed as left operand, Q^T kept in registers) and storing it
    //       transposed -> same S layout. Same products and order per element; bitwise identity checked by --same-as.
    // v23 = v22 with the Q tiles of the simdgroup's fixed S row kept in registers across all key tiles when L*KB <= G
    //       (one (row, key block) pair per simdgroup; loop fully unrolled -> static indices). Same math -> bitwise identical.
    // v22 = v21 with the S (fp32) and P (T) threadgroup rows padded to T_TILE + 8 (stride 64 B -> 96 B, 32 B -> 48 B):
    //       fewer bank conflicts in the S store, softmax reads/sum and P loads. Same math as v16 -> bitwise identical.
    // v21 = v16 with K/V threadgroup rows padded to KS = D_DIM + PAD elements (row stride 512 B -> 528 B: the 8 rows of a
    //       transposed K / V simdgroup_load hit different banks). Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * KS];
    threadgroup T vt2[2 * T_TILE * KS];
    threadgroup float S[8 * LROWS * SS];
    threadgroup T P[8 * LROWS * SS];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
#if LROWS * (T_TILE / 8) <= G
    simdgroup_matrix<T, 8, 8> qreg[D_DIM / 8];
    if (int(sg) < KB * Lq) {
        const int rq = int(sg) / KB;
        #pragma clang loop unroll(full)
        for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + rq * D_DIM + c * 8, Lq * D_DIM, ulong2(0, 0), true);
    }
#endif

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * KS);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * KS + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * KS + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
#if LROWS * (T_TILE / 8) <= G
                qa = qreg[c];
#else
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
#endif
#if LROWS * (T_TILE / 8) <= G
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS);
                simdgroup_multiply_accumulate(acc, kbt, qa, acc);
#else
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
#endif
            }
#if LROWS * (T_TILE / 8) <= G
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS, ulong2(0, 0), true);
#else
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * SS + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * SS + jj] * sc0 - mn);
                    P[qq * SS + jj] = T(pr);
                    S[qq * SS + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * SS + jj] = T(0.0f); S[qq * SS + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * SS + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * SS + kb * 8, SS);
                    simdgroup_load(vm, vt + (kb * 8) * KS + sg * 32 + dt * 8, KS);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V25 = r"""
    // v25 = v24 with V single-buffered (+1 barrier per tile, -8.4 KB threadgroup memory: occupancy test). Same math.
    // v24 = v23 computing S^T = K Q^T (K tile loaded NOT transposed as left operand, Q^T kept in registers) and storing it
    //       transposed -> same S layout. Same products and order per element; bitwise identity checked by --same-as.
    // v23 = v22 with the Q tiles of the simdgroup's fixed S row kept in registers across all key tiles when L*KB <= G
    //       (one (row, key block) pair per simdgroup; loop fully unrolled -> static indices). Same math -> bitwise identical.
    // v22 = v21 with the S (fp32) and P (T) threadgroup rows padded to T_TILE + 8 (stride 64 B -> 96 B, 32 B -> 48 B):
    //       fewer bank conflicts in the S store, softmax reads/sum and P loads. Same math as v16 -> bitwise identical.
    // v21 = v16 with K/V threadgroup rows padded to KS = D_DIM + PAD elements (row stride 512 B -> 528 B: the 8 rows of a
    //       transposed K / V simdgroup_load hit different banks). Same math as v16 -> bitwise identical.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * KS];
    threadgroup T vt[T_TILE * KS];
    threadgroup float S[8 * LROWS * SS];
    threadgroup T P[8 * LROWS * SS];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;
#if LROWS * (T_TILE / 8) <= G
    simdgroup_matrix<T, 8, 8> qreg[D_DIM / 8];
    if (int(sg) < KB * Lq) {
        const int rq = int(sg) / KB;
        #pragma clang loop unroll(full)
        for (int c = 0; c < D_DIM / 8; ++c) simdgroup_load(qreg[c], qg + rq * D_DIM + c * 8, Lq * D_DIM, ulong2(0, 0), true);
    }
#endif

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * KS + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * KS + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * KS + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
#if LROWS * (T_TILE / 8) <= G
                qa = qreg[c];
#else
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
#endif
#if LROWS * (T_TILE / 8) <= G
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS);
                simdgroup_multiply_accumulate(acc, kbt, qa, acc);
#else
                simdgroup_load(kbt, kt + (kb * 8) * KS + c * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
#endif
            }
#if LROWS * (T_TILE / 8) <= G
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS, ulong2(0, 0), true);
#else
            simdgroup_store(acc, S + (r * 8) * SS + kb * 8, SS);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * SS + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * SS + jj] * sc0 - mn);
                    P[qq * SS + jj] = T(pr);
                    S[qq * SS + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * SS + jj] = T(0.0f); S[qq * SS + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * SS + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * SS + kb * 8, SS);
                    simdgroup_load(vm, vt + (kb * 8) * KS + sg * 32 + dt * 8, KS);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V17 = r"""
    // v17 = v16 with the probability sum gathered by shuffles (no threadgroup round trip, same order)
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            float pv[KPT];
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    pv[i] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { P[qq * T_TILE + sub * KPT + i] = T(0.0f); pv[i] = 0.0f; }
            }
            // ascending-key fp32 sum of the quad's probabilities via shuffles (same order as v2's serial loop)
            float sum = 0.0f;
            for (int src = 0; src < 4; ++src)
                for (int i = 0; i < KPT; ++i) sum += simd_shuffle(pv[i], (lane & ~3u) + src);
            if (sub == 0) { ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V18 = r"""
    // v18 = v16 with 4 interleaved S accumulators (row-exact, NOT bitwise = v2)
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8));
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = uint4(0);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            // v18: 4 independent accumulators (dim blocks c mod 4) -> MMA chains of 8 instead of 32; fixed combine order
            simdgroup_float8x8 acc[4];
            for (int u = 0; u < 4; ++u) acc[u] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[c & 3], qa, kbt, acc[c & 3]);
            }
            thread auto& e0 = acc[0].thread_elements(); thread auto& e1 = acc[1].thread_elements();
            thread auto& e2 = acc[2].thread_elements(); thread auto& e3 = acc[3].thread_elements();
            e0[0] = (e0[0] + e1[0]) + (e2[0] + e3[0]); e0[1] = (e0[1] + e1[1]) + (e2[1] + e3[1]);
            simdgroup_store(acc[0], S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V19 = r"""
    // v19 = v16 with K staged transposed ([D][T]) so the S-phase loads are not transposed (same values)
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int e = tid; e < T_TILE * D_DIM / 8; e += 32 * G) {
            const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            const int key = t0 + kk;
            if (kk < tn) {
                const vec<T, 8> kv8 = *((const device vec<T, 8>*)(k + kvh * ks_h + key * ks_n + d8));
                for (int u = 0; u < 8; ++u) kt[(d8 + u) * T_TILE + kk] = kv8[u];      // v19: K^T layout [D][T]
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8));
            } else {
                for (int u = 0; u < 8; ++u) kt[(d8 + u) * T_TILE + kk] = T(0.0f);
                *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = uint4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (c * 8) * T_TILE + kb * 8, T_TILE);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1V20 = r"""
    // v20 = v16 + next tile K/V fetched into registers during the current tile (same values)
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ks_h = k_strides[1], ks_n = k_strides[2];
    const int64_t vs_h = v_strides[1], vs_n = v_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    constexpr int NV = T_TILE * D_DIM / 8 / (32 * G);
    uint4 rk[NV], rv[NV];
#define C35_FETCH(tt) for (int jv = 0; jv < NV; ++jv) { const int e = int(tid) + jv * 32 * G; const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8; const int key = (tt) + kk; \
        if (key < kend) { rk[jv] = *((const device uint4*)(k + kvh * ks_h + key * ks_n + d8)); rv[jv] = *((const device uint4*)(v + kvh * vs_h + key * vs_n + d8)); } \
        else { rk[jv] = uint4(0); rv[jv] = uint4(0); } }
    C35_FETCH(k0)
    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        for (int jv = 0; jv < NV; ++jv) {
            const int e = int(tid) + jv * 32 * G; const int kk = e / (D_DIM / 8), d8 = (e % (D_DIM / 8)) * 8;
            *((threadgroup uint4*)(kt + kk * D_DIM + d8)) = rk[jv];
            *((threadgroup uint4*)(vt + kk * D_DIM + d8)) = rv[jv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + T_TILE < kend) C35_FETCH(t0 + T_TILE)
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1TQ8 = r"""
    // TQ8 = v16 reading a TurboQuant 8-bit MSE KV state (mlx-vlm): q pre-rotated by the key codec, output in the value
    // codec rotated space (inverse rotation applied by the wrapper). Row-exact like v16.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t ki_h = kidx_strides[1], ki_n = kidx_strides[2], vi_h = vidx_strides[1], vi_n = vidx_strides[2];
    const int64_t kn_h = knorm_strides[1], kn_n = knorm_strides[2], vn_h = vnorm_strides[1], vn_n = vnorm_strides[2];
    threadgroup float cbt[256];
    if (tid < 256) cbt[tid] = cb[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        // TQ8: 16 codebook indices (one uint4) per thread per tensor, value = codebook[idx] * norm(token) -> T
        for (int e = tid; e < T_TILE * D_DIM / 16; e += 32 * G) {
            const int kk = e / (D_DIM / 16), d16 = (e % (D_DIM / 16)) * 16;
            const int key = t0 + kk;
            if (kk < tn) {
                const uint4 wk = *((const device uint4*)(kidx + kvh * ki_h + key * ki_n + d16 / 4));
                const uint4 wv = *((const device uint4*)(vidx + kvh * vi_h + key * vi_n + d16 / 4));
                const float nk = float(knorm[kvh * kn_h + key * kn_n]);
                const float nv = float(vnorm[kvh * vn_h + key * vn_n]);
                for (int w = 0; w < 4; ++w)
                    for (int b = 0; b < 4; ++b) {
                        kt[kk * D_DIM + d16 + w * 4 + b] = T(cbt[(wk[w] >> (8 * b)) & 255u] * nk);
                        vt[kk * D_DIM + d16 + w * 4 + b] = T(cbt[(wv[w] >> (8 * b)) & 255u] * nv);
                    }
            } else {
                for (int u = 0; u < 16; ++u) { kt[kk * D_DIM + d16 + u] = T(0.0f); vt[kk * D_DIM + d16 + u] = T(0.0f); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P1Q8 = r"""
    // Q8 = v16 reading an int8 affine KV (mx.quantize bits=8, group_size=32: Q8_0-class), dequantized at staging. Row-exact.
    // v16 = v13 with the S-phase dim loop fully unrolled (same math)
    // v13 = v12 + V tile double-buffered -> the trailing barrier of each tile is dropped (3 barriers per tile, same math)
    // v2: simdgroup MMA. grid threads (32*G, NSPLIT, HKV), threadgroup (32*G,1,1). Query index qi = r*8 + i (row r, head kvh*8+i).
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const int N = nkv[0];
    const int Lq = LROWS;
    const int NQ = 8 * LROWS;
    const float sc0 = sc[0];
    constexpr int KB = T_TILE / 8;   // 8-key blocks per tile

    threadgroup T kt[T_TILE * D_DIM];
    threadgroup T vt2[2 * T_TILE * D_DIM];
    threadgroup float S[8 * LROWS * T_TILE];
    threadgroup T P[8 * LROWS * T_TILE];
    threadgroup float ms[8 * LROWS], ls[8 * LROWS], as_[8 * LROWS];

    if (tid < uint(NQ)) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
    simdgroup_float8x8 O[LROWS][4];
    for (int r = 0; r < LROWS; ++r) for (int dt = 0; dt < 4; ++dt) O[r][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int k0 = int(split) * B_SPLIT;
    const int kend = min(k0 + B_SPLIT, N);
    const int64_t kw_h = kw_strides[1], kw_n = kw_strides[2], vw_h = vw_strides[1], vw_n = vw_strides[2];
    const int64_t ks_h = ks_strides[1], ks_n = ks_strides[2], kb_h = kb_strides[1], kb_n = kb_strides[2];
    const int64_t vs_h = vs_strides[1], vs_n = vs_strides[2], vb_h = vb_strides[1], vb_n = vb_strides[2];
    const device T* qg = q + int64_t(kvh * 8) * Lq * D_DIM;

    for (int t0 = k0; t0 < kend; t0 += T_TILE) {
        const int tn = min(T_TILE, kend - t0);
        threadgroup T* vt = vt2 + (((t0 - k0) / T_TILE) & 1) * (T_TILE * D_DIM);
        // Q8: 16 int8 codes (one uint4) per thread per tensor; value = code * scale + bias of its 32-dim group -> T
        for (int e = tid; e < T_TILE * D_DIM / 16; e += 32 * G) {
            const int kk = e / (D_DIM / 16), d16 = (e % (D_DIM / 16)) * 16;
            const int key = t0 + kk;
            if (kk < tn) {
                const uint4 wk = *((const device uint4*)(kw + kvh * kw_h + key * kw_n + d16 / 4));
                const uint4 wv = *((const device uint4*)(vw + kvh * vw_h + key * vw_n + d16 / 4));
                const int gi = d16 / 32;
                const float sk = float(ks[kvh * ks_h + key * ks_n + gi]), bk = float(kb[kvh * kb_h + key * kb_n + gi]);
                const float sv = float(vs[kvh * vs_h + key * vs_n + gi]), bv = float(vb[kvh * vb_h + key * vb_n + gi]);
                for (int w = 0; w < 4; ++w)
                    for (int b = 0; b < 4; ++b) {
                        kt[kk * D_DIM + d16 + w * 4 + b] = T(float((wk[w] >> (8 * b)) & 255u) * sk + bk);
                        vt[kk * D_DIM + d16 + w * 4 + b] = T(float((wv[w] >> (8 * b)) & 255u) * sv + bv);
                    }
            } else {
                for (int u = 0; u < 16; ++u) { kt[kk * D_DIM + d16 + u] = T(0.0f); vt[kk * D_DIM + d16 + u] = T(0.0f); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for pairs (row r, key block kb)
        for (int pp = sg; pp < KB * Lq; pp += G) {
            const int r = pp / KB, kb = pp % KB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int c = 0; c < D_DIM / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qa, kbt;
                simdgroup_load(qa, qg + r * D_DIM + c * 8, Lq * D_DIM);
                simdgroup_load(kbt, kt + (kb * 8) * D_DIM + c * 8, D_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kbt, acc);
            }
            simdgroup_store(acc, S + (r * 8) * T_TILE + kb * 8, T_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // v12: online softmax with 4 threads per query (same lanes quad of one simdgroup): exact max reduction, exp of each
        // key by its owner, then the owner of key 0 sums the fp32 probabilities in ascending key order -> bitwise = v2
        if (tid < uint(4 * NQ)) {
            const int qq = int(tid) >> 2, sub = int(tid) & 3;
            constexpr int KPT = T_TILE / 4;
            const int r = qq >> 3;
            const int lim = N - Lq + r;
            const float mold = ms[qq];
            float mloc = -INFINITY;
            for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; if (jj < tn && t0 + jj <= lim) mloc = max(mloc, S[qq * T_TILE + jj] * sc0); }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 2));
            const float mn = max(mold, mloc);
            float a = 1.0f;
            if (mn != -INFINITY) {
                a = metal::exp(mold - mn);
                for (int i = 0; i < KPT; ++i) {
                    const int jj = sub * KPT + i;
                    float pr = 0.0f;
                    if (jj < tn && t0 + jj <= lim) pr = metal::exp(S[qq * T_TILE + jj] * sc0 - mn);
                    P[qq * T_TILE + jj] = T(pr);
                    S[qq * T_TILE + jj] = pr;
                }
            } else {
                for (int i = 0; i < KPT; ++i) { const int jj = sub * KPT + i; P[qq * T_TILE + jj] = T(0.0f); S[qq * T_TILE + jj] = 0.0f; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                float sum = 0.0f;
                for (int jj = 0; jj < T_TILE; ++jj) sum += S[qq * T_TILE + jj];
                ls[qq] = ls[qq] * a + sum; ms[qq] = mn; as_[qq] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(a) O + P V on this simdgroup's 32-dim slice
        for (int r = 0; r < LROWS; ++r) {
            const float ar = as_[r * 8 + fm];
            for (int dt = 0; dt < 4; ++dt) {
                thread auto& el = O[r][dt].thread_elements();
                el[0] *= ar; el[1] *= ar;
                for (int kb = 0; kb < KB; ++kb) {
                    simdgroup_matrix<T, 8, 8> pm, vm;
                    simdgroup_load(pm, P + (r * 8) * T_TILE + kb * 8, T_TILE);
                    simdgroup_load(vm, vt + (kb * 8) * D_DIM + sg * 32 + dt * 8, D_DIM);
                    simdgroup_multiply_accumulate(O[r][dt], pm, vm, O[r][dt]);
                }
            }
        }
    }
    for (int r = 0; r < LROWS; ++r) {
        const int h = int(kvh) * 8 + fm;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        for (int dt = 0; dt < 4; ++dt) {
            thread auto& el = O[r][dt].thread_elements();
            po[base * D_DIM + sg * 32 + dt * 8 + fn] = el[0];
            po[base * D_DIM + sg * 32 + dt * 8 + fn + 1] = el[1];
        }
    }
    if (tid < uint(NQ)) {
        const int h = int(kvh) * 8 + (int(tid) & 7), r = int(tid) >> 3;
        const int64_t base = (int64_t(split) * HQ_N + h) * Lq + r;
        pm[base] = ms[tid]; pl[base] = ls[tid];
    }
"""

_P2B = r"""
    // pass 2, v2: same arithmetic and order as _P2 (bitwise identical); loads of 8 splits issued before their FMAs
    const uint d = thread_position_in_grid.x;
    const uint hr = thread_position_in_grid.y;     // h * L + r
    const int N = nkv[0];
    const int Lq = LROWS;
    const int r = int(hr) % Lq;
    const int kmax = N - Lq + r;                   // last key visible to this row
    const int ns = kmax / B_SPLIT + 1;             // splits holding visible keys
    const int64_t st = int64_t(HQ_N) * Lq;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = max(M, pm[int64_t(s) * st + hr]);
    float num = 0.0f, den = 0.0f;
    int s = 0;
    for (; s + 8 <= ns; s += 8) {
        float mv[8], ov[8], lv[8];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; ++j) { const int64_t b = int64_t(s + j) * st + hr; mv[j] = pm[b]; ov[j] = po[b * D_DIM + d]; lv[j] = pl[b]; }
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; ++j) { const float w = metal::exp(mv[j] - M); num += w * ov[j]; den += w * lv[j]; }
    }
    for (; s < ns; ++s) {
        const int64_t b = int64_t(s) * st + hr;
        const float w = metal::exp(pm[b] - M);
        num += w * po[b * D_DIM + d];
        den += w * pl[b];
    }
    out[hr * D_DIM + d] = T_IN(num / den);
"""

_P2C = r"""
    // pass 2, v3: as _P2B with P2U splits per batch, also for the max pass (bitwise identical to _P2)
    const uint d = thread_position_in_grid.x;
    const uint hr = thread_position_in_grid.y;     // h * L + r
    const int N = nkv[0];
    const int Lq = LROWS;
    const int r = int(hr) % Lq;
    const int kmax = N - Lq + r;                   // last key visible to this row
    const int ns = kmax / B_SPLIT + 1;             // splits holding visible keys
    const int64_t st = int64_t(HQ_N) * Lq;
    float M = -INFINITY;
    int s0 = 0;
    for (; s0 + P2U <= ns; s0 += P2U) {
        float mv[P2U];
        #pragma clang loop unroll(full)
        for (int j = 0; j < P2U; ++j) mv[j] = pm[int64_t(s0 + j) * st + hr];
        #pragma clang loop unroll(full)
        for (int j = 0; j < P2U; ++j) M = max(M, mv[j]);
    }
    for (; s0 < ns; ++s0) M = max(M, pm[int64_t(s0) * st + hr]);
    float num = 0.0f, den = 0.0f;
    int s = 0;
    for (; s + P2U <= ns; s += P2U) {
        float mv[P2U], ov[P2U], lv[P2U];
        #pragma clang loop unroll(full)
        for (int j = 0; j < P2U; ++j) { const int64_t b = int64_t(s + j) * st + hr; mv[j] = pm[b]; ov[j] = po[b * D_DIM + d]; lv[j] = pl[b]; }
        #pragma clang loop unroll(full)
        for (int j = 0; j < P2U; ++j) { const float w = metal::exp(mv[j] - M); num += w * ov[j]; den += w * lv[j]; }
    }
    for (; s < ns; ++s) {
        const int64_t b = int64_t(s) * st + hr;
        const float w = metal::exp(pm[b] - M);
        num += w * po[b * D_DIM + d];
        den += w * pl[b];
    }
    out[hr * D_DIM + d] = T_IN(num / den);
"""

_P2 = r"""
    // grid: threads (D, HQ*L, 1) ; one thread per output element, splits merged in ascending order
    const uint d = thread_position_in_grid.x;
    const uint hr = thread_position_in_grid.y;     // h * L + r
    const int N = nkv[0];
    const int Lq = LROWS;
    const int r = int(hr) % Lq;
    const int kmax = N - Lq + r;                   // last key visible to this row
    const int ns = kmax / B_SPLIT + 1;             // splits holding visible keys
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = max(M, pm[int64_t(s) * HQ_N * Lq + hr]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const int64_t b = int64_t(s) * HQ_N * Lq + hr;
        const float w = metal::exp(pm[b] - M);
        num += w * po[b * D_DIM + d];
        den += w * pl[b];
    }
    out[hr * D_DIM + d] = T_IN(num / den);
"""

_cache = {}


VERSION = int(os.environ.get("C35_GQA_VERSION", "2"))  # production kernel version (env override)


def _kernels(L):
    if (L, VERSION) not in _cache:
        hdr = f"#define ABL {os.environ.get('C35_GQA_ABL', '0')}\n#define LROWS {L}\n#define B_SPLIT {B}\n#define T_TILE {T}\n#define D_DIM {D}\n#define HQ_N {HQ}\n#define G {G}\n#define KS (D_DIM + {PAD})\n#define SS (LROWS <= 4 ? (T_TILE + 8) : T_TILE)\n#define P2U {P2U}\n"
        k1 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p1v{VERSION}_P{PAD}_L{L}_B{B}_T{T}_A{os.environ.get('C35_GQA_ABL', '0')}", input_names=["q", "k", "v", "nkv", "sc"],
                                  output_names=["po", "pm", "pl"], header=hdr, source={1: _P1, 2: _P1V2, 3: _P1V3, 4: _P1V4, 5: _P1V5, 6: _P1V6, 7: _P1V7, 8: _P1V8, 9: _P1V9, 10: _P1V10, 11: _P1V11, 12: _P1V12, 13: _P1V13, 14: _P1V14, 15: _P1V15, 16: _P1V16, 17: _P1V17, 18: _P1V18, 19: _P1V19, 20: _P1V20, 21: _P1V21, 22: _P1V22, 23: _P1V23, 24: _P1V24, 25: _P1V25, 26: _P1V26, 27: _P1V27, 28: _P1V28, 29: _P1V29, 30: _P1V30, 31: _P1V31}[VERSION].replace("T_IN", "T"),
                                  ensure_row_contiguous=False)
        k2 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p2v{P2V}u{P2U}_L{L}_B{B}_T{T}", input_names=["po", "pm", "pl", "nkv"],
                                  output_names=["out"], header=hdr, source=({2: _P2B, 3: _P2C}.get(P2V, _P2)).replace("T_IN", "T"))
        _cache[(L, VERSION)] = (k1, k2)
    return _cache[(L, VERSION)]


# ---- v31 (EXP-091): per-call split size and staging variants (all bitwise identical for a given split size B)
V31_KDEV = int(os.environ.get("C35_V31_KDEV", "-1"))  # -1 auto: K staged in threadgroup memory for L <= 6, read from device for L 7-8
V31_QREG = int(os.environ.get("C35_V31_QREG", "-1"))  # -1 auto: Q fragments in registers for L == 1 or (L <= 3 and N > 8192)
BRULE = os.environ.get("C35_GQA_BRULE", "0") in ("1", "2")   # split size from the row position (same for verify rows and 1-row decode)
BRULE2 = os.environ.get("C35_GQA_BRULE", "0") == "2"   # H31: wave-exact split size, passed at runtime (one kernel for every B)
_cache31 = {}


def _brule(p):
    """Split size for a query row with p keys before it: few splits leave the GPU idle at short context, many splits make
    pass 2 and the partials traffic dominate at long context (EXP-088)."""
    return 128 if p < 8192 else 256 if p < 49152 else 512 if p < 98304 else 1024


def _brule2(p):
    """H31 (wave-exact): 40*k splits = k full waves of 80 threadgroups (4 per core x 20 cores, 2 kv heads), k by position
    (dense grid N 8K-262K, scratchpad h31/wavegrid.json: k=1 best up to ~106K, k=2 to ~205K, k=3 beyond)."""
    if p < 8192:
        return 128
    k = 1 if p < 110592 else 2 if p < 212992 else 3
    return 16 * -(-p // (640 * k))


def _v31_tg_bytes(L, kdev, dbuf):
    return (2 if dbuf else 1) * T * D * 2 + (8 * D * 2 if kdev else T * D * 2) + 4 * 8 * L * T * 4 + 8 * L * T * 2 + 3 * 8 * L * 4


def _kernels31(L, bs, kdev, qreg):
    key = (L, bs, kdev, qreg)
    if key not in _cache31:
        dbuf = 1 if _v31_tg_bytes(L, kdev, 1) <= 32768 else 0
        assert _v31_tg_bytes(L, kdev, dbuf) <= 32768, ("v31 threadgroup memory", L, kdev)
        hdr = (f"#define LROWS {L}\n#define B_SPLIT {bs}\n#define T_TILE {T}\n#define D_DIM {D}\n#define HQ_N {HQ}\n#define G {G}\n"
               f"#define KDEV {kdev}\n#define QREG {qreg}\n#define DBUF {dbuf}\n#define P2U {P2U}\n")
        k1 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p1v31_k{kdev}q{qreg}d{dbuf}_L{L}_B{bs}_T{T}", input_names=["q", "k", "v", "nkv", "sc"],
                                  output_names=["po", "pm", "pl"], header=hdr, source=_P1V31, ensure_row_contiguous=False)
        k2 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p2v{P2V}u{P2U}_v31_L{L}_B{bs}_T{T}", input_names=["po", "pm", "pl", "nkv"],
                                  output_names=["out"], header=hdr, source=({2: _P2B, 3: _P2C}.get(P2V, _P2)).replace("T_IN", "T"))
        _cache31[key] = (k1, k2)
    return _cache31[key]


_cache31r = {}


def _kernels31r(L, kdev, qreg):
    """v31 with the split size read from a buffer (bsplit[0]): same source, B_SPLIT is only used for k0/kend (pass 1)
    and the visible-split count (pass 2), so the arithmetic equals the compiled-B kernels (checked bitwise)."""
    key = (L, kdev, qreg)
    if key not in _cache31r:
        dbuf = 1 if _v31_tg_bytes(L, kdev, 1) <= 32768 else 0
        hdr = (f"#define LROWS {L}\n#define B_SPLIT (bsplit[0])\n#define T_TILE {T}\n#define D_DIM {D}\n#define HQ_N {HQ}\n#define G {G}\n"
               f"#define KDEV {kdev}\n#define QREG {qreg}\n#define DBUF {dbuf}\n#define P2U {P2U}\n")
        k1 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p1v31r_k{kdev}q{qreg}d{dbuf}_L{L}_T{T}", input_names=["q", "k", "v", "nkv", "sc", "bsplit"],
                                  output_names=["po", "pm", "pl"], header=hdr, source=_P1V31, ensure_row_contiguous=False)
        k2 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p2v{P2V}u{P2U}_v31r_L{L}_T{T}", input_names=["po", "pm", "pl", "nkv", "bsplit"],
                                  output_names=["out"], header=hdr, source=({2: _P2B, 3: _P2C}.get(P2V, _P2)).replace("T_IN", "T"))
        _cache31r[key] = (k1, k2)
    return _cache31r[key]


def _gqa31(q, k, v, scale, bs, runtime_b=False):
    _, hq, L, d = q.shape
    N = k.shape[2]
    assert hq == HQ and d == D and k.shape[1] == HKV and 1 <= L <= 8, (q.shape, k.shape)
    kdev = V31_KDEV if V31_KDEV >= 0 else (0 if L <= 6 else 1)
    qreg = V31_QREG if V31_QREG >= 0 else int(L == 1 or (L <= 3 and N > 8192))
    q = mx.contiguous(q)
    ns = (N + bs - 1) // bs
    k1, k2 = _kernels31r(L, kdev, qreg) if runtime_b else _kernels31(L, bs, kdev, qreg)
    nkv = mx.array([N], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    xb = [mx.array([bs], dtype=mx.int32)] if runtime_b else []
    po, pm, pl = k1(inputs=[q, k, v, nkv, sc] + xb, template=[("T", q.dtype)],
                    grid=(32 * G, ns, HKV), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(ns, HQ, L, D), (ns, HQ, L), (ns, HQ, L)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = k2(inputs=[po, pm, pl, nkv] + xb, template=[("T", q.dtype)],
                grid=(D, HQ * L, 1), threadgroup=(D, 1, 1),
                output_shapes=[(1, HQ, L, D)], output_dtypes=[q.dtype])
    return out


def _gqa31_rows(q, k, v, scale):
    """v31 entry: with BRULE each row uses the split size of its own position; rows are grouped by split size, each group
    against the keys its last row may see (bottom-right causal) -> every row equals the 1-row decode at its position."""
    L, N = q.shape[2], k.shape[2]
    if not BRULE:
        return _gqa31(q, k, v, scale, B)
    rule = _brule2 if BRULE2 else _brule
    bs = [rule(N - L + r) for r in range(L)]
    if bs[0] == bs[-1]:
        return _gqa31(q, k, v, scale, bs[0], BRULE2)
    outs, c0 = [], 0
    for c1 in range(1, L + 1):
        if c1 == L or bs[c1] != bs[c0]:
            n = N - (L - c1)
            outs.append(_gqa31(q[:, :, c0:c1, :], k[:, :, :n, :], v[:, :, :n, :], scale, bs[c0], BRULE2))
            c0 = c1
    return mx.concatenate(outs, axis=2)


def gqa_attention(q, k, v, scale):
    if VERSION == 31:
        return _gqa31_rows(q, k, v, scale)
    _, hq, L, d = q.shape
    N = k.shape[2]
    assert hq == HQ and d == D and k.shape[1] == HKV and 1 <= L <= 8, (q.shape, k.shape)
    q = mx.contiguous(q)
    ns = (N + B - 1) // B
    k1, k2 = _kernels(L)
    nkv = mx.array([N], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    po, pm, pl = k1(inputs=[q, k, v, nkv, sc], template=[("T", q.dtype)],
                    grid=(32 * G, ns, HKV), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(ns, HQ, L, D), (ns, HQ, L), (ns, HQ, L)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = k2(inputs=[po, pm, pl, nkv], template=[("T", q.dtype)],
                grid=(D, HQ * L, 1), threadgroup=(D, 1, 1),
                output_shapes=[(1, HQ, L, D)], output_dtypes=[q.dtype])
    return out


_cache_tq = {}


def _kernels_tq(L):
    if L not in _cache_tq:
        hdr = f"#define LROWS {L}\n#define B_SPLIT {B}\n#define T_TILE {T}\n#define D_DIM {D}\n#define HQ_N {HQ}\n#define G {G}\n"
        k1 = mx.fast.metal_kernel(name=f"c35_gqa_attn_tq8_L{L}_B{B}_T{T}", input_names=["q", "kidx", "knorm", "vidx", "vnorm", "cb", "nkv", "sc"],
                                  output_names=["po", "pm", "pl"], header=hdr, source=_P1TQ8, ensure_row_contiguous=False)
        k2 = mx.fast.metal_kernel(name=f"c35_gqa_attn_p2f_L{L}_B{B}_T{T}", input_names=["po", "pm", "pl", "nkv"],
                                  output_names=["out"], header=hdr, source=_P2.replace("T_IN", "T"))
        _cache_tq[L] = (k1, k2)
    return _cache_tq[L]


def _rht(x, signs, inverse):
    """mlx-vlm TurboQuant RHT (D = 256, power of two): forward hadamard(signs*x)/16, inverse signs*hadamard(x)/16."""
    if inverse:
        return mx.hadamard_transform(x, scale=1.0 / 16.0) * signs
    return mx.hadamard_transform(x * signs, scale=1.0 / 16.0)


def gqa_attention_tq(q, kidx, knorm, vidx, vnorm, codebook, ksigns, vsigns, scale):
    """Row-exact attention over a TurboQuant 8-bit MSE KV state: kidx/vidx [1,2,N,64] uint32, knorm/vnorm [1,2,N].
    q . (n R^T c) = n (R q) . c -> rotate q once; sum_j p_j n_j R^T c_j = R^T (sum_j p_j n_j c_j) -> rotate the output once."""
    _, hq, L, d = q.shape
    N = kidx.shape[2]
    assert hq == HQ and d == D and kidx.shape[1] == HKV and kidx.shape[3] == D // 4 and 1 <= L <= 8, (q.shape, kidx.shape)
    qr = mx.contiguous(_rht(q.astype(mx.float32), ksigns, False).astype(q.dtype))
    ns = (N + B - 1) // B
    k1, k2 = _kernels_tq(L)
    nkv = mx.array([N], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    po, pm, pl = k1(inputs=[qr, kidx, knorm, vidx, vnorm, codebook, nkv, sc], template=[("T", q.dtype)],
                    grid=(32 * G, ns, HKV), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(ns, HQ, L, D), (ns, HQ, L), (ns, HQ, L)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32])
    (o,) = k2(inputs=[po, pm, pl, nkv], template=[("T", mx.float32)],
              grid=(D, HQ * L, 1), threadgroup=(D, 1, 1),
              output_shapes=[(1, HQ, L, D)], output_dtypes=[mx.float32])
    return _rht(o, vsigns, True).astype(q.dtype)


_cache_q8 = {}


def gqa_attention_q8(q, kw, ks, kb, vw, vs, vb, scale):
    """Row-exact attention over an int8 affine KV: *w [1,2,N,64] uint32 (mx.quantize bits=8 gs=32), *s/*b [1,2,N,8]."""
    _, hq, L, d = q.shape
    N = kw.shape[2]
    assert hq == HQ and d == D and kw.shape[1] == HKV and kw.shape[3] == D // 4 and ks.shape[3] == D // 32 and 1 <= L <= 8, (q.shape, kw.shape, ks.shape)
    if L not in _cache_q8:
        hdr = f"#define LROWS {L}\n#define B_SPLIT {B}\n#define T_TILE {T}\n#define D_DIM {D}\n#define HQ_N {HQ}\n#define G {G}\n"
        _cache_q8[L] = mx.fast.metal_kernel(name=f"c35_gqa_attn_q8_L{L}_B{B}_T{T}", input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "nkv", "sc"],
                                            output_names=["po", "pm", "pl"], header=hdr, source=_P1Q8, ensure_row_contiguous=False)
    k1 = _cache_q8[L]; _, k2 = _kernels(L)
    q = mx.contiguous(q)
    ns = (N + B - 1) // B
    nkv = mx.array([N], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    po, pm, pl = k1(inputs=[q, kw, ks, kb, vw, vs, vb, nkv, sc], template=[("T", q.dtype)],
                    grid=(32 * G, ns, HKV), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(ns, HQ, L, D), (ns, HQ, L), (ns, HQ, L)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = k2(inputs=[po, pm, pl, nkv], template=[("T", q.dtype)],
                grid=(D, HQ * L, 1), threadgroup=(D, 1, 1),
                output_shapes=[(1, HQ, L, D)], output_dtypes=[q.dtype])
    return out


def reference(q, k, v, scale):
    """fp32 reference, causal bottom-right."""
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    L, N = q.shape[2], k.shape[2]
    kf = mx.repeat(kf, G, axis=1); vf = mx.repeat(vf, G, axis=1)
    s = (qf * scale) @ kf.transpose(0, 1, 3, 2)
    pos = mx.arange(N)[None, :] <= (N - L + mx.arange(L))[:, None]
    s = mx.where(pos, s, -mx.inf)
    return (mx.softmax(s, axis=-1) @ vf)


if __name__ == "__main__":
    import time, statistics, sys
    for _v in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        if f"--v{_v}" in sys.argv: VERSION = _v
    mx.random.seed(0)
    sc = D ** -0.5
    # correctness + row-exactness (verify row r == decode at the same position)
    for N in (300, 4096 + 37):
        alloc = N + 200
        kb = mx.random.normal((1, HKV, alloc, D)).astype(mx.bfloat16)
        vb = mx.random.normal((1, HKV, alloc, D)).astype(mx.bfloat16)
        k, v = kb[:, :, :N, :], vb[:, :, :N, :]
        for L in (1, 4, 8):
            q = mx.random.normal((1, HQ, L, D)).astype(mx.bfloat16)
            o = gqa_attention(q, k, v, sc); ref = reference(q, k, v, sc)
            err = mx.abs(o.astype(mx.float32) - ref).max().item()
            # row-exact: row r of the L-row call equals a 1-row call with the KV truncated to N-L+r+1
            exact = all(mx.array_equal(o[:, :, r:r + 1, :], gqa_attention(q[:, :, r:r + 1, :], kb[:, :, :N - L + r + 1, :], vb[:, :, :N - L + r + 1, :], sc)).item() for r in range(L))
            ms = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal" if L > 1 else None)
            err_mlx = mx.abs(ms.astype(mx.float32) - ref).max().item()
            print(f"N={N} L={L}: max|err| ours {err:.2e}  mlx-sdpa {err_mlx:.2e}  row-exact {exact}", flush=True)
            assert (err < 2e-2 and exact) or VERSION == 15   # v15 = ablation (profiling only)
    if "--v1" in sys.argv:
        VERSION = 1
    if "--v3" in sys.argv:
        VERSION = 3
    if "--same-as" in sys.argv:   # bitwise equality with another kernel version on random inputs (incl. partial tiles)
        other = int(sys.argv[sys.argv.index("--same-as") + 1]); mine = VERSION; diff = 0
        for N in (300, 2051, 4133, 33000):
            alloc = N + 256
            kb_ = mx.random.normal((1, HKV, alloc, D)).astype(mx.bfloat16); vb_ = mx.random.normal((1, HKV, alloc, D)).astype(mx.bfloat16)
            for L in range(1, 9):
                q = mx.random.normal((1, HQ, L, D)).astype(mx.bfloat16)
                VERSION = mine; o1 = gqa_attention(q, kb_[:, :, :N, :], vb_[:, :, :N, :], sc)
                VERSION = other; o2 = gqa_attention(q, kb_[:, :, :N, :], vb_[:, :, :N, :], sc)
                if not mx.array_equal(o1, o2).item(): diff += 1; print("DIFF N", N, "L", L)
        VERSION = mine
        print(f"v{mine} vs v{other}: {'BITWISE IDENTICAL' if diff == 0 else f'{diff} mismatching cases'}")
    if "--bench" not in sys.argv:
        sys.exit(0)

    CHAIN = "--chain" in sys.argv
    def bench(fn, iters=30, layers=10):
        def step():
            if not CHAIN:
                return [fn() for _ in range(layers)]   # 10 independent calls in one graph
            # dependent chain like real layers: each call waits for the previous one
            out = fn()
            for _ in range(layers - 1):
                out = fn(out)
            return out
        for _ in range(3): mx.eval(step())
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); mx.eval(step()); ts.append(time.perf_counter() - t)
        return statistics.median(ts) / layers
    print(f"{'N':>7} {'L':>2} {'ours ms':>8} {'GB/s':>6} {'mlx ms':>8} {'GB/s':>6}")
    for N in ((2048, 8192, 16384, 32768, 65536, 131072) if "--moreN" in sys.argv else (2048, 32768, 65536, 131072)):
        k = mx.random.normal((1, HKV, N, D)).astype(mx.bfloat16); v = mx.random.normal((1, HKV, N, D)).astype(mx.bfloat16)
        mx.eval(k, v); kvb = k.nbytes + v.nbytes
        for L in ((1, 2, 3, 4, 8) if "--allL" in sys.argv else tuple(int(x) for x in os.environ.get("C35_BENCH_L", "1 4 8").split())):
            q = mx.random.normal((1, HQ, L, D)).astype(mx.bfloat16); mx.eval(q)
            t1 = bench(lambda prev=None: gqa_attention(q if prev is None else q + 0 * prev, k, v, sc))
            t2 = bench(lambda prev=None: mx.fast.scaled_dot_product_attention(q if prev is None else q + 0 * prev, k, v, scale=sc, mask="causal" if L > 1 else None))
            print(f"{N:7d} {L:2d} {t1*1e3:8.3f} {kvb/t1/1e9:6.1f} {t2*1e3:8.3f} {kvb/t2/1e9:6.1f}", flush=True)
