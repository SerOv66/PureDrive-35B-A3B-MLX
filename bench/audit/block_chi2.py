"""AUDIT EXP-086 (H19): independent exactness test of C35_BLOCK_VERIFY with the runtime's own functions
(_sample_draft_sparse with the production draft sampler T 0.3 / top-p 0.95 / top-k, _stochastic_verify_tokens_sparse with
the target sampler T 0.6 / top-p 0.95 / top-k), on a CONTEXT-DEPENDENT toy model (first-order Markov chain over V tokens,
target and draft rows conditioned on the previous token, so every verify row is conditioned on the drafted prefix).
Chi-square of the first 3 emitted tokens (full joint) against the exact target joint. Modes: token k=2 (control),
block k=2 (production), block k=3, block with deterministic copy drafts k=4 (one-hot q, _copy_draft_q), block k=2 and
copy k=4 with the scheduler's paged-boundary truncation (m -> to_boundary, emit draft m), and two mutations of the
block weights that must be rejected. Runs on the CPU device (no GPU use).
Usage: python block_chi2.py <workspace> <trials> [mode ...]"""
import math, os, sys, time
sys.path.insert(0, os.path.join(sys.argv[1], "runtimes", "omlx"))
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
from omlx.utils.sampling import make_sampler
from omlx.patches.mlx_lm_mtp import batch_generator as bg

TRIALS = int(sys.argv[2])
MODES = sys.argv[3:] or ["token2", "block2", "block3", "copy4", "block2_align", "copy4_align", "mutA", "mutB"]
V, TOPK, NPOS = 16, 12, 3
rng = np.random.default_rng(12345)
T_log = rng.normal(0, 1.5, (V, V)).astype(np.float32)          # row = previous token
D_log = (T_log + rng.normal(0, 0.8, (V, V))).astype(np.float32)  # imperfect draft
tgt = mx.array(T_log) - mx.logsumexp(mx.array(T_log), axis=-1, keepdims=True)
drf = mx.array(D_log) - mx.logsumexp(mx.array(D_log), axis=-1, keepdims=True)
target = make_sampler(temp=0.6, top_p=0.95, top_k=TOPK)
draft = make_sampler(temp=0.3, top_p=0.95, top_k=TOPK)
ORIG_W = bg._c35_block_log_weights

def eff(prev):   # exact target sampler distribution given the previous token
    lp = np.array(tgt[prev]); ids = np.argsort(-lp)[:TOPK]; vals = lp[ids]
    pr = np.exp(vals); keep = (np.cumsum(pr) - pr) < 0.95
    vals = np.where(keep, vals, -np.inf) / 0.6
    p = np.exp(vals - vals.max()); p /= p.sum(); out = np.zeros(V); out[ids] = p; return out

E = np.stack([eff(i) for i in range(V)])
joint = (E[0][:, None, None] * E[:, :, None] * E[None, :, :]).reshape(-1)   # c0 = 0

def chi2(obs, exp):
    order = np.argsort(exp); o, e = obs[order], exp[order]
    small = e < 5
    o2, e2 = np.append(o[~small], o[small].sum()), np.append(e[~small], e[small].sum())
    if e2[-1] < 5: o2, e2 = o2[:-1], e2[:-1]
    x2 = float(((o2 - e2) ** 2 / e2).sum()); df = len(o2) - 1
    z = ((x2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return x2, df, 0.5 * math.erfc(z / math.sqrt(2))

def run(mode, trials, seed):
    block = mode != "token2"
    bg._C35_BLOCK_VERIFY = block
    bg._c35_block_log_weights = ORIG_W
    if mode == "mutA":   # missing min(., 1): p_i may exceed 1
        bg._c35_block_log_weights = lambda r: mx.cumsum(r)
    if mode == "mutB":   # per-token weights, no running product
        bg._c35_block_log_weights = lambda r: mx.minimum(r, 0.0)
    k = {"token2": 2, "block2": 2, "block3": 3, "copy4": 4, "block2_align": 2, "block3_align": 3, "copy4_align": 4, "mutA": 2, "mutB": 2}[mode]
    copy = mode.startswith("copy")
    align = {"block2_align": 2, "block3_align": 2, "copy4_align": 3}.get(mode, 0)
    mx.random.seed(seed)
    seqs = np.zeros((trials, NPOS), dtype=np.int64); acc = cyc = trunc = 0
    for t in range(trials):
        out, last = [], 0
        while len(out) < NPOS:
            if copy:
                d, x = [], last
                for _ in range(k):
                    x = int(np.argmax(T_log[x])); d.append(x)   # deterministic copy-like proposal: a function of the context only
                drafts = mx.array(d, dtype=mx.uint32); q = bg._copy_draft_q(d, V)
            else:
                toks, idss, lqs, x = [], [], [], last
                for _ in range(k):
                    tok, ids, lq = bg._sample_draft_sparse(draft, drf[x][None], TOPK)
                    toks.append(tok.reshape(1)); idss.append(ids.reshape(1, -1)); lqs.append(lq.reshape(1, -1))
                    x = int(tok.item())
                drafts = mx.concatenate(toks).astype(mx.uint32)
                q = bg.SparseDraftQ(mx.concatenate(idss), mx.concatenate(lqs), V)
            rows = mx.array([last] + [int(v) for v in drafts.tolist()[:k]])
            r = bg._stochastic_verify_tokens_sparse(target, tgt[rows], drafts, q, TOPK).tolist()
            m, dd, res, bonus = r[0], r[1:k + 1], r[k + 1:2 * k + 1], r[2 * k + 1]
            emit_last = res[m] if m < k else bonus
            if align:   # batch_generator: to_boundary = ((emitted // align) + 1) * align - emitted; 0 < to_boundary < m -> m = to_boundary
                tb = ((len(out) // align) + 1) * align - len(out)
                if 0 < tb < m:
                    m, emit_last = tb, dd[tb]; trunc += 1
            emit = dd[:m] + [emit_last]
            out += emit; last = out[-1]; acc += m; cyc += 1
        seqs[t] = out[:NPOS]
    return seqs, acc / cyc, trunc

if __name__ == "__main__":
    for i, mode in enumerate(MODES):
        n = TRIALS if not mode.startswith("mut") else max(TRIALS // 3, 1000)
        t0 = time.time()
        seqs, apc, trunc = run(mode, n, 1000 + i)
        obs = np.bincount(seqs[:, 0] * V * V + seqs[:, 1] * V + seqs[:, 2], minlength=V ** 3).astype(float)
        x2, df, p = chi2(obs, joint * n)
        tv = 0.5 * np.abs(obs / n - joint).sum()
        print(f"{mode:12s} trials {n} | accepted/cycle {apc:.4f} | truncations {trunc} | joint(3 tok) chi2 {x2:.1f} df {df} p {p:.4g} | TV {tv:.4f} | {time.time() - t0:.0f}s", flush=True)
