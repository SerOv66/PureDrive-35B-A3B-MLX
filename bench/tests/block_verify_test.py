"""H19 exactness test: speculative sampling with the runtime's own draft sampler and sparse stochastic verifier, token
verification vs block verification (C35_BLOCK_VERIFY), on a context-free position-dependent toy model (V=32 tokens).
Emitted tokens at positions 0..3 must follow the target sampler's effective distribution (top-k 20, top-p 0.95, T 0.6)
independently per position: chi-square per position and on the (pos0, pos1) joint. Usage: python bench/tests/block_verify_test.py [trials]"""
import math, os, sys
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "runtimes", "omlx"))
import numpy as np
import mlx.core as mx
from omlx.utils.sampling import make_sampler
from omlx.patches.mlx_lm_mtp import batch_generator as bg

V, K, NPOS = 32, 2, 4
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(0)
tgt_logits = rng.normal(0, 2.0, (NPOS + K + 1, V)).astype(np.float32)
drf_logits = (tgt_logits + rng.normal(0, float(sys.argv[2]) if len(sys.argv) > 2 else 0.6, tgt_logits.shape)).astype(np.float32)   # imperfect draft
tgt_lp = mx.array(tgt_logits) - mx.logsumexp(mx.array(tgt_logits), axis=-1, keepdims=True)
drf_lp = mx.array(drf_logits) - mx.logsumexp(mx.array(drf_logits), axis=-1, keepdims=True)
target = make_sampler(temp=0.6, top_p=0.95, top_k=20)
draft = make_sampler(temp=0.3, top_p=0.95, top_k=20)

def effective(row):   # the target sampler's distribution, same filters as _stochastic_verify_tokens_sparse
    lp = np.array(tgt_lp[row]); ids = np.argsort(-lp)[:20]; vals = lp[ids]
    pr = np.exp(vals); keep = (np.cumsum(pr) - pr) < 0.95
    vals = np.where(keep, vals, -np.inf) / 0.6
    p = np.exp(vals - vals.max()); p /= p.sum(); out = np.zeros(V); out[ids] = p; return out

def run(block):
    bg._C35_BLOCK_VERIFY = block
    mx.random.seed((1 if block else 2) + 100 * (int(sys.argv[3]) if len(sys.argv) > 3 else 0))
    seqs = np.zeros((TRIALS, NPOS), dtype=np.int64); acc_tot = 0; cyc = 0
    for t in range(TRIALS):
        pos, out = 0, []
        while pos < NPOS:
            toks, idss, lqs = [], [], []
            for j in range(K):
                tok, ids, lq = bg._sample_draft_sparse(draft, drf_lp[pos + j][None], 20)
                toks.append(tok.reshape(1)); idss.append(ids.reshape(1, -1)); lqs.append(lq.reshape(1, -1))
            drafts = mx.concatenate(toks).astype(mx.uint32)
            q = bg.SparseDraftQ(mx.concatenate(idss), mx.concatenate(lqs), V)
            r = bg._stochastic_verify_tokens_sparse(target, tgt_lp[pos:pos + K + 1], drafts, q, 20).tolist()
            m = r[0]; d = r[1:K + 1]; res = r[K + 1:2 * K + 1]; bonus = r[2 * K + 1]
            emit = d[:m] + [res[m] if m < K else bonus]
            out += emit; pos += len(emit); acc_tot += m; cyc += 1
        seqs[t] = out[:NPOS]
    return seqs, acc_tot / cyc

def chi2_p(obs, exp):
    mask = exp >= 5; o, e = obs[mask], exp[mask]
    o = np.append(o, obs[~mask].sum()); e = np.append(e, exp[~mask].sum())
    if e[-1] < 5: o, e = o[:-1], e[:-1]
    x2 = float(((o - e) ** 2 / e).sum()); df = len(o) - 1
    z = ((x2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))   # Wilson-Hilferty
    return x2, df, 0.5 * math.erfc(z / math.sqrt(2))

for block in (False, True):
    seqs, acc = run(block)
    ps = [chi2_p(np.bincount(seqs[:, i], minlength=V).astype(float), effective(i) * TRIALS) for i in range(NPOS)]
    joint_exp = np.outer(effective(0), effective(1)).reshape(-1) * TRIALS
    joint_obs = np.bincount(seqs[:, 0] * V + seqs[:, 1], minlength=V * V).astype(float)
    pj = chi2_p(joint_obs, joint_exp)
    print(f"{'block' if block else 'token'}: accepted/cycle {acc:.4f} | per-position p-values " +
          " ".join(f"{p:.3f}" for _, _, p in ps) + f" | joint(0,1) chi2 {pj[0]:.1f} df {pj[1]} p {pj[2]:.3f}", flush=True)
