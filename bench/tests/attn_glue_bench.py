"""Time of the full-attention "glue" per layer in the verify (L rows): q/gate split, q/k RMSNorm, M-RoPE, KV cache write,
sigmoid(gate)*out — everything of Qwen3_5Attention except the q/k/v/o projections and the attention kernel.
Real module (mlx-vlm Qwen3_5Attention, text config of the checkpoint, read-only), random weights, KVCache prefilled to N.
Each step depends on the previous one (chain, built lazily, one eval); per-step time minus a chain of 1 trivial op = glue cost.
Usage: runtimes/venv-omlx/bin/python bench/tests/attn_glue_bench.py [--rows 3] [--n 2048] [--steps 200]"""
import argparse, json, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.models.cache import KVCache

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=3)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--qk-prep", action="store_true", help="apply the fused q/k prep patch (EXP-093)")
a = ap.parse_args()
cfg = json.load(open(Path(__import__("os").environ["MODEL_DIR"]) / "config.json"))
tc = dict(cfg.get("text_config", cfg)); tc.setdefault("intermediate_size", tc.get("moe_intermediate_size", 512))
args = q35.TextConfig.from_dict(tc)
if a.qk_prep:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtimes" / "omlx"))
    from omlx.patches import c35_qk_prep; c35_qk_prep.apply()
att = q35.Qwen3_5Attention(args)
att.set_dtype(mx.bfloat16)
L, H, HKV, D = a.rows, args.num_attention_heads, args.num_key_value_heads, args.head_dim


def fresh_cache():
    c = KVCache()
    c.update_and_fetch(mx.random.normal((1, HKV, a.n, D)).astype(mx.bfloat16), mx.random.normal((1, HKV, a.n, D)).astype(mx.bfloat16))
    mx.eval(c.keys, c.values)
    return c


qo = mx.random.normal((1, L, H * D * 2)).astype(mx.bfloat16)
kk = mx.random.normal((1, L, HKV * D)).astype(mx.bfloat16)
vv = mx.random.normal((1, L, HKV * D)).astype(mx.bfloat16)
attn_out = mx.random.normal((1, L, H * D)).astype(mx.bfloat16)
mx.eval(qo, kk, vv, attn_out)


def glue_step(c, dep):
    q_in = qo + dep                                   # dependency on the previous step (dep is a 0-valued scalar)
    queries, keys, values, gate, _ = att._prepare_projected_qkv(q_in, kk, vv, c, None, None, None)
    out = attn_out * mx.sigmoid(gate)
    return (queries[0, 0, 0, 0] + keys[0, 0, -1, 0] + values[0, 0, -1, 0] + out[0, 0, 0]) * 0


def floor_step(c, dep):
    return (qo[0, 0, 0] + dep) * 0


def run(fn):
    ts = []
    for rep in range(5):
        c = fresh_cache(); dep = mx.array(0, dtype=mx.bfloat16)
        for _ in range(10):
            dep = fn(c, dep); mx.eval(dep)
        for _ in range(a.steps):                      # build the dependent chain first (host time excluded)
            dep = fn(c, dep)
        t = time.perf_counter()
        mx.eval(dep)                                  # encode + GPU of the whole chain
        ts.append((time.perf_counter() - t) / a.steps * 1e3)
    return statistics.median(ts)


g, f = run(glue_step), run(floor_step)
print(f"rows {L} N {a.n}: glue step {g:.4f} ms, floor {f:.4f} ms -> glue {g - f:.4f} ms/layer, x10 layers = {10 * (g - f):.3f} ms/cycle")
