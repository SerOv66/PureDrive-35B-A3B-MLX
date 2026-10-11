"""EXP-093 (+ H28/EXP-110: main-model position_ids 3-D/2-D, int32/int64, rope delta, different M-RoPE sections, C35_QK_PREP_MAIN=1): bitwise identity of the fused q/k prep (runtimes/omlx/omlx/patches/c35_qk_prep.py) against the stock
mlx-vlm Qwen3_5Attention._prepare_projected_qkv: queries, keys/values fetched from the cache, gate; L 1..16, cache offsets
from 0 to 131k (large RoPE angles), random norm weights, input scales 1e-2..30. Sensitivity: a perturbed eps must differ.
Usage: runtimes/venv-omlx/bin/python bench/tests/qk_prep_test.py"""
import json, os, sys
os.environ["C35_QK_PREP_MAIN"] = "1"
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.models.cache import KVCache

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "runtimes" / "omlx"))
cfg = json.load(open(Path(__import__("os").environ["MODEL_DIR"]) / "config.json"))
tc = dict(cfg.get("text_config", cfg)); tc.setdefault("intermediate_size", tc.get("moe_intermediate_size", 512))
att = q35.Qwen3_5Attention(q35.TextConfig.from_dict(tc))
att.set_dtype(mx.bfloat16)
orig = q35.Qwen3_5Attention._prepare_projected_qkv
from omlx.patches import c35_qk_prep
c35_qk_prep.apply()
new = q35.Qwen3_5Attention._prepare_projected_qkv
assert new is not orig
H, HK, D = att.num_attention_heads, att.num_key_value_heads, att.head_dim


def cache_at(n, kv):
    c = KVCache()
    if n:
        c.update_and_fetch(kv[0][:, :, :n], kv[1][:, :, :n])
    return c


mx.random.seed(3)
bad = n = 0
for seed, sc in ((0, 1.0), (1, 0.01), (2, 30.0)):
    att.q_norm.weight = (mx.random.normal((D,)) * 0.5 + 1).astype(mx.bfloat16)
    att.k_norm.weight = (mx.random.normal((D,)) * 0.5 + 1).astype(mx.bfloat16)
    for off in (0, 5, 2047, 8191, 40000, 131070):
        kv = [(mx.random.normal((1, HK, max(off, 1), D))).astype(mx.bfloat16) for _ in range(2)]
        for L in (1, 2, 3, 8, 16):
            qo = (mx.random.normal((1, L, H * D * 2)) * sc).astype(mx.bfloat16)
            kk = (mx.random.normal((1, L, HK * D)) * sc).astype(mx.bfloat16)
            vv = mx.random.normal((1, L, HK * D)).astype(mx.bfloat16)
            # H28: main-model call shapes (position_ids = arange + delta, 3 equal M-RoPE sections, or 2-D), int32 and int64
            pids = [None, mx.broadcast_to(mx.arange(off, off + L)[None, None], (3, 1, L)),
                    mx.arange(off, off + L).astype(mx.int64)[None], mx.broadcast_to((mx.arange(L) + mx.array(off).astype(mx.int64))[None, None], (3, 1, L)),
                    # AUDIT EXP-110 C1: rope delta (text after an image) and 3 different sections (image tokens), contiguous and int64
                    mx.broadcast_to(mx.arange(off + 37, off + 37 + L)[None, None], (3, 1, L)),
                    mx.stack([mx.full((L,), off), off + mx.arange(L) // 4, off + 1 + mx.arange(L) % 4])[:, None, :].astype(mx.int32),
                    mx.stack([mx.full((L,), off + 3), off + mx.arange(L) % 3, off + 2 * mx.arange(L)])[:, None, :].astype(mx.int64)]
            for pid in pids:
                a = orig(att, qo, kk, vv, cache_at(off, kv), pid, None, "causal")
                b = new(att, qo, kk, vv, cache_at(off, kv), pid, None, "causal")
                same = all(mx.array_equal(x, y).item() for x, y in zip(a[:4], b[:4]))
                n += 1; bad += not same
                if not same:
                    print(f"DIFF seed {seed} scale {sc} off {off} L {L} pid {None if pid is None else pid.shape}:",
                          [mx.array_equal(x, y).item() for x, y in zip(a[:4], b[:4])])
# sensitivity: a different eps in the fused kernel must be detected
eps0 = att.q_norm.eps; att.q_norm.eps = att.k_norm.eps = eps0 * 1.5
qo = mx.random.normal((1, 3, H * D * 2)).astype(mx.bfloat16) * 1e-3; kk = mx.random.normal((1, 3, HK * D)).astype(mx.bfloat16) * 1e-3
vv = mx.random.normal((1, 3, HK * D)).astype(mx.bfloat16)
kv = [mx.random.normal((1, HK, 100, D)).astype(mx.bfloat16) for _ in range(2)]
b = new(att, qo, kk, vv, cache_at(100, kv), None, None, "causal")
att.q_norm.eps = att.k_norm.eps = eps0
a = orig(att, qo, kk, vv, cache_at(100, kv), None, None, "causal")
sens = not mx.array_equal(a[0], b[0]).item()
eng = "k" in c35_qk_prep._KERNEL
print(f"qk_prep vs stock: {n - bad}/{n} BITWISE IDENTICAL; sensitivity (eps x1.5 detected): {sens}; fused kernel engaged: {eng}")
sys.exit(0 if bad == 0 and sens and eng else 1)
