"""K2: bitwise identity of the fused attention output glue (runtimes/omlx/omlx/patches/c35_attn_out.py) against the stock
mlx-vlm Qwen3_5Attention.__call__ (random weights, real module): whole layer output for L 1..16, cache offsets 0..40000,
input scales 1e-2..30, gate values spanning the sigmoid range. Sensitivity: a perturbed o_proj input must differ.
Usage: runtimes/venv-omlx/bin/python bench/tests/attn_out_test.py"""
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.models.cache import KVCache

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "runtimes" / "omlx"))
cfg = json.load(open(Path(__import__("os").environ["MODEL_DIR"]) / "config.json"))
tc = dict(cfg.get("text_config", cfg)); tc.setdefault("intermediate_size", tc.get("moe_intermediate_size", 512))
att = q35.Qwen3_5Attention(q35.TextConfig.from_dict(tc))
mx.random.seed(5)
att.set_dtype(mx.bfloat16)
orig = q35.Qwen3_5Attention.__call__
from omlx.patches import c35_attn_out
c35_attn_out.apply()
new = q35.Qwen3_5Attention.__call__
assert new is not orig
H, HK, D = att.num_attention_heads, att.num_key_value_heads, att.head_dim
Dm = tc["hidden_size"]


def cache_at(n, kv):
    c = KVCache()
    if n:
        c.update_and_fetch(kv[0][:, :, :n], kv[1][:, :, :n])
    return c


bad = n = 0
for sc in (1.0, 0.01, 30.0):
    for off in (0, 5, 2047, 40000):
        kv = [(mx.random.normal((1, HK, max(off, 1), D))).astype(mx.bfloat16) for _ in range(2)]
        for L in (1, 2, 3, 8, 16):
            x = (mx.random.normal((1, L, Dm)) * sc).astype(mx.bfloat16)
            a = orig(att, x, "causal" if L > 1 else None, cache_at(off, kv))
            b = new(att, x, "causal" if L > 1 else None, cache_at(off, kv))
            same = mx.array_equal(a, b).item()
            n += 1; bad += not same
            if not same:
                print(f"DIFF scale {sc} off {off} L {L}: max |a-b| {mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()}")
# sensitivity: the kernel output itself against a one-ulp-perturbed gate
qo = mx.random.normal((1, 3, H * 2 * D)).astype(mx.bfloat16); at = mx.random.normal((1, H, 3, D)).astype(mx.bfloat16)
ref = at.transpose(0, 2, 1, 3).reshape(1, 3, -1) * mx.sigmoid(mx.split(qo.reshape(1, 3, H, -1), 2, axis=-1)[1].reshape(1, 3, -1))
got = c35_attn_out.attn_out(at, qo, H, D)
sens = not mx.array_equal(ref, c35_attn_out.attn_out(at, qo * 1.01, H, D)).item()
eng = "k" in c35_attn_out._KERNEL
print(f"attn_out vs stock: {n - bad}/{n} BITWISE IDENTICAL (layer); kernel vs ops identical: {mx.array_equal(ref, got).item()}; "
      f"sensitivity: {sens}; fused kernel engaged: {eng}")
sys.exit(0 if bad == 0 and sens and eng and mx.array_equal(ref, got).item() else 1)
