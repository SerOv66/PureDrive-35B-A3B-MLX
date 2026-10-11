"""AUDIT EXP-110: C35_QK_PREP_MAIN=1 with position_ids that are NOT cache.offset + arange (M-RoPE delta after an image,
or image tokens with different sections). Same setup as bench/tests/qk_prep_test.py (random weights, one attention layer).
Expected if H28 is exact: every case SAME. Usage: runtimes/venv-mlx/bin/python bench/audit/h28_fallback_test.py"""
import json, os, sys
os.environ["C35_QK_PREP_MAIN"] = "1"
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.models.cache import KVCache

W = Path(__import__("os").environ.get("REPO_DIR", "."))
sys.path.insert(0, str(W / "runtimes" / "omlx"))
cfg = json.load(open(Path(__import__("os").environ["MODEL_DIR"]) / "config.json"))
tc = dict(cfg.get("text_config", cfg)); tc.setdefault("intermediate_size", tc.get("moe_intermediate_size", 512))
att = q35.Qwen3_5Attention(q35.TextConfig.from_dict(tc)); att.set_dtype(mx.bfloat16)
orig = q35.Qwen3_5Attention._prepare_projected_qkv
from omlx.patches import c35_qk_prep
c35_qk_prep.apply()
new = q35.Qwen3_5Attention._prepare_projected_qkv
H, HK, D = att.num_attention_heads, att.num_key_value_heads, att.head_dim
mx.random.seed(110)
off = 1000
kv = [mx.random.normal((1, HK, off, D)).astype(mx.bfloat16) for _ in range(2)]


def cache():
    c = KVCache(); c.update_and_fetch(kv[0], kv[1]); return c


def case(name, L, pid):
    qo = mx.random.normal((1, L, H * D * 2)).astype(mx.bfloat16)
    kk = mx.random.normal((1, L, HK * D)).astype(mx.bfloat16)
    vv = mx.random.normal((1, L, HK * D)).astype(mx.bfloat16)
    a = orig(att, qo, kk, vv, cache(), pid, None, "causal")
    b = new(att, qo, kk, vv, cache(), pid, None, "causal")
    same = [mx.array_equal(x, y).item() for x, y in zip(a[:4], b[:4])]
    print(f"{name:58s} L={L:3d} q/k/v/gate equal {same} -> {'SAME' if all(same) else 'DIFF'}", flush=True)


delta = 37   # M-RoPE delta of the text that follows an image (positions != cache offset)
for L in (8, 128):
    case("text-only: pos = offset + arange (3 equal sections)", L, mx.broadcast_to(mx.arange(off, off + L)[None, None], (3, 1, L)))
    case("text after image: pos = offset + delta + arange", L, mx.broadcast_to(mx.arange(off + delta, off + delta + L)[None, None], (3, 1, L)))
    img = mx.stack([mx.full((L,), off), off + mx.arange(L) // 4, off + mx.arange(L) % 4])[:, None, :]   # image grid (t, h, w)
    case("image tokens: 3 different sections", L, img.astype(mx.int32))
