"""Per-component cost of a target forward with L rows (decode L=1 vs verify-like L=2..5), mlx-vlm stock kernels.
Times each decoder sub-module (GDN mixer, full attention, MoE block) and the head with an eval barrier per module
(adds a sync per module, so compare components across L rather than reading absolute totals).
Usage: python bench/profile_layers.py --ctx 2048 --L 1 3 5"""
import argparse, copy, time, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from mlx_vlm.utils import load
from mlx_vlm.models import cache as vcache
import prompts

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--L", type=int, nargs="+", default=[1, 3, 5])
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--c35", action="store_true", help="decode/verify attention through the c35 row-exact kernel (production path)")
a = ap.parse_args()

model, _ = load(prompts.MODEL_DIR)
lm = model.language_model
ids = prompts.prompt_ids(a.ctx, 0)
c0 = vcache.make_prompt_cache(lm)
x = mx.array(ids)[None]
for i in range(0, x.shape[1], 2048):
    lm(x[:, i:i + 2048], cache=c0); mx.eval([c.state for c in c0])

T = collections.defaultdict(float)
# time the attention core (SDPA) separately inside the full-attention module
from mlx_vlm.models.qwen3_5 import language as q35
_SDPA = q35.scaled_dot_product_attention
if a.c35:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
    from gqa_attn import gqa_attention
def _sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
    mx.eval(queries, keys, values)
    t = time.perf_counter()
    if a.c35 and queries.shape[2] <= 8 and queries.shape[1] == 16 and keys.shape[1] == 2 and not kw \
            and (mask is None or (isinstance(mask, str) and mask == "causal")):
        out = gqa_attention(queries, keys, values, scale)
    else:
        out = _SDPA(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)
    mx.eval(out); T["sdpa"] += time.perf_counter() - t
    return out
q35.scaled_dot_product_attention = _sdpa
layers = lm.model.layers
# instance-level dispatch: replace the attribute on the layer with a thin proxy module
class Proxy:
    """Plain callable wrapper (not an nn.Module) timing one sub-module with eval barriers."""
    def __init__(self, inner, name):
        object.__setattr__(self, "inner", inner); object.__setattr__(self, "name", name)
    def __call__(self, *args, **kw):
        mx.eval(args[0])
        t = time.perf_counter(); out = self.inner(*args, **kw); mx.eval(out); T[self.name] += time.perf_counter() - t
        return out
    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "inner"), k)
for l in layers:
    for attr, kind in (("linear_attn", "gdn"), ("self_attn", "attn"), ("mlp", "moe")):
        m = getattr(l, attr, None)
        if m is not None and not isinstance(m, Proxy):
            setattr(l, attr, Proxy(m, kind))

tok = ids[-1]
print(f"ctx {a.ctx}")
for L in a.L:
    inp = mx.array([[tok] * L])
    best = None
    for it in range(a.iters + 2):
        c = copy.deepcopy(c0); T.clear(); mx.synchronize()
        t = time.perf_counter(); out = lm(inp, cache=c); mx.eval(out.logits); tot = time.perf_counter() - t
        if it >= 2 and (best is None or tot < best[0]): best = (tot, dict(T))
        del c
    tot, comp = best
    other = tot - sum(comp.values())
    print(f"L={L}: total {tot*1e3:7.2f} ms | " + " | ".join(f"{k} {v*1e3:6.2f}" for k, v in sorted(comp.items())) + f" | other(head/embed/norms/sync) {other*1e3:6.2f}")
