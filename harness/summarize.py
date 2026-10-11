#!/usr/bin/env python3
"""Aggregates harness/results/*/{meta.json,proxy.jsonl} into tables. Usage: summarize.py [--runs] [id_prefix]"""
import json, glob, sys, statistics as st, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
runs = []
for m in sorted(glob.glob(f"{R}/*/meta.json")):
    meta = json.load(open(m)); d = os.path.dirname(m)
    reqs = [json.loads(l) for l in open(f"{d}/proxy.jsonl")] if os.path.exists(f"{d}/proxy.jsonl") else []
    main = [r for r in reqs if r["n_tools"] > 0 or not r["has_schema"]]
    u = lambda r, k: (r.get("usage") or {}).get(k) or 0
    pt = sum(u(r, "prompt_tokens") for r in main); ct = sum(u(r, "completion_tokens") for r in main)
    cached = sum(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) for r in main)
    gen = sum(u(r, "generation_duration") for r in main)
    srv = sum(r["t_end"] - r["t0"] for r in main)
    meta.update(req=len(main), aux=len(reqs) - len(main), ptok=pt, ctok=ct, cached=cached, hit=round(cached / pt, 3) if pt else 0,
                dec_tps=round(ct / gen, 1) if gen else 0, ttft_med=round(st.median([r["ttft"] for r in main if r["ttft"]]), 2) if main else 0,
                ttft_max=round(max([r["ttft"] or 0 for r in main], default=0), 1), model_s=round(srv, 1), leak=sum(1 for r in main if r.get("toolcall_leak")),
                tcalls=sum(r["n_tool_calls"] for r in main), maxprompt=max([u(r, "prompt_tokens") for r in main], default=0))
    runs.append(meta)
pre = [a for a in sys.argv[1:] if not a.startswith("--")]
if pre: runs = [r for r in runs if r["id"].startswith(pre[0])]
if "--runs" in sys.argv:
    for r in runs:
        print(f'{r["id"]:38} ok={int(r["success"])} wall={r["wall_s"]:6.1f}s req={r["req"]:3} aux={r["aux"]} tcalls={r["tcalls"]:3} ptok={r["ptok"]:7} cached={r["hit"]:.0%} ctok={r["ctok"]:5} dec={r["dec_tps"]:5.1f} ttft_med={r["ttft_med"]:5.2f} max={r["ttft_max"]:5.1f} leak={r["leak"]} model_s={r["model_s"]}')
g = collections.defaultdict(list)
for r in runs: g[(r["harness"], r["task"])].append(r)
print("\nharness x task: n, successes, median wall (s), median requests, median prompt/completion tokens, median prefix-hit, decode tok/s (weighted mean), median TTFT (s)")
for (h, t), rs in sorted(g.items()):
    med = lambda k: st.median([x[k] for x in rs])
    print(f"{h:7} {t:12} n={len(rs)} ok={sum(x['success'] for x in rs)}/{len(rs)} wall={med('wall_s'):6.1f} req={med('req'):5.1f} ptok={med('ptok'):8.0f} ctok={med('ctok'):6.0f} hit={med('hit'):.0%} dec={st.mean([x['dec_tps'] for x in rs]):5.1f} ttft={med('ttft_med'):5.2f}")
