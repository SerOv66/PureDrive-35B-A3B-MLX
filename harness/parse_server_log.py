#!/usr/bin/env python3
"""Extracts from the oMLX server log the requests in a time window: tokens, time, tok/s, prompt, prefix-cache hit
(line 'Prefix cache restore ... cached=N suffix=M'), MTP statistics (tokens/cycle, acceptance).
Usage: parse_server_log.py <log> [--since 'YYYY-MM-DD HH:MM:SS'] [--until '...'] [--json]"""
import re, sys, json
log = sys.argv[1]; arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
since, until = arg("--since") or "", arg("--until") or "9999"
rx_cc = re.compile(r"^(\S+ \S+),\d+ - omlx.server - INFO - Chat completion: .*?(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), prompt: (\d+), finish_reason=(\w+)")
rx_pc = re.compile(r"Prefix cache restore .*?cached=(\d+) suffix=(\d+)")
rx_mtp = re.compile(r"MTP\[\d+\] finish=(\w+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+)")
out, cached, mtp = [], 0, None
for line in open(log, errors="replace"):
    if (m := rx_pc.search(line)): cached = int(m[1])
    elif (m := rx_mtp.search(line)): mtp = (float(m[4]), int(m[5]) / max(1, int(m[6])))
    elif (m := rx_cc.match(line)):
        if since <= m[1] <= until:
            out.append(dict(t=m[1], tokens=int(m[2]), secs=float(m[3]), tps=float(m[4]), prompt=int(m[5]), finish=m[6], cached=cached,
                            tok_cycle=mtp[0] if mtp else None, accept=round(mtp[1], 3) if mtp else None))
        cached, mtp = 0, None
if "--json" in sys.argv: print(json.dumps(out)); sys.exit()
for r in out: print(f'{r["t"]} {r["tokens"]:5}tok {r["secs"]:6.2f}s {r["tps"]:6.1f}tok/s(incl. prefill) prompt={r["prompt"]:6} cached={r["cached"]:6} tok/cycle={r["tok_cycle"]} acc={r["accept"]} {r["finish"]}')
if out:
    T = sum(r["tokens"] for r in out); S = sum(r["secs"] for r in out)
    print(f'TOTAL requests={len(out)} token={T} seconds={S:.1f} mean tok/s={T/S:.1f} prompt={sum(r["prompt"] for r in out)} cached={sum(r["cached"] for r in out)}')
