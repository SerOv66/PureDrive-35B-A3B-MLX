"""Summarize oMLX Lightning MTP cycle stats from a server log: per request and median.
Usage: python scripts/mtp_stats.py logs/omlx_server_X.log [--min-tokens 256]
With C35_MTP_PREDRAFT=1 (best since EXP-105) the logged backbone/mtp timings count the draft preparation twice (AUDIT EXP-105
issue 5): backbone/cyc and the derived tok/s are NOT the cycle time; use ms/cycle = 1000 * tok/cyc / decode tok/s of the bench.
tok/cyc and accept stay valid."""
import re, sys, statistics
mt = int(sys.argv[sys.argv.index("--min-tokens") + 1]) if "--min-tokens" in sys.argv else 256
R = re.compile(r'MTP\[(\d+)\] finish=\w+ tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+).*?timing\[backbone=([\d.]+)ms mtp=([\d.]+)ms')
rows = []
for line in open(sys.argv[1]):
    m = R.search(line)
    if m:
        uid, tok, cyc, tpc, acc, prop, bb, mtp = m.groups()
        tok, cyc, bb, mtp = int(tok), int(cyc), float(bb), float(mtp)
        if tok >= mt: rows.append((int(uid), tok, cyc, float(tpc), int(acc) / max(1, int(prop)), bb / cyc, mtp / cyc, tok / ((bb + mtp) / 1000)))
for r in rows: print("uid %3d tok %4d cyc %4d tok/cyc %.2f accept %.2f backbone/cyc %.1fms mtp/cyc %.2fms -> %.1f tok/s" % r)
if rows:
    med = lambda i: statistics.median(r[i] for r in rows)
    print("MEDIAN n=%d tok/cyc %.2f accept %.2f backbone/cyc %.1fms -> %.1f tok/s" % (len(rows), med(3), med(4), med(5), med(7)))
