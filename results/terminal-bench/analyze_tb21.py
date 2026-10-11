#!/usr/bin/env python3
"""Terminal-Bench 2.1 analysis. Stdlib only. Usage: python3 analyze_tb21.py [csv ...] (default: tb21_results.csv next to this file)
Per arm: last row per task counts (an infra row is superseded by its single rerun); blank reward = fail.
pass@1 with Wilson 95% CI, overall and without custom-memory-heap-crash (gdb cannot attach or read core dumps under x86-64 emulation)."""
import csv, math, os, statistics, sys

BP = os.path.dirname(os.path.abspath(__file__))
EXCL = "custom-memory-heap-crash"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def final_rows(path):
    last = {}
    for r in csv.DictReader(open(path)):
        last[r["task"]] = r  # rows are appended in order: the rerun (attempt 2) overrides the infra row
    return last


def fmt(k, n):
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% (95%% CI %.1f–%.1f)" % (k, n, 100 * k / n if n else 0, 100 * lo, 100 * hi)


def report(path):
    rows = final_rows(path)
    passed = {t for t, r in rows.items() if r["pass"] == "1"}
    infra = [t for t, r in rows.items() if "infra" in r["notes"].split(";")[0]]
    out = ["### %s" % os.path.basename(path), "",
           "- tasks finished: %d (infra-tainted final rows: %s)" % (len(rows), ", ".join(infra) or "none"),
           "- pass@1: " + fmt(len(passed), len(rows))]
    if EXCL in rows:
        out.append("- pass@1 without %s: %s" % (EXCL, fmt(len(passed - {EXCL}), len(rows) - 1)))
    num = lambda c: [float(r[c]) for r in rows.values() if r[c]]
    for c, lab in (("wall_s", "wall s"), ("turns", "turns"), ("decode_tps", "decode tok/s"), ("prompt_tokens", "prompt tokens")):
        v = num(c)
        if v:
            out.append("- %s: median %.0f, range %.0f–%.0f" % (lab, statistics.median(v), min(v), max(v)))
    out += ["", "| task | pass | wall s | turns | tok/s |", "|---|---|---|---|---|"]
    out += ["| %s | %s | %s | %s | %s |" % (t, r["pass"] or "fail (no reward)", r["wall_s"], r["turns"], r["decode_tps"]) for t, r in rows.items()]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    lo, hi = wilson(17, 30)  # self-check against a known value: 17/30 -> 39.2%–72.6% (hand-computed)
    assert abs(lo - 0.392) < 1e-3 and abs(hi - 0.726) < 1e-3, (lo, hi)
    paths = sys.argv[1:] or [os.path.join(BP, f) for f in ("tb21_results.csv",)]
    print("\n".join(report(p) for p in paths if os.path.exists(p)))
