#!/usr/bin/env python3
"""Compare two run_quality.py result files: PASS/FAIL per part with the thresholds of quality/THRESHOLDS.md.

Usage: compare.py ref.json cand.json [--profile lossless|numeric] [--thresholds quality/THRESHOLDS.md]
Exit code: 0 = all compared parts PASS, 1 = at least one FAIL, 2 = inputs not comparable.
A part missing from either file is reported as SKIP (not a failure) -- check the printed table.
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_quality import first_div  # noqa: E402

KEY = dict(coding=lambda i: i["task_id"], reasoning=lambda i: i["index"], tools=lambda i: i["id"],
           needle=lambda i: (i["ctx"], i["depth"]), identity=lambda i: i["id"])


def load_thresholds(path):
    """The first fenced ```json block of THRESHOLDS.md."""
    m = re.search(r"```json\s*\n(.*?)\n```", open(path).read(), re.S)
    if not m:
        raise SystemExit(f"no ```json block in {path}")
    return json.loads(m.group(1))


class NotComparable(Exception):
    pass


def _items(doc, part):
    return {KEY[part](i): i for i in doc["parts"][part]["items"]}


def compare_part(part, ref, cand, rule):
    """-> (verdict, detail). Raises NotComparable when the item sets differ."""
    ri, ci = _items(ref, part), _items(cand, part)
    if ri.keys() != ci.keys():
        raise NotComparable(f"{part}: item sets differ (suite/subset mismatch): "
                            f"{sorted(map(str, ri.keys() ^ ci.keys()))[:6]}")
    if part == "identity":
        bad = []
        for k in ri:
            rt, ct = ri[k].get("text") or "", ci[k].get("text") or ""
            if not rt or not ct or ri[k].get("error") or ci[k].get("error"):   # failed/empty generations never count as identical
                bad.append(f"{k}: EMPTY/ERROR"); continue
            d = first_div(rt, ct)
            if d is not None:
                bad.append(f"{k}: DIVERGE at char {d}")
        ident = len(ri) - len(bad)
        detail = f"{ident}/{len(ri)} IDENTICAL" + (" -- " + "; ".join(bad) if bad else "")
        if rule.get("informational"):
            return "INFO", detail
        return ("PASS" if ident >= rule.get("min_identical", len(ri)) else "FAIL"), detail
    if part == "needle":
        lost = [k for k in ri if ri[k]["passed"] and not ci[k]["passed"]]
        gained = [k for k in ri if not ri[k]["passed"] and ci[k]["passed"]]
        detail = f"ref {sum(i['passed'] for i in ri.values())}/{len(ri)}, cand {sum(i['passed'] for i in ci.values())}/{len(ci)}" \
                 + (f", LOST {lost}" if lost else "") + (f", gained {gained}" if gained else "")
        return ("PASS" if len(lost) <= rule.get("max_lost", 0) else "FAIL"), detail
    rp = sum(i["passed"] for i in ri.values())
    cp = sum(i["passed"] for i in ci.values())
    flips = [f"{k}:{'pass->FAIL' if ri[k]['passed'] else 'fail->pass'}" for k in ri if ri[k]["passed"] != ci[k]["passed"]]
    ok = cp >= rp - rule.get("max_drop", 0) and (not rule.get("exact") or cp == rp)
    detail = f"ref {rp}/{len(ri)}, cand {cp}/{len(ci)} (delta {cp - rp:+d}; allowed drop {rule.get('max_drop', 0)}" \
             + (", must be equal" if rule.get("exact") else "") + ")" + (f"; flips: {', '.join(flips)}" if flips else "")
    return ("PASS" if ok else "FAIL"), detail


def compare(ref, cand, rules):
    """rules: {part: rule}. Returns (rows[(part, verdict, detail)], overall_ok)."""
    for k in ("suite",):
        if ref.get(k) != cand.get(k):
            raise NotComparable(f"{k} differs: {ref.get(k)} vs {cand.get(k)}")
    rows = []
    for part in KEY:
        if part not in ref.get("parts", {}) or part not in cand.get("parts", {}):
            rows.append((part, "SKIP", "missing in " + ("ref" if part not in ref.get("parts", {}) else "cand")))
        elif part not in rules:
            rows.append((part, "SKIP", "no rule in THRESHOLDS.md"))
        else:
            rows.append((part, *compare_part(part, ref, cand, rules[part])))
    judged = [r for r in rows if r[1] in ("PASS", "FAIL")]
    return rows, bool(judged) and all(r[1] == "PASS" for r in judged)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref"); ap.add_argument("cand")
    ap.add_argument("--profile", default="lossless", help="profile name in THRESHOLDS.md (lossless | numeric)")
    ap.add_argument("--thresholds", default=os.path.join(HERE, "THRESHOLDS.md"))
    a = ap.parse_args()
    thr = load_thresholds(a.thresholds)
    if a.profile not in thr["profiles"]:
        print(f"unknown profile {a.profile}; available: {list(thr['profiles'])}", file=sys.stderr); return 2
    ref, cand = json.load(open(a.ref)), json.load(open(a.cand))
    try:
        rows, ok = compare(ref, cand, thr["profiles"][a.profile])
    except NotComparable as e:
        print(f"NOT COMPARABLE: {e}", file=sys.stderr); return 2
    print(f"profile={a.profile} suite={cand.get('suite')} ref={ref.get('model')} cand={cand.get('model')}")
    for part, verdict, detail in rows:
        print(f"  {part:10s} {verdict:5s} {detail}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
