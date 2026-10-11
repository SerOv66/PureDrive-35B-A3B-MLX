#!/usr/bin/env python3
"""Extract the measurement data from a verifier output JSON (logs/verify/<ts>.json written by bench/verify_target.py).

Usage: python extract.py <verify.json> [<verify.json> ...]   ->  writes <ts>_measurements.json next to each input.

Kept unchanged: timestamp, configuration (command, env, baseline env), the per-cell decode medians ("speed", tok/s: matrix cells
"<ctx>/<mode>", "constancy/..." and "stability/..."), the quality-gate and identity reports, the identity-at-32K result,
thermal pressure, swap growth, duration, and the sha256 of the source file.
Also removed, wherever they appear: server epoch timestamps (integer "created" fields copied from the /v1/models response), so that no
local-versus-UTC offset can be derived; wall-clock strings are kept.
Omitted: the verifier's overall verdict and the list of failure reasons (which, in the published runs, consisted of per-cell
comparisons of the medians with an optional speed threshold). The per-run values behind every median are in the matching
<ts>_matrix.csv / _constancy.csv / _stability.csv files. No number is modified.
"""
import hashlib, json, os, sys

DROP = {"fail", "verdict"}


def no_epoch(o):
    """Recursively drop integer epoch 'created' fields (server timestamps)."""
    if isinstance(o, dict):
        return {k: no_epoch(v) for k, v in o.items() if not (k == "created" and isinstance(v, int) and 10**9 < v < 2 * 10**9)}
    if isinstance(o, list):
        return [no_epoch(x) for x in o]
    return o

for path in sys.argv[1:]:
    raw = open(path, "rb").read()
    d = json.loads(raw)
    out = no_epoch({k: v for k, v in d.items() if k not in DROP})
    out["extract_of"] = {"file": os.path.basename(path), "sha256": hashlib.sha256(raw).hexdigest(),
                         "omitted_fields": sorted(k for k in d if k in DROP)}
    dst = os.path.join(os.path.dirname(os.path.abspath(path)), f"{d['ts']}_measurements.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(dst)
