#!/bin/bash
# Usage: check.sh [workdir] — exit 0 = success. Works on a temporary copy and restores the seed's tests/data (the agent cannot "win" by editing them).
T="$(cd "$(dirname "$0")" && pwd)"; D="${1:-$T/../../work/$(basename "$T")}"
PY="${HARNESS_PYTHON:-python3}"   # needs pytest
S="$(mktemp -d)"; cp -R "$D/." "$S/"; rm -rf "$S/.git" "$S/.pytest_cache"
for p in tests data; do [ -d "$T/seed/$p" ] && rm -rf "$S/$p" && cp -R "$T/seed/$p" "$S/$p"; done
cd "$S" && rc=0
"$PY" -m pytest -q -p no:cacheprovider tests || rc=1
grep -rn "calc_total" shop && { echo "leftover references to calc_total"; rc=1; }
grep -q "def compute_total" shop/totals.py || { echo "compute_total not in shop/totals.py"; rc=1; }
grep -q "def calc_total\|def compute_total" shop/pricing.py && { echo "the function is still in pricing.py"; rc=1; }
rm -rf "$S"; exit $rc
