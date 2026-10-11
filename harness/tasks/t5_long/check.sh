#!/bin/bash
# Long session: textwrap tests + API.md (>=90% of the public names per module). Works on a copy, restores the seed's tests.
T="$(cd "$(dirname "$0")" && pwd)"; D="${1:-$T/../../work/$(basename "$T")}"
PY="${HARNESS_PYTHON:-python3}"   # needs pytest
S="$(mktemp -d)"; cp -R "$D/." "$S/"; rm -rf "$S/.git" "$S/.pytest_cache"; rm -rf "$S/tests"; cp -R "$T/seed/tests" "$S/tests"
cd "$S" && rc=0
gtimeout 120 "$PY" -m pytest -q -p no:cacheprovider tests || rc=1
"$PY" "$T/check_api.py" "$S" || rc=1
rm -rf "$S"; exit $rc
