#!/bin/bash
# Builds tasks/t5_long/seed/lib from a CPython 3.13.14 standard library (nine modules, PSF license; not shipped here)
# and plants the one-line textwrap bug the task asks the agent to find. Verifies the result against seed_lib.sha256.
# Usage: CPYTHON313_STDLIB=/path/to/lib/python3.13 build_seed.sh
T="$(cd "$(dirname "$0")" && pwd)"; S="${CPYTHON313_STDLIB:?set CPYTHON313_STDLIB to the lib/python3.13 directory of CPython 3.13.14}"
mkdir -p "$T/seed/lib"
for m in argparse optparse configparser difflib textwrap shlex glob fnmatch getopt; do cp "$S/$m.py" "$T/seed/lib/$m.py"; done
python3 - "$T/seed/lib/textwrap.py" <<'PY'
import sys
p = sys.argv[1]; lines = open(p).read().split("\n")
a = "            width = self.width - len(indent)"
assert lines.count(a) == 1
lines[lines.index(a)] = a + " - (1 if lines else 0)"
open(p, "w").write("\n".join(lines))
PY
(cd "$T/seed/lib" && shasum -a 256 -c "$T/seed_lib.sha256") || { echo "seed_lib.sha256 mismatch: wrong CPython version?"; exit 1; }
