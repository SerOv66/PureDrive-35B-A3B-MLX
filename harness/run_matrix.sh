#!/bin/bash
# Matrix harness x task x repetitions (interleaved order). Restarts the server before the scripts/run.sh budget (1500 s) expires.
# Usage: run_matrix.sh "<harness...>" "<task...>" "<rep...>" [cap_s]
R="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$R/.." && pwd)"; cd "$ROOT"
CAP=${4:-900}; SS=0; N=0
for rep in $3; do for t in $2; do for h in $1; do
  alive=$(curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -c coder-q6)
  if [ "$alive" != 1 ] || [ $(( $(date +%s) - SS )) -gt 1150 ]; then
    scripts/kill_run.sh server-omlx >/dev/null; sleep 3
    N=$((N+1)); "$R/server_up.sh" "logs/harness_server_b$(date +%H%M).log" || { echo "server does not start"; exit 3; }; SS=$(date +%s)
  fi
  "$R/run_one.sh" "$h" "$t" "$rep" "$CAP"
done; done; done
scripts/kill_run.sh server-omlx >/dev/null; echo MATRIX_DONE
