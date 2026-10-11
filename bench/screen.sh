#!/bin/bash
# Quick screening : start configs/omlx_q6.sh with the given env, 1 greedy run of 512 tok at 2K/32K/128K
# (+ default sampling if MODES set), stop the server, print MTP cycle stats. Total <= 10 min with a warm prefix cache.
# Usage: bench/screen.sh <exp> <runtime-label> VAR=val ...   (env vars passed to the server)
W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
exp=$1 label=$2; shift 2
scripts/kill_run.sh server-omlx >/dev/null
for i in $(seq 60); do pgrep -f "omlx-server|llama-server" >/dev/null || break; sleep 1; done; sleep 3   # no double residency
( env "$@" scripts/run.sh server-omlx 1500 "logs/${exp}_server.log" -- configs/omlx_q6.sh & )
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 2; done
SP=$(pgrep -f omlx-server | head -1)
scripts/run.sh bench-screen 600 "logs/${exp}_screen.log" -- ${PYTHON:-runtimes/venv-mlx/bin/python} bench/bench_http.py --model coder-q6 \
  --ctx ${CTXS:-2048 32768 131072} --runs ${RUNS:-1} --modes ${MODES:-greedy} --gen 512 --exp "$exp" --variant Q6-oQ6e --runtime "$label" \
  --flags "screen $*" --mtp on --spec lightning-mtp --server-pid "$SP" --save "logs/${exp}_screen.json"
grep -E "^ctx|^==" "logs/${exp}_screen.log"
${PYTHON:-runtimes/venv-mlx/bin/python} scripts/mtp_stats.py "logs/${exp}_server.log" | tail -${STATN:-4}
scripts/kill_run.sh server-omlx >/dev/null
