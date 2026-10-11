#!/bin/bash
# 256K measurements (256K addendum): server with the env of a config JSON (as bench/ab_screen.sh), then for each
# ctx one prefill step (1 greedy run, 16 tokens: fills the prefix cache, 131K -> 196K -> 262K shared prefixes) and one
# measure step (3 runs x 512 tokens, greedy + default), each a separate <=600 s run . Prompts: bench/prompts_256k.py
# (frozen client bench/bench_http.py, rows in RESULTS.csv). The caller raises the wired limit (platform-specific sysctl, see docs/METHODOLOGY.md) and keeps memguard on.
# Usage: bench/run_256k.sh <cfg.json> <exp> [ctx ...]   (default: 196608 262144; extra server env via SERVER_ENV="VAR=val ...")
W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
cfg=$1 exp=$2; shift 2; ctxs=${*:-196608 262144}
PY=${PYTHON:-runtimes/venv-mlx/bin/python}
args=(); while IFS= read -r l; do [ -n "$l" ] && args+=("$l"); done < <($PY -c "import json,sys; e=json.load(open(sys.argv[1]))['env']; e.pop('OMLX_EXTRA',None); print('\n'.join(f'{k}={v}' for k,v in e.items()))" "$cfg")
extra=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('OMLX_EXTRA',''))" "$cfg")
scripts/kill_run.sh server-omlx >/dev/null
for i in $(seq 60); do pgrep -f "omlx-server|llama-server" >/dev/null || break; sleep 1; done; sleep 3   # no double residency
( OMLX_EXTRA="$extra" env "${args[@]}" ${SERVER_ENV:-} scripts/run.sh server-omlx 1500 "logs/${exp}_server.log" -- configs/omlx_q6.sh & )
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 2; done
SP=$(pgrep -f omlx-server | head -1)
common=(--model coder-q6 --variant Q6-oQ6e --runtime "$(basename "$cfg" .json)" --mtp on --spec lightning-mtp --server-pid "$SP" --warmup 0)
for ctx in $ctxs; do
  echo "=== $exp ctx $ctx prefill $(date +%T)"
  scripts/run.sh bench-256k 600 "logs/${exp}_bench.log" -- $PY bench/prompts_256k.py bench "${common[@]}" --ctx "$ctx" --runs 1 \
    --modes greedy --gen 16 --exp "${exp}W" --flags "256k prefill $(basename "$cfg")" || { echo "prefill $ctx failed"; break; }
  echo "=== $exp ctx $ctx measure $(date +%T)"
  scripts/run.sh bench-256k 600 "logs/${exp}_bench.log" -- $PY bench/prompts_256k.py bench "${common[@]}" --ctx "$ctx" --runs 3 \
    --modes greedy default --gen 512 --exp "$exp" --flags "256k $(basename "$cfg")" --save "logs/${exp}_${ctx}.json" || { echo "measure $ctx failed"; break; }
done
grep -E "^ctx|^==" "logs/${exp}_bench.log"
$PY scripts/mtp_stats.py "logs/${exp}_server.log" | tail -${STATN:-4}
scripts/kill_run.sh server-omlx >/dev/null
