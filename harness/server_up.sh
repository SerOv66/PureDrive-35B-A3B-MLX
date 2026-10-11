#!/bin/bash
# Starts the oMLX server with the certified cand5 configuration (same procedure as bench/screen.sh). Usage: harness/server_up.sh <log>
# Waits for readiness (port 8000). Stop with scripts/kill_run.sh server-omlx.
# Env: PYTHON (default: python3; any interpreter with the json module).
W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
log=${1:-logs/harness_server.log}
PY="${PYTHON:-python3}"
curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null && { echo "port 8000 is busy"; exit 1; }
[ -z "$(ls logs/run 2>/dev/null)" ] || { echo "logs/run is not empty"; exit 1; }
args=(); while IFS= read -r l; do [ -n "$l" ] && args+=("$l"); done < <("$PY" -c "import json; e=json.load(open('configs/q6_omlx_cand5.json'))['env']; e.pop('OMLX_EXTRA',None); print('\n'.join(f'{k}={v}' for k,v in e.items()))")
OMLX_EXTRA="$("$PY" -c "import json; print(json.load(open('configs/q6_omlx_cand5.json'))['env'].get('OMLX_EXTRA',''))")"
# same sampling for every harness (neither sends any): the benchmark's "default" profile (T 0.6, top_p 0.95, top_k 20)
OMLX_EXTRA="${OMLX_EXTRA}, \"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20"
export OMLX_EXTRA
( env "${args[@]}" scripts/run.sh server-omlx 1500 "$log" -- configs/omlx_q6.sh & )
for i in $(seq 120); do sleep 3; curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q coder-q6 && { echo "ready after $((i*3))s"; exit 0; }; done
echo "readiness timeout"; exit 2
