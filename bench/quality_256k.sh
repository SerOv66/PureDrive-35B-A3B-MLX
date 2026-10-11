#!/bin/bash
# 256K long-context quality (EXP-107): server cand4 + HOT_MAX=0, then quality/longctx.py steps at --ctxs 262144, each <=600 s.
# Usage: [CTX=262144 OUT=... ONLY=CTX/TYPE] bench/quality_256k.sh <tag> <step> [<step> ...]   step = prefill:N | cases
cd "${REPO_DIR:-$PWD}"; tag=$1; shift
PY=${PYTHON:-runtimes/venv-mlx/bin/python}; cfg=configs/q6_omlx_cand4.json
args=(); while IFS= read -r l; do [ -n "$l" ] && args+=("$l"); done < <($PY -c "import json,sys; e=json.load(open(sys.argv[1]))['env']; e.pop('OMLX_EXTRA',None); print('\n'.join(f'{k}={v}' for k,v in e.items()))" "$cfg")
extra=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('OMLX_EXTRA',''))" "$cfg")
scripts/kill_run.sh server-omlx >/dev/null; sleep 3
( OMLX_EXTRA="$extra" env "${args[@]}" HOT_MAX=0 scripts/run.sh server-omlx 1500 "logs/${tag}_server.log" -- configs/omlx_q6.sh & )
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 2; done
Q=(quality/longctx.py --url http://127.0.0.1:8000 --model coder-q6 --label cand4-bf16 --ctxs ${CTX:-262144} --out ${OUT:-quality/results/longctx_256k_cand4.json})
for st in "$@"; do
  echo "=== $tag $st $(date +%T)"
  case $st in
    prefill:*) scripts/run.sh quality-256k 600 "logs/${tag}_quality.log" -- $PY "${Q[@]}" --prefill-to "${st#prefill:}" ;;
    cases) scripts/run.sh quality-256k 600 "logs/${tag}_quality.log" -- $PY "${Q[@]}" --budget 540 ${ONLY:+--only $ONLY} ;;
  esac
  echo "rc $?"
done
scripts/kill_run.sh server-omlx >/dev/null
