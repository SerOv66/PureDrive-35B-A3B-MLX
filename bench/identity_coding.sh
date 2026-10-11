#!/bin/bash
# Identity of a candidate config: coding 57/57 MTP-on (env) vs MTP-off (baseline_env), CACHE=0.
# Usage: bench/identity_coding.sh <cfg.json> <outdir> [reference coding json to compare the baseline with]
W=${REPO_DIR:-$PWD}; cd "$W"
CFG=${1:?config}; O=${2:?outdir}; mkdir -p $O; PY=${PYTHON:-runtimes/venv-mlx/bin/python}
wait_ready() { for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 2; done; return 1; }
wait_gone() { for i in $(seq 90); do ls logs/run/ | grep -q . || { curl -s -m 1 http://127.0.0.1:8000/v1/models >/dev/null || return 0; }; sleep 1; done; return 1; }
envs() { $PY -c "import json,sys; e=json.load(open(sys.argv[2]))[sys.argv[1]]; e.pop('OMLX_EXTRA',None); print('\n'.join(f'{k}={v}' for k,v in e.items()))" $1 $CFG; }
run_side() {  # $1 = env|baseline_env, $2 = tag
  local a=(); while IFS= read -r l; do [ -n "$l" ] && a+=("$l"); done < <(envs $1)
  echo "##### $2 $(date +%T)"
  ( [ $1 = env ] && a+=('OMLX_EXTRA="mtp_fixed_depth": 2'); env "${a[@]}" CACHE=0 scripts/run.sh server-omlx 1500 $O/server_$2.log -- configs/omlx_q6.sh & )
  sleep 3; wait_ready || echo "server not ready"
  scripts/run.sh bench-identity-q 600 $O/coding_$2.log -- $PY quality/run_quality.py --url http://127.0.0.1:8000 --model coder-q6 --part coding --suite quick --out $O/coding_$2.json
  scripts/kill_run.sh server-omlx; wait_gone; sleep 8
}
run_side baseline_env base; run_side env cand
echo "##### identity cand vs base"; $PY bench/audit/cmp_coding.py $O/coding_base.json:$O/coding_cand.json 2>&1 | tail -3
[ -n "$3" ] && { echo "##### info: base vs $3"; $PY bench/audit/cmp_coding.py $3:$O/coding_base.json 2>&1 | tail -2; }
grep -h "c35 GQA attention engaged" $O/server_cand.log | sort | uniq -c | head
echo "##### IDENTITY DONE $(date +%T)"
