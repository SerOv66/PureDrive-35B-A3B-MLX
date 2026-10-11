#!/bin/bash
# AUDIT EXP-110: are H28 (C35_QK_PREP_MAIN) + K2 (C35_ATTN_OUT) bit-exact in the real server at long context?
# cand4 env (MTP on, BRULE=1 -> same attention numerics), CACHE=0, side A = cand4, side B = cand4 + the two flags.
# Greedy 512 tok at 16300 and 49102: texts must be IDENTICAL. One server at a time (copy of bench/audit/identity_driver.sh).
W=${REPO_DIR:-$PWD}; cd "$W"
A=logs/audit110; PY=${PYTHON:-runtimes/venv-mlx/bin/python}
wait_ready() { for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 2; done; return 1; }
wait_gone() { for i in $(seq 120); do ls logs/run/ | grep -q . || { curl -s -m 1 http://127.0.0.1:8000/v1/models >/dev/null || return 0; }; sleep 1; done; return 1; }
envs() { $PY -c "import json,sys; e=dict(json.load(open('configs/q6_omlx_cand4.json'))['env']); e['CACHE']='0'; print('\n'.join(f'{k}={v}' for k,v in e.items()))"; }
run_side() {  # $1 = tag, rest = extra env
  local tag=$1; shift
  local a=(); while IFS= read -r l; do [ -n "$l" ] && a+=("$l"); done < <(envs)
  a+=("$@"); printf '%s\n' "${a[@]}" > $A/h28k2_env_$tag.txt
  echo "##### $tag $(date +%T) extra: $*"
  env "${a[@]}" scripts/run.sh server-omlx 900 $A/h28k2_server_$tag.log -- configs/omlx_q6.sh &
  sleep 3; wait_ready || { echo "server not ready"; scripts/kill_run.sh server-omlx; wait_gone; return 1; }
  for c in 16300 49102; do
    scripts/run.sh bench-aud110-h28 600 $A/h28k2_thr${c}_$tag.log -- $PY bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx $c --runs 1 --modes greedy --gen 512 --warmup 0 --exp AUD110-h28 --variant Q6-oQ6e --runtime audit --results /dev/null --save $A/h28k2_thr${c}_$tag.json
    echo "thr$c rc=$? $(date +%T)"
  done
  scripts/kill_run.sh server-omlx; wait_gone; echo "server $tag gone rc=$? $(date +%T)"; sleep 8
}
run_side base
run_side flags C35_QK_PREP_MAIN=1 C35_ATTN_OUT=1
for c in 16300 49102; do
  $PY -c "
import json; a, b = (json.load(open(f'$A/h28k2_thr${c}_{s}.json')) for s in ('base', 'flags'))
for k in a: print('thr$c', k, 'IDENTICAL' if a[k] == b.get(k) else 'DIFF', len(a[k]))"
done
grep -h "fused q/k prep engaged\|fused attention output engaged" $A/h28k2_server_flags.log | sed 's/^.* - INFO - //' | sort | uniq -c
echo "##### DONE $(date +%T)"
