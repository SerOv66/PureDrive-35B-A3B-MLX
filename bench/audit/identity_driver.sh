#!/bin/bash
# EXP-110 identity (copy of the AUDIT EXP-092 driver): one server at a time, CACHE=0. Base = baseline_env of configs/q6_omlx_cand5_wip.json (MTP off, BRULE=2),
# cand = env (MTP on). Per side: coding quick (57) + greedy 512 tok on real prompts whose generation crosses changes of the BRULE=2
# split size: 8192 (128->208, 3 tasks), 16640 (416->432), 49280 (1232->1248), 110592 (k 1->2: 2768->1392) -> verify windows straddling a change.
W=${REPO_DIR:-$PWD}; cd "$W"
A=logs/EXP-110; PY=${PYTHON:-runtimes/venv-mlx/bin/python}
wait_ready() { for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 2; done; return 1; }
wait_gone() { for i in $(seq 120); do ls logs/run/ | grep -q . || { curl -s -m 1 http://127.0.0.1:8000/v1/models >/dev/null || return 0; }; sleep 1; done; return 1; }
envs() { $PY -c "import json,sys; e=dict(json.load(open('configs/q6_omlx_cand5_wip.json'))[sys.argv[1]]); e['CACHE']='0'; print('\n'.join(f'{k}={v}' for k,v in e.items()))" $1; }
run_side() {  # $1 = env|baseline_env, $2 = tag
  local a=(); while IFS= read -r l; do [ -n "$l" ] && a+=("$l"); done < <(envs $1)
  printf '%s\n' "${a[@]}" > $A/env_$2.txt
  echo "##### $2 $(date +%T) env: ${a[*]}"
  env "${a[@]}" scripts/run.sh server-omlx 1500 $A/server_$2.log -- configs/omlx_q6.sh &
  sleep 3; wait_ready || { echo "server not ready"; scripts/kill_run.sh server-omlx; wait_gone; return 1; }
  scripts/run.sh bench-exp110-q 600 $A/coding_$2.log -- $PY quality/run_quality.py --url http://127.0.0.1:8000 --model coder-q6 --part coding --suite quick --out $A/coding_$2.json
  echo "coding rc=$? $(date +%T)"
  scripts/run.sh bench-exp110-thr 600 $A/thr8k_$2.log -- $PY bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx 8102 --runs 3 --modes greedy --gen 512 --warmup 0 --exp EXP110-thr --variant Q6-oQ6e --runtime audit --results /dev/null --save $A/thr8k_$2.json
  echo "thr8k rc=$? $(date +%T)"
  scripts/run.sh bench-exp110-thr 600 $A/thr16k_$2.log -- $PY bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx 16300 --runs 1 --modes greedy --gen 512 --warmup 0 --exp EXP110-thr --variant Q6-oQ6e --runtime audit --results /dev/null --save $A/thr16k_$2.json
  echo "thr16k rc=$? $(date +%T)"
  scripts/run.sh bench-exp110-thr 600 $A/thr49k_$2.log -- $PY bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx 49102 --runs 1 --modes greedy --gen 512 --warmup 0 --exp EXP110-thr --variant Q6-oQ6e --runtime audit --results /dev/null --save $A/thr49k_$2.json
  echo "thr49k rc=$? $(date +%T)"
  scripts/run.sh bench-exp110-thr 600 $A/thr110k_$2.log -- $PY bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx 110300 --runs 1 --modes greedy --gen 512 --warmup 0 --exp EXP110-thr --variant Q6-oQ6e --runtime audit --results /dev/null --save $A/thr110k_$2.json
  echo "thr110k rc=$? $(date +%T)"
  scripts/kill_run.sh server-omlx; wait_gone; echo "server $2 gone rc=$? $(date +%T)"; sleep 8
  memory_pressure | tail -1
}
run_side baseline_env base
run_side env cand
echo "##### DONE $(date +%T)"
